import os
import pandas as pd
import asyncio
import tiktoken
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple

from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_covariates,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)

load_dotenv()

GRAPH_TIMESTAMP = "20240714-004619"
API_KEY = os.environ["GRAPHRAG_API_KEY"]
DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

INPUT_DIR = "src/graphrag/input"
ARTIFACTS_DIR = f"src/graphrag/output/{GRAPH_TIMESTAMP}/artifacts"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
DEFAULT_COMMUNITY_LEVEL = 3


class GraphRAGManager:
    """
    A class to manage GraphRAG operations including data loading, question generation, and search.
    """

    def __init__(
        self,
        community_level: int = DEFAULT_COMMUNITY_LEVEL,
        graph_timestamp: str = GRAPH_TIMESTAMP,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Initialize the GraphRAGManager with necessary configurations and data.

        Args:
            community_level (int): The community level for global search, 0 for base (i.e., community) 3 for top hierarchy level
            graph_timestamp (str): Timestamp for the graph data.
            llm_model (str): Language model to use.
            embedding_model (str): Embedding model to use.
        """
        self.graph_timestamp = graph_timestamp
        self.api_key = os.environ["GRAPHRAG_API_KEY"]
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        self.community_level = community_level
        self.input_dir = INPUT_DIR
        self.artifacts_dir = ARTIFACTS_DIR
        self.lancedb_uri = LANCEDB_URI

        self._load_dataframes()
        self._process_data()

        self.llm = self._setup_llm()
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self.text_embedder = self._setup_text_embedder()

        self.local_context_builder = self._setup_local_context_builder()
        self.local_context_params = self._get_local_context_params()
        self.llm_params = self._get_llm_params()

        self.local_question_generator = self._setup_question_generator()
        self.local_search_engine = self._setup_local_search_engine()

        self.global_context_builder = self._setup_global_context_builder()
        self.global_search_engine = self._setup_global_search_engine()

    def _load_dataframes(self) -> None:
        """Load raw dataframes from parquet files."""
        # Define table names
        community_report_table = "create_final_community_reports"
        entity_table = "create_final_nodes"
        entity_embedding_table = "create_final_entities"
        relationship_table = "create_final_relationships"
        text_unit_table = "create_final_text_units"
        covariate_table = "create_final_covariates" # src\graphrag\output\20240714-004619\artifacts\create_final_covariates.parquet

        # Load dataframes
        self.entity_df = pd.read_parquet(f"{self.artifacts_dir}/{entity_table}.parquet")
        self.entity_embedding_df = pd.read_parquet(
            f"{self.artifacts_dir}/{entity_embedding_table}.parquet"
        )
        self.relationship_df = pd.read_parquet(
            f"{self.artifacts_dir}/{relationship_table}.parquet"
        )
        self.text_unit_df = pd.read_parquet(
            f"{self.artifacts_dir}/{text_unit_table}.parquet"
        )
        self.report_df = pd.read_parquet(
            f"{self.artifacts_dir}/{community_report_table}.parquet"
        )

        # Preprocess text unit dataframe
        self.text_unit_df["document_ids"] = self.text_unit_df["document_ids"].apply(
            lambda x: [str(id) for id in x]
        )
        self.covariate_df = pd.read_parquet(
            f"{self.artifacts_dir}/{covariate_table}.parquet"
        )

    def _process_data(self) -> None:
        """Process loaded dataframes into entities, relationships, text units, and reports."""
        self.entities = read_indexer_entities(
            self.entity_df, self.entity_embedding_df, self.community_level
        )

        # Set up LanceDB vector store
        self.description_embedding_store = LanceDBVectorStore(
            collection_name="entity_description_embeddings",
        )
        self.description_embedding_store.connect(db_uri=self.lancedb_uri)
        store_entity_semantic_embeddings(
            entities=self.entities, vectorstore=self.description_embedding_store
        )

        self.relationships = read_indexer_relationships(self.relationship_df)
        self.text_units = read_indexer_text_units(self.text_unit_df)
        self.reports = read_indexer_reports(
            self.report_df, self.entity_df, self.community_level
        )
        claims = read_indexer_covariates(self.covariate_df)
        self.covariates = {"claims": claims}

        print(f"Processed {len(self.entities):,} entities")
        print(f"Processed {len(self.relationships):,} relationships")
        print(f"Processed {len(self.text_units):,} text units")
        print(f"Processed {len(self.covariates):,} claims")
        print(f"Processed {len(self.reports):,} reports")

    def _setup_llm(self) -> ChatOpenAI:
        """Set up and return a ChatOpenAI instance."""
        return ChatOpenAI(
            api_key=self.api_key,
            model=self.llm_model,
            api_type=OpenaiApiType.OpenAI,
            max_retries=20,
        )

    def _setup_text_embedder(self) -> OpenAIEmbedding:
        """Set up and return an OpenAIEmbedding instance."""
        return OpenAIEmbedding(
            api_key=self.api_key,
            api_type=OpenaiApiType.OpenAI,
            model=self.embedding_model,
            max_retries=20,
        )

    def _setup_local_context_builder(self) -> LocalSearchMixedContext:
        """Set up and return a LocalSearchMixedContext instance."""
        return LocalSearchMixedContext(
            community_reports=self.reports,
            text_units=self.text_units,
            entities=self.entities,
            relationships=self.relationships,
            covariates=self.covariates,
            entity_text_embeddings=self.description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.TITLE,
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )
    

    def _get_local_context_params(self) -> Dict[str, Any]:
        """
        Return local context parameters.
        
        Param Notes:
            text_unit_prop: proportion of context window dedicated to related text units
            community_prop: proportion of context window dedicated to community reports.
            The remaining proportion is dedicated to entities and relationships. Sum of text_unit_prop and community_prop should be <= 1
            conversation_history_max_turns: maximum number of turns to include in the conversation history.
            conversation_history_user_turns_only: if True, only include user queries in the conversation history.
            top_k_mapped_entities: number of related entities to retrieve from the entity description embedding store.
            top_k_relationships: control the number of out-of-network relationships to pull into the context window.
            include_entity_rank: if True, include the entity rank in the entity table in the context window. Default entity rank = node degree.
            include_relationship_weight: if True, include the relationship weight in the context window.
            include_community_rank: if True, include the community rank in the context window.
            return_candidate_context: if True, return a set of dataframes containing all candidate entity/relationship/covariate records that
            could be relevant. Note that not all of these records will be included in the context window. The "in_context" column in these
            dataframes indicates whether the record is included in the context window.
            max_tokens: maximum number of tokens to use for the context window.
        
        """
        return {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 20,
            "top_k_relationships": 30,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": True,
            "embedding_vectorstore_key": EntityVectorStoreKey.TITLE,
            "max_tokens": 12_000,
        }

    def _get_llm_params(self) -> Dict[str, Any]:
        """Return LLM parameters."""
        return {
            "max_tokens": 4_000,
            "temperature": 0.1,
        }

    def _setup_question_generator(self) -> LocalQuestionGen:
        """Set up and return a LocalQuestionGen instance."""
        return LocalQuestionGen(
            llm=self.llm,
            context_builder=self.local_context_builder,
            token_encoder=self.token_encoder,
            llm_params=self.llm_params,
            context_builder_params=self.local_context_params,
        )

    def _setup_local_search_engine(self) -> LocalSearch:
        """Set up and return a LocalSearch instance."""
        return LocalSearch(
            llm=self.llm,
            context_builder=self.local_context_builder,
            token_encoder=self.token_encoder,
            llm_params=self.llm_params,
            context_builder_params=self.local_context_params,
            response_type="structured report",
        )

    def _setup_global_context_builder(self) -> GlobalCommunityContext:
        """Set up and return a GlobalCommunityContext instance."""
        return GlobalCommunityContext(
            community_reports=self.reports,
            entities=self.entities,
            token_encoder=self.token_encoder,
        )

    def _setup_global_search_engine(self) -> GlobalSearch:
        """Set up and return a GlobalSearch instance."""
        global_context_builder_params = {
            "use_community_summary": False,
            "shuffle_data": False,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": 12_000,
            "context_name": "Reports",
        }

        map_llm_params = {
            "max_tokens": 1000,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        reduce_llm_params = {
            "max_tokens": 2000,
            "temperature": 0.0,
        }

        return GlobalSearch(
            llm=self.llm,
            context_builder=self.global_context_builder,
            token_encoder=self.token_encoder,
            max_data_tokens=12_000,
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=False,
            json_mode=True,
            context_builder_params=global_context_builder_params,
            concurrent_coroutines=32,
            response_type="multiple-page report",
        )

    async def generate_questions(
        self,
        question_history: List[str],
        context_data: Dict[str, Any] = None,
        question_count: int = 5,
    ) -> List[str]:
        """
        Generate questions using the local question generator.

        Args:
            question_history (List[str]): List of previous questions.
            context_data (Dict[str, Any], optional): Additional context data. Defaults to None.
            question_count (int, optional): Number of questions to generate. Defaults to 5.

        Returns:
            List[str]: List of generated questions.
        """
        candidate_questions = await self.local_question_generator.agenerate(
            question_history=question_history,
            context_data=context_data,
            question_count=question_count,
        )
        return candidate_questions.response

    async def search_local(self, query: str) -> Any:
        """
        Perform a local search using the local search engine.

        Args:
            query (str): The search query.

        Returns:
            Any: The search results.
        """
        return await self.local_search_engine.asearch(query)

    async def search_global(self, query: str) -> Any:
        """
        Perform a global search using the global search engine.

        Args:
            query (str): The search query.

        Returns:
            Any: The search results.
        """
        return await self.global_search_engine.asearch(query)
