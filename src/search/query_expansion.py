from abc import ABC, abstractmethod
import enum
import functools
import re
from typing import Any, Iterator, List
import pandas as pd
from pydantic import BaseModel, Field
from tenacity import Retrying, stop_after_attempt, wait_fixed
import instructor
import openai
import enum
from typing import List
import numpy as np
from textwrap import fill

from src.agent.tools.semantic_search import SemanticSearch
from src.agent.tools.splade_search import SPLADESparseSearch
from src.search.base import SearchEngine, SearchEngineConfig, SearchType
from src.search.doc_joiner import DocJoinerDF
from src.search.models import QueryPlanConfig


class SubQuestionList(BaseModel):
    """List of sub-questions related to a high level question"""

    questions: List[str] = Field(
        description="Sub-questions related to the main question."
    )


def generate_subquestions(query: str, n: str = "2 to 3") -> SubQuestionList:
    client = instructor.patch(openai.OpenAI())

    template = f"""
    Your task is to decompose an original user question into {n} distinct atomic sub-questions, \
    such that when they are resolved, the high level question will be answered.
    \n\n# Original question:\n{query}
    """

    return client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_model=SubQuestionList,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "You are a world class query understanding AI.",
            },
            {
                "role": "user",
                "content": template,
            },
        ],
    )


class SearchType(str, enum.Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    SPLADE = "splade"


import enum
import instructor

from typing import List
import numpy as np
from openai import OpenAI
from pydantic import Field, BaseModel
from tenacity import Retrying, stop_after_attempt, wait_fixed
from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

client = instructor.patch(OpenAI())


class SubQuestionList(BaseModel):
    """List of sub-questions related to a high level question"""

    questions: List[str] = Field(
        description="Sub-questions related to the main question."
    )


@functools.cache
def generate_subquestions(query: str, n: str = "2 to 4") -> SubQuestionList:
    client = instructor.patch(openai.OpenAI())

    template = f"""
    Your task is to decompose an original user question into {n} distinct atomic sub-questions, \
    such that when they are resolved, the high level question will be answered.
    \n\n# Original question:\n{query}
    """

    return client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_model=SubQuestionList,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "You are a world class query understanding AI.",
            },
            {
                "role": "user",
                "content": template,
            },
        ],
    )


class Document(BaseModel):
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    metadata: dict = Field(default_factory=dict)
    
    
class BaseLoader(ABC):
    """Interface for Document Loader.

    Implementations should implement the lazy-loading method using generators
    to avoid loading all Documents into memory at once.

    `load` is provided just for user convenience and should not be overridden.
    """

    # Sub-classes should not implement this method directly. Instead, they
    # should implement the lazy load method.
    def load(self) -> List[Document]:
        """Load data into Document objects."""
        return list(self.lazy_load())

    @abstractmethod
    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        if type(self).load != BaseLoader.load:
            return iter(self.load())
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )


class BaseDataFrameLoader(BaseLoader):
    def __init__(self, data_frame: Any, *, page_content_column: str = "text"):
        """Initialize with dataframe object.

        Args:
            data_frame: DataFrame object.
            page_content_column: Name of the column containing the page content.
              Defaults to "text".
        """
        self.data_frame = data_frame
        self.page_content_column = page_content_column

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from dataframe."""

        for _, row in self.data_frame.iterrows():
            text = row[self.page_content_column]
            metadata = row.to_dict()
            metadata.pop(self.page_content_column)
            yield Document(page_content=text, metadata=metadata)


class DataFrameLoader(BaseDataFrameLoader):
    """Load all columns in a `Pandas` DataFrame except embedding vectors."""

    def __init__(self, data_frame: Any, page_content_column: str = "text"):
        """Initialize with dataframe object.

        Args:
            data_frame: Pandas DataFrame object.
            page_content_column: Name of the column containing the page content.
              Defaults to "text".
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Unable to import pandas, please install with `pip install pandas`."
            ) from e

        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError(
                f"Expected data_frame to be a pd.DataFrame, got {type(data_frame)}"
            )
        super().__init__(data_frame, page_content_column=page_content_column)
        
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from dataframe."""

        for _, row in self.data_frame.iterrows():
            text = row[self.page_content_column]
            metadata = {}
            for column, value in row.items():
                if column != self.page_content_column:
                    if not (
                        isinstance(value, (np.ndarray, list))
                        and (
                            all(isinstance(x, float) for x in value)
                            or value.dtype.kind == "f"
                        )
                    ):
                        metadata[column] = value
            yield Document(page_content=text, metadata=metadata)

class SearchType(str, enum.Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    SPLADE = "splade"
    OTHER = "other"
    
    
def dataframe2documents(df: pd.DataFrame, text_column: str = "text") -> List[Document]:
    df["original_index"] = df.index.values
    loader = DataFrameLoader(
        data_frame=df,
        page_content_column=text_column,
    )
    docs = loader.load()
    return docs


def documents2dataframe(documents: List[Document], text_column: str = "text") -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            text_column: chunk.page_content,
            **chunk.metadata,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    if "original_index" in df.columns:
        df.set_index("original_index", inplace=True)
        df.index.name = None
    return df

  
class ChunkMetric(BaseModel):
    source: str
    document_id: str
    score: float
    rank: int


class RetrievalContainer(BaseModel):
    base_question: str | None = None
    sub_question: str | None = None
    sub_question_keywords: str | None = None
    retrieval_docs: List[Document]
    metrics: List[ChunkMetric]
    text_column: str | None = None
    
    @property
    def to_pandas(self):
        if self.text_column is not None:
            return documents2dataframe(self.retrieval_docs, text_column=self.text_column)
        else:
            return documents2dataframe(self.retrieval_docs)
        

def get_chunk_metrics(df: pd.DataFrame, source: Any):
    chunk_metrics = [
        ChunkMetric(
            source=source.__class__.__name__,
            document_id=str(row.name),
            score=round(row["score"], 4),
            rank=rank
        )
        for rank, (_, row) in enumerate(df.iterrows(), start=1)
    ]
    return chunk_metrics


class SubQuestion(BaseModel):
    """
    Class representing a single question or sub-question from the legal insurance domain.

    """

    chain_of_thought: str = Field(
        description="Reasoning behind the question with respect to its role in answering the main question.",
        exclude=True,
    )
    sub_question_topic: str = Field(
        ...,
        description="A concise topic title for the question.",
    )
    sub_question_query: str = Field(
        ...,
        description="A detailed context rich question suitable for vector-based similarity search for relevant documents.",
    )
    sub_question_keywords: List[str] = Field(
        default_factory=list,
        description="Extracted keywords suitable for exact-match text search that support the question or topic. Should be highly specific. Can be technical domain-specific items such as form numbers, formal references, codified rules, citations, etc.",
    )

    def execute(
        self, 
        search_engines: List[SearchEngine],
        config: QueryPlanConfig,
    ) -> pd.DataFrame:
        """
        Executes both vector and keyword searches based on the sub-question query and keywords.

        Returns:
            pd.DataFrame: A DataFrame containing the results from both searches.
        """
        metrics = []
        wrapped_thought = fill(self.chain_of_thought, width=100)
        logger.info(
            f"\n\nThought: {wrapped_thought}\nSearch topic: {self.sub_question_topic}"
        )
        search_results = []
        for search_engine in search_engines:
            if search_engine.__class__.__name__ in ['SPLADESparseSearch', 'FTSSearch']:
                query = ", ".join(self.sub_question_keywords)
            else:
                query = self.sub_question_query
            logger.info(f"Running search using {search_engine.__class__.__name__}: {query}")
            results = search_engine.query_similar_documents(
                query, top_k=config.num_hits,
            )
            search_metrics = get_chunk_metrics(results, search_engine)
            metrics.extend(search_metrics)
            search_results.append(results)

        df_joiner = DocJoinerDF(
            join_mode="reciprocal_rank_fusion", 
            top_k=config.num_rerank,
        )
        hybrid_results = df_joiner.run(
            dataframes=search_results, 
            text_column=search_engines[0].text_column,
        )
        hybrid_metrics = get_chunk_metrics(hybrid_results, df_joiner)
        metrics.extend(hybrid_metrics)
        
        results_container = RetrievalContainer(
            base_question=config.query,
            sub_question=self.sub_question_query,
            sub_question_keywords=", ".join(self.sub_question_keywords),
            retrieval_docs=dataframe2documents(hybrid_results, search_engines[0].text_column),
            metrics=metrics,
            text_column=search_engines[0].text_column,
        )
        logger.info(
            f"Retrieved {', '.join(str(len(results)) for results in search_results)} records from searches"
        )
        logger.info(
            f"Returning {len(hybrid_results)} joined and deduplicated records"
        )
        logger.info("-" * 75)
        return results_container


class MultiSearch(BaseModel):
    """
    Class representing multiple sub-questions that are mutually exclusive and
    collectively exhaustive in relation to an input user question.

    Args:
        questions (List[SubQuestion]): The list of sub-questions.
    """

    questions: List[SubQuestion] = Field(
        ...,
        description="List of questions and searches to perform.",
    )

    def execute(
        self, config: QueryPlanConfig, 
    ) -> pd.DataFrame:
        """Executes a list of searches."""
        search_engines = [SearchEngine.create(config) for config in config.search_configs]
        results = []
        for search in self.questions:
            search_result = search.execute(search_engines, config)
            results.append(search_result)
        return results


def segment_search_query(user_query: str, n: str = "2 to 3") -> MultiSearch:
    """
    Convert a string into multiple search queries.

    Args:
        data (str): The string to convert into search queries.
        n (str): A number range in string format to 'ask' the llm to generate.

    Returns:
        MultiSearch: An object representing multiple search queries.
    """
    single_question = ['0', '1']
    if n in single_question:
        task = "Please re-write the following user query to optimize downstream vector similarity and keyword definition searches. Generate *ONE** sub-question using the correct format, with the optimized question and keywords fully capturing the user query."
    else:
        task = f"Please generate **{n}** sub-questions to help research a user question."
    client = instructor.patch(openai.OpenAI())
    return client.chat.completions.create(
        model="gpt-4-turbo-preview",
        temperature=0.1,
        response_model=MultiSearch,
        max_retries=Retrying(
            stop=stop_after_attempt(5), 
            wait=wait_fixed(1)
        ),
        messages=[
            {
                "role": "system",
                "content": "You are a legal query understanding AI specializing in research optimization. breaking down nuanced legal questions into simple, manageable sub-questions. When presented with a user legal question or research topic, your role is to generate a list of sub-questions that, when answered, will comprehensively address the original query.",
            },
            {
                "role": "user",
                "content": task,
            },
            {   "role": "user", "content": f"Here is the user query: {user_query}"},
        ],
    )
