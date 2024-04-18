from typing import List
import openai
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
import warnings
import instructor
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import lancedb
from pprint import pprint

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


class ResolvedImprovedEntities(BaseModel):
    """An improved list of extracted entities."""
    
    entities: str = Field(
        ...,
        description="Accurately resolved and improved entities useful for downstream retrieval. Avoid filler words that do not add value as search input. For example, 'Liberty Mutual Insurance Company' should simply be 'Liberty Mutual'.",
    )


class MatchedEntitySearch:
    """
    This class provides methods for ingesting data, extracting named entities,
    resolving and refining entities using an LLM, and performing searches based on queries and
    extracted entities. It leverages a pre-trained named entity recognition (NER) model and
    integrates with a LanceDB database for storage and retrieval.

    Example usage:
        >>> search = MatchedEntitySearch(db_path="./.lancedb")
        >>> search.ingest_data(df, table_name="context", mode='overwrite')
        >>> query = "What is the capital of France?"
        >>> search_results = search.search(query, limit=100) # will contain 'France'
    """

    def __init__(self, db_path: str, model_id: str = "dslim/bert-base-NER"):
        """
        Initialize the MatchedEntitySearch instance.

        This method sets up the necessary components for entity search functionality:
        - Initializes the device (GPU if available, else CPU)
        - Loads the tokenizer and model for named entity recognition (NER) using the specified model ID
        - Creates an NER pipeline using the loaded model and tokenizer
        - Connects to the LanceDB database using the provided database path
        - Initializes the table attribute to None

        Args:
            db_path (str): The path to the LanceDB database file.
            model_id (str, optional): The ID of the pre-trained model to use for named entity recognition.
                Defaults to "dslim/bert-base-NER".

        Attributes:
            device (torch.device): The device to use for computations (GPU if available, else CPU).
            tokenizer (AutoTokenizer): The tokenizer for the NER model.
            model (AutoModelForTokenClassification): The pre-trained model for named entity recognition.
            nlp (pipeline): The NER pipeline created using the loaded model and tokenizer.
            db (lancedb.LanceDB): The LanceDB database connection.
            tbl (lancedb.Table): The table attribute, initially set to None.

        Example:
            >>> search = MatchedEntitySearch(db_path="./.lancedb")
        """
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        warnings.filterwarnings("ignore", category=UserWarning)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id)
        self.nlp = pipeline(
            "ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="max", device=self.device
        )

        self.db = lancedb.connect(db_path)
        self.tbl = None
        logger.info(f"Loaded database with tables: {self.db.table_names()}")

    def create_table(self, table_name: str, data: List[dict], mode: str = 'overwrite'):
        """
        Create a new table in the LanceDB database.

        Args:
            table_name (str): The name of the table to create.
            data (List[dict]): The data to be inserted into the table.
            mode (str): The mode for creating the table ('overwrite' or 'append').
        """
        self.tbl = self.db.create_table(table_name, data, mode=mode)
        
    def open_table(self, table_name: str):
        """
        Open an existing table in the LanceDB database.

        Args:
            table_name (str): The name of the table to open.
        """
        self.tbl = self.db.open_table(table_name)

    def ingest_data(self, df: pd.DataFrame, table_name: str, batch_size: int = 10, mode: str = 'overwrite'):
        """
        Ingest data from a DataFrame, extract named entities, resolve them using an LLM, and insert the data into a LanceDB table.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be ingested.
            table_name (str): The name of the table to create or append to.
            batch_size (int): The number of rows to process in each batch.
            mode (str): The mode for creating the table ('overwrite' or 'append').
        """
        data = []

        for i in tqdm(range(0, len(df), batch_size)):
            i_end = min(i + batch_size, len(df))
            batch = df.iloc[i:i_end].copy()

            texts = batch["context"].tolist()
            idx = batch["index"].tolist()

            emb = [self.get_embedding(t) for t in texts]
            entity_batch = self.extract_named_entities(texts)

            resolved_batch = [self.resolve_entities(", ".join(t)) for t in entity_batch]
            processed_entities = [r.entities for r in resolved_batch]
            batch["named_entities"] = processed_entities[0]

            meta = batch.to_dict(orient="records")

            to_upsert = list(zip(idx, emb, meta, batch["named_entities"], batch["context"]))
            for id, emb, meta, entity, text in to_upsert:
                temp = {
                    "vector": np.array(emb),
                    "metadata": meta,
                    "named_entities": entity,
                    "context": text,
                    "id": id,
                }
                data.append(temp)

        if mode == 'overwrite':
            self.create_table(table_name, data, mode=mode)
        elif mode == 'append':
            if table_name in self.db.table_names():
                self.open_table(table_name)
                self.tbl.add(data)
            else:
                self.create_table(table_name, data, mode=mode)

    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for the given text using OpenAI's text-embedding-ada-002 model.

        Args:
            text (str): The input text to get the embedding for.

        Returns:
            List[float]: The embedding vector for the input text.
        """
        client = openai.OpenAI()
        model_name = "text-embedding-ada-002"
        response = client.embeddings.create(input=text, model=model_name)
        return response.data[0].embedding

    @staticmethod
    def deduplicate(seq: List[str]) -> List[str]:
        """
        Remove duplicates from the input list while preserving the order.

        Args:
            seq (List[str]): The input list of strings.

        Returns:
            List[str]: The deduplicated list of strings.
        """
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    def get_entities_by_type_and_score(self, entities: List[dict], entity_types: List[str], score_threshold: float) -> List[str]:
        """
        Get entities of specific types within a certain score threshold.

        Args:
            entities (List[dict]): A list of entity dictionaries.
            entity_types (List[str]): A list of desired entity types (e.g., ['ORG', 'PER', 'LOC']).
            score_threshold (float): The minimum score threshold for entities.

        Returns:
            List[str]: A list of entities matching the specified types and score threshold.
        """
        entity_name_list = [
            entity['word']
            for entity in entities
            if entity['entity_group'] in entity_types and entity['score'] >= score_threshold
        ]
        top_entities_distinct = self.deduplicate(entity_name_list)
        top_entities = [item for item in top_entities_distinct if len(item) > 1]
        return top_entities

    def extract_named_entities(self, text_batch: List[str]) -> List[List[str]]:
        """
        Extract named entities from a batch of text using the NER pipeline.

        Args:
            text_batch (List[str]): A list of text strings to extract entities from.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains the extracted entities for the corresponding text.
        """
        extracted_batch = self.nlp(text_batch)
        entities = []
        for text in extracted_batch:
            ne = self.get_entities_by_type_and_score(text, entity_types=['ORG', 'LOC'], score_threshold=0.985)
            entities.append(ne)
        return entities

    def resolve_entities(self, content: str) -> ResolvedImprovedEntities:
        """
        Resolve and refine the extracted entities using an LLM.

        Args:
            content (str): The input string containing the extracted entities.

        Returns:
            ResolvedImprovedEntities: The resolved and improved entities.
        """
        client = instructor.patch(openai.OpenAI())
        return client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            response_model=ResolvedImprovedEntities,
            messages=[
                {
                    "role": "system",
                    "content": "You are an entity resolution and enhancement AI. Your task is to refine a list of extracted entities by removing, combining, or otherwise improving the text. Make every word count.",
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )  # type: ignore
        
    def search(self, query: str, limit: int = 100, strict: bool = False, verbose: bool = False) -> pd.DataFrame:
        """
        Search the LanceDB table for relevant results based on the query and extracted entities.

        This method performs the following steps:
        1. Extracts named entities from the query using the NER pipeline.
        2. Resolves and refines the extracted entities using an LLM.
        3. Retrieves the embedding for the query.
        4. Searches the LanceDB table for relevant results based on the embedding similarity.
        5. Calculates the match count between the query entities and the entities in each result.
        6. Sorts the results based on the match count and vector similarity.

        Args:
            query (str): The search query.
            limit (int, optional): The maximum number of search results to return. Defaults to 100.
            strict (bool, optional): If True, performs strict matching of entities. Defaults to False.
            verbose (bool, optional): If True, logs additional information. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing the search results, sorted by match count and vector similarity.

        Example:
            >>> query = "What is the capital of France?"
            >>> search_results = search.search(query, limit=100)
            >>> search_results.head() # results will contain 'France'
        """
        ne = self.extract_named_entities([query])
        resolved_batch = [self.resolve_entities(", ".join(t)) for t in ne]
        processed_entities = [r.entities for r in resolved_batch]

        xq = self.get_embedding(query)
        xdf = self.tbl.search(np.array(xq)).limit(limit).to_pandas()
        xdf["score"] = 1 - xdf["_distance"]
        
        res = []
        
        if strict:
            for _, row in xdf.iterrows():
                row_entities = set(row["named_entities"].lower().split(', '))
                query_entities = set(processed_entities[0].lower().split(', '))
                match_count = len(query_entities.intersection(row_entities))
                if match_count > 0:
                    res.append((row['context'], row['id'], match_count))
        else:
            for _, row in xdf.iterrows():
                match_count = sum(1 for i in processed_entities[0].lower().split(', ') if i in row["named_entities"].lower())
                if match_count > 0:
                    res.append((row['context'], row['id'], match_count))

        res.sort(key=lambda x: x[2], reverse=True)

        idx_list = [r[1] for r in res]
        df_out = xdf[xdf["id"].isin(idx_list)].copy()

        df_out["match_count"] = [r[2] for r in res]
        df_out = df_out.sort_values(by=["match_count", "score"], ascending=[False, False])
        if verbose:
            logger.info(f"Found {len(df_out)} matches")
            logger.info(f"Extracted Named Entities: {processed_entities}")
        return df_out


# Example usage
# entity_search = MatchedEntitySearch(db_path="./.lancedb")
# entity_search.ingest_data(df, table_name="context", mode='overwrite')

# query = "What is the capital of France?"
# search_results = entity_search.search(query, limit=100)
# 'Extracted Named Entities': [['France']]
# Results will have 'France' in the `named_entities` column
# Output df will be ordered by 'Count' of entity matches, then vector similarity (1 - distance)