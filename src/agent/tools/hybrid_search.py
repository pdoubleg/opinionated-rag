import os
from typing import Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.agent.tools.semantic_search import SemanticSearch
from src.agent.tools.splade_search import SPLADESparseSearch
from src.embedding_models.models import ColbertReranker
from src.search.base import SearchEngine, SearchEngineConfig, SearchType
from src.search.models import Filter

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


DATA_PATH = "data/splade.parquet"


class HybridSearchConfig(SearchEngineConfig):
    """Configuration for hybrid search engine."""
    
    type: SearchType = SearchType.HYBRID
    data_path: str = DATA_PATH
    text_column: str = "context"
    dense_embedding_column: str = "openai_embeddings"
    sparse_embedding_column: str = "splade_embeddings"


class HybridSearchEngine(SearchEngine):
    """Hybrid search engine."""
    
    def __init__(self, config: HybridSearchConfig = HybridSearchConfig()):
        """
        Initialize the hybrid search engine.

        Args:
            config: Configuration for the hybrid search engine.
            
        Example:
            >>> config = HybridSearchConfig(
            >>>     data_path=DATA_PATH,
            >>>     text_column=TEXT_COLUMN,
            >>>     dense_embedding_column=DENSE_EMBEDDING_COLUMN,
            >>>     sparse_embedding_column=SPARSE_EMBEDDING_COLUMN,
            >>> )
            >>> search_engine = HybridSearchEngine.create(config)
            >>> results = search_engine.query_similar_documents(query, top_k=5)
            
        """
        super().__init__(config)
        self.config: HybridSearchConfig = config


class HybridSearch:
    """
    Hybrid Search class for querying similar documents using dense and sparse embeddings
        followed by reranking with ColbertReranker.

    Args:
        df (pd.DataFrame): Input DataFrame containing the documents.
        text_column (str): Name of the column containing the text data.
        dense_embedding_column (str): Name of the column containing the dense embeddings.
        sparse_embedding_column (str): Name of the column containing the sparse embeddings.

    Raises:
        ValueError: If the specified dense_embedding_column or sparse_embedding_column is not present in the DataFrame.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        dense_embedding_column: str = "openai_embeddings",
        sparse_embedding_column: str = "splade_embeddings",
    ):
        self.df = df
        self.dense_embedding_column = dense_embedding_column
        self.sparse_embedding_column = sparse_embedding_column
        self.text_column = text_column

        load_dotenv()
        self.vector_search = SemanticSearch(
            df=self.df,
            text_column=self.text_column,
            embedding_column=self.dense_embedding_column,
        )
        self.splade_search = SPLADESparseSearch(
            df=self.df,
            text_column=self.text_column,
            embedding_column=self.sparse_embedding_column,
        )
        self.reranker = ColbertReranker(column=self.text_column)

    def query_similar_documents(
        self,
        query: str,
        top_k: int,
        filter_criteria: Optional[dict | Filter] = None,
    ) -> pd.DataFrame:
        """
        Query similar documents using Hybrid Search.

        Args:
            query (str): Query string for searching similar documents.
            top_k (int): Number of top similar documents to retrieve.
            filter_criteria (Optional[dict | Filter]): Filter criteria for the search (default: None).

        Returns:
            pd.DataFrame: DataFrame containing the top similar documents.

        Example:
            >>> hybrid_search = HybridSearch(df, "text", "dense_embeddings", "sparse_embeddings")
            >>> results = hybrid_search.query_similar_documents("example query", top_k=10)
        """
        if filter_criteria is not None:
            if isinstance(filter_criteria, Filter):
                filter_criteria = filter_criteria.where

        adjusted_k = 2 * top_k

        vector_res = self.vector_search.query_similar_documents(
            query=query, top_k=adjusted_k, filter_criteria=filter_criteria
        )
        splade_res = self.splade_search.query_similar_documents(
            query=query, top_k=adjusted_k, filter_criteria=filter_criteria
        )

        vector_res.drop(
            columns=[self.sparse_embedding_column, self.dense_embedding_column],
            inplace=True,
        )
        splade_res.drop(
            columns=[self.sparse_embedding_column, self.dense_embedding_column],
            inplace=True,
        )

        reranked_res = self.reranker.rerank_hybrid(query, vector_res, splade_res)

        return reranked_res.head(top_k)
