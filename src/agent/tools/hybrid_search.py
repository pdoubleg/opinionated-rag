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
    type: SearchType = SearchType.HYBRID
    data_path: str = DATA_PATH
    text_column: str = "context"
    dense_embedding_column: str = "openai_embeddings"
    sparse_embedding_column: str = "splade_embeddings"


class HybridSearchEngine(SearchEngine):
    def __init__(self, config: HybridSearchConfig = HybridSearchConfig()):
        super().__init__(config)
        self.config: HybridSearchConfig = config


class HybridSearch:
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
