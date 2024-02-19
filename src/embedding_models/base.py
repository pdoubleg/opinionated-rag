import logging
from typing import ClassVar, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic_settings import BaseSettings

from src.types import EmbeddingFunction

logging.getLogger("openai").setLevel(logging.ERROR)


class EmbeddingModelsConfig(BaseSettings):
    model_type: str = "openai"
    dims: int = 0
    context_length: int = 512


class EmbeddingModel(ABC):
    """
    Abstract base class for an embedding model.
    """

    @classmethod
    def create(cls, config: EmbeddingModelsConfig) -> "EmbeddingModel":
        from src.embedding_models.models import (
            OpenAIEmbeddings,
            OpenAIEmbeddingsConfig,
            SentenceTransformerEmbeddings,
            SentenceTransformerEmbeddingsConfig,
        )

        if isinstance(config, OpenAIEmbeddingsConfig):
            return OpenAIEmbeddings(config)
        elif isinstance(config, SentenceTransformerEmbeddingsConfig):
            return SentenceTransformerEmbeddings(config)
        else:
            raise ValueError(f"Unknown embedding config: {config.__repr_name__}")

    @abstractmethod
    def embedding_fn(self) -> EmbeddingFunction:
        pass

    @property
    @abstractmethod
    def embedding_dims(self) -> int:
        pass

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        [emb1, emb2] = self.embedding_fn()([text1, text2])
        return float(
            np.array(emb1)
            @ np.array(emb2)
            / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        )


class Reranker(ABC):
    def __init__(self, return_score: str = "relevance"):
        """
        Initialize a reranker to rank the results from multiple searches.

        Parameters
        ----------
        return_score : str, default "relevance"
            Options are "relevance" or "all".
            The type of score to return. If "relevance", will return only the relevance
            score. If "all", will return all scores from the vector and FTS search along
            with the relevance score.
        """
        if return_score not in ["relevance", "all"]:
            raise ValueError("return_score must be either 'relevance' or 'all'")
        self.score = return_score

    @abstractmethod
    def rerank_hybrid(
        self,
        query: str,
        vector_results: Union[pa.Table, pd.DataFrame],
        fts_results: Union[pa.Table, pd.DataFrame],
    ) -> Union[pa.Table, pd.DataFrame]:
        """
        Abstract method to rerank the individual results from multiple searches.

        Parameters
        ----------
        query : str
            The input query.
        vector_results : Union[pa.Table, pd.DataFrame]
            The results from the vector search.
        fts_results : Union[pa.Table, pd.DataFrame]
            The results from the FTS search.

        Returns
        -------
        Union[pa.Table, pd.DataFrame]
            The reranked results.
        """
        pass

    def merge_results(
        self, 
        vector_results: Union[pa.Table, pd.DataFrame], 
        fts_results: Union[pa.Table, pd.DataFrame]
    ) -> Union[pa.Table, pd.DataFrame]:
        """
        Merge the results from the vector and FTS search by concatenating the results and removing duplicates.

        Note:
            This method doesn't take score into account. It'll keep the instance that was encountered first.
            This is designed for rerankers that don't use the score.

        Parameters
        ----------
        vector_results : Union[pa.Table, pd.DataFrame]
            The results from the vector search.
        fts_results : Union[pa.Table, pd.DataFrame]
            The results from the FTS search.

        Returns
        -------
        Union[pa.Table, pd.DataFrame]
            The combined results after merging and deduplication.
        """
        if isinstance(vector_results, pd.DataFrame) and isinstance(fts_results, pd.DataFrame):
            combined = pd.concat([vector_results, fts_results]).drop_duplicates().reset_index(drop=True)
            return combined
        elif isinstance(vector_results, pa.Table) and isinstance(fts_results, pa.Table):
            combined = pa.concat_tables([vector_results, fts_results], promote=True)
            row_id = combined.column("_rowid")
            mask = np.full((combined.shape[0]), False)
            _, mask_indices = np.unique(np.array(row_id), return_index=True)
            mask[mask_indices] = True
            combined = combined.filter(mask=mask)
            return combined
        else:
            raise TypeError("vector_results and fts_results must be of the same type")
