import logging
from typing import ClassVar
from abc import ABC, abstractmethod

import numpy as np
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

        return_score : str, default "relevance"
            opntions are "relevance" or "all"
            The type of score to return. If "relevance", will return only the relevance
            score. If "all", will return all scores from the vector and FTS search along
            with the relevance score.
        """
        if return_score not in ["relevance", "all"]:
            raise ValueError("score must be either 'relevance' or 'all'")
        self.score = return_score

    @abstractmethod
    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> None:
        """
        Abstract method to rerank the individual results from multiple searches.

        Args:
            query (str): The input query.
            vector_results (pa.Table): The results from the vector search.
            fts_results (pa.Table): The results from the FTS search.
        """
        pass

    def merge_results(
        self, vector_results: pa.Table, fts_results: pa.Table
    ) -> pa.Table:
        """
        Merge the results from the vector and FTS search by concatenating the results and removing duplicates.

        Note:
            This method doesn't take score into account. It'll keep the instance that was encountered first.
            This is designed for rerankers that don't use the score.

        Args:
            vector_results (pa.Table): The results from the vector search.
            fts_results (pa.Table): The results from the FTS search.

        Returns:
            pa.Table: The combined results after merging and deduplication.
        """
        combined = pa.concat_tables([vector_results, fts_results], promote=True)
        row_id = combined.column("_rowid")

        # deduplicate
        mask = np.full((combined.shape[0]), False)
        _, mask_indices = np.unique(np.array(row_id), return_index=True)
        mask[mask_indices] = True
        combined = combined.filter(mask=mask)

        return combined
