import asyncio
from functools import cached_property
import os
import pandas as pd
import pyarrow as pa
from typing import Callable, List, Tuple

import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel

from src.embedding_models.base import EmbeddingModel, EmbeddingModelsConfig, Reranker
from src.llm.utils import retry_with_exponential_backoff
from src.types import Embeddings
from src.llm.utils import batched


class ColbertReranker(Reranker):
    """
    Reranks the results using pre-trained ColBERTv2 model.

    Parameters
    ----------
    model_name : str, default "colbert-ir/colbertv2.0"
        The name of the encoder model to use.
    column : str, default "content"
        The name of the column to use as input to the cross encoder model.
    return_score : str, default "relevance"
        options are "relevance" or "all". Only "relevance" is supported for now.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        column: str = "content",
        return_score: str = "relevance",
    ):
        self.model_name = model_name
        self.column = column
        self.return_score = return_score
        self.torch = torch

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pd.DataFrame,
        fts_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Reranks the combined results from vector and full-text search results.

        Parameters
        ----------
        query : str
            The search query.
        vector_results : pd.DataFrame
            The results from vector search.
        fts_results : pd.DataFrame
            The results from full-text search.

        Returns
        -------
        pd.DataFrame
            The reranked results as a DataFrame.
        """
        combined_results = (
            pd.concat([vector_results, fts_results])
            .drop_duplicates()
            .reset_index(drop=True)
        )
        docs = combined_results[self.column].tolist()

        tokenizer, model = self._model

        # Encode the query
        query_encoding = tokenizer(query, return_tensors="pt")
        query_embedding = model(**query_encoding).last_hidden_state.mean(dim=1)
        scores = []
        # Get score for each document
        for document in docs:
            document_encoding = tokenizer(
                document, return_tensors="pt", truncation=True, max_length=512
            )
            document_embedding = model(**document_encoding).last_hidden_state
            # Calculate MaxSim score
            score = self.maxsim(query_embedding.unsqueeze(0), document_embedding)
            scores.append(score.item())

        # Add the scores to the DataFrame
        combined_results["_relevance_score"] = scores

        if self.return_score == "relevance":
            combined_results = combined_results.drop(
                columns=["score", "_distance"], errors="ignore"
            )

        combined_results = combined_results.sort_values(
            by="_relevance_score", ascending=False
        )

        return combined_results
    
    def rerank(
        self,
        query: str,
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Reranks the results from a single DataFrame based on their relevance to the query.

        Parameters
        ----------
        query : str
            The search query.
        results_df : pd.DataFrame
            The DataFrame containing the results to be reranked. Must have a 'content' column.

        Returns
        -------
        pd.DataFrame
            The reranked results as a DataFrame, including a new '_relevance_score' column.
        """
        if self.column not in results_df.columns:
            raise ValueError(f"The DataFrame must contain a '{self.column}' column.")

        docs = results_df[self.column].tolist()

        tokenizer, model = self._model

        # Encode the query
        query_encoding = tokenizer(query, return_tensors="pt")
        query_embedding = model(**query_encoding).last_hidden_state.mean(dim=1)
        scores = []

        # Get score for each document
        for document in docs:
            document_encoding = tokenizer(
                document, return_tensors="pt", truncation=True, max_length=512
            )
            document_embedding = model(**document_encoding).last_hidden_state
            # Calculate MaxSim score
            score = self.maxsim(query_embedding.unsqueeze(0), document_embedding)
            scores.append(score.item())

        # Add the scores to the DataFrame
        results_df["_relevance_score"] = scores

        if self.return_score == "relevance":
            results_df = results_df.drop(
                columns=["score", "_distance"], errors="ignore"
            )

        results_df = results_df.sort_values(
            by="_relevance_score", ascending=False
        )

        return results_df

    @cached_property
    def _model(self) -> Tuple[AutoTokenizer, AutoModel]:
        """
        Loads the tokenizer and model for the reranker.

        Returns
        -------
        Tuple[AutoTokenizer, AutoModel]
            The tokenizer and model.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)

        return tokenizer, model

    def maxsim(
        self, query_embedding: torch.Tensor, document_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the maximum similarity score between the query and document embeddings.

        Parameters
        ----------
        query_embedding : torch.Tensor
            The query embedding tensor.
        document_embedding : torch.Tensor
            The document embedding tensor.

        Returns
        -------
        torch.Tensor
            The maximum similarity score.
        """
        expanded_query = query_embedding.unsqueeze(2)
        expanded_doc = document_embedding.unsqueeze(1)

        sim_matrix = self.torch.nn.functional.cosine_similarity(
            expanded_query, expanded_doc, dim=-1
        )
        max_sim_scores, _ = self.torch.max(sim_matrix, dim=2)
        avg_max_sim = self.torch.mean(max_sim_scores, dim=1)

        return avg_max_sim


class OpenAIEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "openai"
    model_name: str = "text-embedding-ada-002"
    api_key: str = ""
    organization: str = ""
    dims: int = 1536
    context_length: int = 8192


class SentenceTransformerEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "sentence-transformer"
    model_name: str = "BAAI/bge-large-en-v1.5"
    context_length: int = 512


class OpenAIEmbeddings(EmbeddingModel):
    def __init__(self, config: OpenAIEmbeddingsConfig = OpenAIEmbeddingsConfig()):
        super().__init__()
        self.config = config
        load_dotenv()
        self.config.api_key = os.getenv("OPENAI_API_KEY", "")
        if self.config.api_key == "":
            raise ValueError(
                """OPENAI_API_KEY env variable must be set to use 
                OpenAIEmbeddings. Please set the OPENAI_API_KEY value 
                in your .env file.
                """
            )

        self.aclient = AsyncOpenAI(api_key=self.config.api_key)
        self.client = OpenAI(api_key=self.config.api_key)
        self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)

    def truncate_texts(self, texts: List[str]) -> List[List[int]]:
        """
        Truncate texts to the embedding model's context length.
        """
        return [
            self.tokenizer.encode(text, disallowed_special=())[
                : self.config.context_length
            ]
            for text in texts
        ]

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        @retry_with_exponential_backoff
        def fn(texts: List[str]) -> Embeddings:
            tokenized_texts = self.truncate_texts(texts)
            embeds = []
            for batch in batched(tokenized_texts, 500):
                result = self.client.embeddings.create(
                    input=batch, model=self.config.model_name
                )
                batch_embeds = [d.embedding for d in result.data]
                embeds.extend(batch_embeds)
            return embeds

        return fn
    
    
    async def aget_embedding(self, text: str):
        response = await self.aclient.embeddings.create(input=text, model=self.config.model_name)
        return response.data[0].embedding


    def apply_async_get_embedding(self, df: pd.DataFrame, text_col: str):
        loop = asyncio.get_event_loop()
        tasks = [loop.create_task(self.aget_embedding(row[text_col])) for _, row in df.iterrows()]
        return loop.run_until_complete(asyncio.gather(*tasks))


    @property
    def embedding_dims(self) -> int:
        return self.config.dims


class SentenceTransformerEmbeddings(EmbeddingModel):
    def __init__(
        self,
        config: SentenceTransformerEmbeddingsConfig = SentenceTransformerEmbeddingsConfig(),
    ):
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                """
                To use, pip install sentence_transformers embeddings.
                """
            )

        super().__init__()
        self.config = config
        self.model = SentenceTransformer(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.config.context_length = self.tokenizer.model_max_length

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        def fn(texts: List[str]) -> Embeddings:
            embeds = []
            for batch in batched(texts, 500):
                batch_embeds = self.model.encode(batch, convert_to_numpy=True).tolist()
                embeds.extend(batch_embeds)
            return embeds

        return fn

    @property
    def embedding_dims(self) -> int:
        dims = self.model.get_sentence_embedding_dimension()
        if dims is None:
            raise ValueError(
                f"Could not get embedding dimension for model {self.config.model_name}"
            )
        return dims  # type: ignore


def embedding_model(embedding_fn_type: str = "openai") -> EmbeddingModel:
    """
    Args:
        embedding_fn_type: "openai" or "sentencetransformer" # others soon
    Returns:
        EmbeddingModel
    """
    if embedding_fn_type == "openai":
        return OpenAIEmbeddings  # type: ignore
    else:  # default sentence transformer
        return SentenceTransformerEmbeddings  # type: ignore
