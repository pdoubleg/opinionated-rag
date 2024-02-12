from functools import cached_property
import os
import pyarrow as pa
from typing import Callable, List

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.embedding_models.base import EmbeddingModel, EmbeddingModelsConfig, Reranker
from src.llm.utils import retry_with_exponential_backoff
from src.types import Embeddings
from src.llm.utils import batched
from src.utils.gen_utils import safe_import


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
        return_score="relevance",
    ):
        super().__init__(return_score)
        self.model_name = model_name
        self.column = column
        self.torch = safe_import("torch")  # import here for faster ops later

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ):
        combined_results = self.merge_results(vector_results, fts_results)
        docs = combined_results[self.column].to_pylist()

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

        # replace the self.column column with the docs
        combined_results = combined_results.drop(self.column)
        combined_results = combined_results.append_column(
            self.column, pa.array(docs, type=pa.string())
        )
        # add the scores
        combined_results = combined_results.append_column(
            "_relevance_score", pa.array(scores, type=pa.float32())
        )
        if self.score == "relevance":
            combined_results = combined_results.drop_columns(["score", "_distance"])
        elif self.score == "all":
            pass

        combined_results = combined_results.sort_by(
            [("_relevance_score", "descending")]
        )

        return combined_results

    @cached_property
    def _model(self):
        transformers = safe_import("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        model = transformers.AutoModel.from_pretrained(self.model_name)

        return tokenizer, model

    def maxsim(self, query_embedding, document_embedding):
        # Expand dimensions for broadcasting
        # Query: [batch, length, size] -> [batch, query, 1, size]
        # Document: [batch, length, size] -> [batch, 1, length, size]
        expanded_query = query_embedding.unsqueeze(2)
        expanded_doc = document_embedding.unsqueeze(1)

        # Compute cosine similarity across the embedding dimension
        sim_matrix = self.torch.nn.functional.cosine_similarity(
            expanded_query, expanded_doc, dim=-1
        )

        # Take the maximum similarity for each query token (across all document tokens)
        # sim_matrix shape: [batch_size, query_length, doc_length]
        max_sim_scores, _ = self.torch.max(sim_matrix, dim=2)

        # Average these maximum scores across all query tokens
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
