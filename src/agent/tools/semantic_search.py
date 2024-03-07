import asyncio
import os
from dotenv import load_dotenv
import numpy as np
from openai import AsyncOpenAI
import pandas as pd
import faiss
from typing import Dict, List, Tuple, Optional, Any

from pydantic import BaseModel, Field



class Filter(BaseModel):
    where: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="The attribute to filter on and the value to select.",
    )
    name: str = Field(
        ..., 
        description="A display name for the filter.")

    @property
    def off(self):
        return None

    @property
    def display_filter(self) -> str:
        if self.where is not None:
            first_key, first_value = next(iter(self.where.items()))
            s = f"Search Criteria: {first_key.replace('_', ' ').title()} = {first_value}"
        else:
            s = "Search Criteria: All States"
        return s

    @property
    def filter_key(self) -> str:
        if self.where is not None:
            k, _ = next(iter(self.where.items()))
        else:
            k = ""
        return k

    @property
    def filter_value(self) -> str:
        if self.where is not None:
            _, v = next(iter(self.where.items()))
        else:
            v = ""
        return v
    

class SemanticSearch:
    def __init__(self, df: pd.DataFrame, embedding_col_name: str = "embeddings"):
        self.df = df
        self.embedding_col_name = embedding_col_name
        self.model_name="text-embedding-ada-002"
        load_dotenv()
        self.client = AsyncOpenAI()

    async def aget_embedding(self, text: str):
        response = await self.client.embeddings.create(
            input=text, model=self.model_name
        )
        return response.data[0].embedding

    def aget_embedding_df(self, df: pd.DataFrame, text_col: str):
        loop = asyncio.get_event_loop()
        tasks = [
            loop.create_task(self.aget_embedding(row[text_col]))
            for _, row in df.iterrows()
        ]
        return loop.run_until_complete(asyncio.gather(*tasks))
    
    async def encode_string(self, text: str) -> np.ndarray:
        embedding = await self.aget_embedding(text)
        return np.array(embedding)

    # def encode_string(self, text: str) -> np.ndarray:
    #     embedding = self.aget_embedding(text)
    #     return np.array(embedding)

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings.shape) == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        if embeddings.shape[0] == 1:
            embeddings = np.transpose(embeddings)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norm
        return normalized_embeddings

    def check_and_normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        # Calculate the L2 norm for each embedding
        norms = np.linalg.norm(embeddings, axis=1)

        # Check if the norms are close to 1 (with a small tolerance)
        if not np.allclose(norms, 1, atol=1e-6):
            print("Embeddings are not normalized, normalizing now...")
            embeddings = self.normalize_embeddings(embeddings)

        return embeddings

    def build_faiss_index(
        self, embeddings: np.ndarray, use_cosine_similarity: bool = False
    ) -> faiss.Index:
        if use_cosine_similarity:
            # Check and normalize the embeddings if needed
            embeddings = self.check_and_normalize_embeddings(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
        else:
            index = faiss.IndexFlatL2(embeddings.shape[1])

        index.add(embeddings.astype("float32"))
        return index

    def search_faiss_index(
        self,
        index: faiss.Index,
        embedding: np.ndarray,
        top_n: int,
        use_cosine_similarity: bool,
        similarity_threshold: float,
    ) -> Tuple[List[int], np.ndarray]:
        distances, indices = index.search(
            embedding.reshape(1, -1).astype("float32"), top_n + 1
        )

        if use_cosine_similarity:
            similarity_scores = distances.flatten()
        else:
            similarity_scores = 1 - distances.flatten()

        # Exclude results that are too similar
        indices = indices.flatten()[similarity_scores < similarity_threshold]
        similarity_scores = similarity_scores[similarity_scores < similarity_threshold]

        return indices[:top_n], similarity_scores[:top_n]

    async def query_similar_documents(
        self,
        text: str,
        top_n: int,
        filter_criteria: Optional[dict | Filter] = None,
        use_cosine_similarity: bool = True,
        similarity_threshold: float = 0.98,
    ) -> pd.DataFrame:
        query_embedding = await self.encode_string(text)

        if filter_criteria is not None:
            filtered_df = self.df.copy()
            if isinstance(filter_criteria, Filter):
                filter_criteria = filter_criteria.where
            for key, value in filter_criteria.items():
                filtered_df = filtered_df[filtered_df[key] == value]
        else:
            filtered_df = self.df.copy()

        filtered_embeddings = np.vstack(filtered_df[self.embedding_col_name].values)

        index_ = self.build_faiss_index(filtered_embeddings, use_cosine_similarity)
        indices, sim_scores = self.search_faiss_index(
            index_, query_embedding, top_n, use_cosine_similarity, similarity_threshold
        )
        results_df = filtered_df.iloc[indices].copy()
        # Add 'similarity scores' to the DataFrame
        results_df["sim_score_dense"] = sim_scores

        return results_df
