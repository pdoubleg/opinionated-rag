import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Optional
from tqdm import tqdm

from src.agent.tools.semantic_search import Filter
from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

class SparseEmbeddingsSplade:
    def __init__(
        self, df: pd.DataFrame, text_column: str, embedding_column: str = "splade_embeddings"
    ):
        """
        Initializes the SparseEmbeddingsSplade class with a DataFrame, a text column, and optionally an embedding column.
        If the embedding column is not provided or does not exist in the DataFrame, it generates SPLADE embeddings.

        Args:
            df (pd.DataFrame): The DataFrame containing the documents.
            text_column (str): The name of the column in the DataFrame containing the text to embed.
            embedding_column (Optional[str]): The name of the column in the DataFrame containing pre-computed embeddings.
        """
        self.df = df
        self.text_column = text_column
        self.embedding_column = embedding_column
        self.sparse_model_id: str = "naver/splade-cocondenser-ensembledistil"
        self.tokenizer = AutoTokenizer.from_pretrained(self.sparse_model_id)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(self.sparse_model_id)

        if self.embedding_column not in self.df.columns:
            logger.info("Generating sparse (SPLADE) embeddings...")
            self.df = self.add_splade_embeddings_to_df()
            logger.info(f"Done! Embeddings have been saved to self.df['{embedding_column}']")
        else:
            logger.info(f"Using pre-computed '{self.text_column}' embeddings from existing column: {self.embedding_column}")

    def splade_embed_documents(self, docs: List[str]) -> List[np.ndarray]:
        """
        Embeds a list of documents using a sparse model and displays progress.

        Args:
            docs (List[str]): List of documents to be embedded.

        Returns:
            List[np.ndarray]: List of document embeddings.
        """

        sparse_embeddings = []
        for doc in tqdm(docs, desc="Embedding documents"):
            tokens = self.tokenizer(
                doc, return_tensors="pt", padding=True, truncation=True
            )
            output = self.sparse_model(**tokens)
            doc_embedding = (
                torch.max(
                    torch.log(1 + torch.relu(output.logits))
                    * tokens.attention_mask.unsqueeze(-1),
                    dim=1,
                )[0]
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            sparse_embeddings.append(doc_embedding)
        return sparse_embeddings

    def splade_embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a query using a sparse model.

        Args:
            query (str): Query to be embedded.

        Returns:
            np.ndarray: Query embedding.
        """
        tokens = self.tokenizer(
            query, return_tensors="pt", padding=True, truncation=True
        )
        output = self.sparse_model(**tokens)
        query_embedding = (
            torch.max(
                torch.log(1 + torch.relu(output.logits))
                * tokens.attention_mask.unsqueeze(-1),
                dim=1,
            )[0]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        return query_embedding
    
    @staticmethod
    def sigmoid_similarity(scores: List[float]) -> List[float]:
        """
        Apply a sigmoid function to similarity scores to map them to a (0, 1) range.

        Args:
            scores (List[float]): The raw dot product similarity scores.

        Returns:
            List[float]: The transformed similarity scores.
        """
        return [1 / (1 + np.exp(-score)) for score in scores]


    @staticmethod
    def dot_product_similarity(
        doc_embeddings: np.ndarray, 
        query_embedding: np.ndarray,
    ) -> List[float]:
        """
        Calculate the dot product similarity between the query embedding and each document embedding.

        Args:
            doc_embeddings (np.ndarray): The document embeddings.
            query_embedding (np.ndarray): The query embedding.

        Returns:
            List[float]: The similarity scores.
        """
        similarities = [np.dot(query_embedding, doc_emb) for doc_emb in doc_embeddings]
        return similarities

    def add_splade_embeddings_to_df(self) -> pd.DataFrame:
        """
        Adds a new column to the dataframe with sparse embeddings of the text column.

        Args:
            df (pd.DataFrame): Dataframe to be processed.
            text_column (str): Name of the column in df which contains the text to be embedded.

        Returns:
            pd.DataFrame: Dataframe with the new column of splade embeddings.
        """
        self.df["splade_embeddings"] = self.splade_embed_documents(
            self.df[self.text_column].tolist()
        )
        return self.df

    def query_similar_documents(
        self, 
        query: str, 
        top_n: int = 20, 
        filter_criteria: dict | Filter | None = None,
        norm_score: bool = False,
    ) -> pd.DataFrame:
        """
        Search documents based on the similarity of their embeddings to the query embedding.

        Args:
            query (str): The query to search for.
            top_n (int): The number of top documents to return.
            filter_criteria (optional, dict): A dictionary of key (column names) value pairs to filter the df.

        Returns:
            pd.DataFrame: A dataframe containing the top_n documents and their similarity scores, sorted by similarity.
        """
        if filter_criteria is not None:
            filtered_df = self.df.copy()
            if isinstance(filter_criteria, Filter):
                filter_criteria = filter_criteria.where
            for key, value in filter_criteria.items():
                filtered_df = filtered_df[filtered_df[key] == value]
        else:
            filtered_df = self.df.copy()

        query_embedding = self.splade_embed_query(query)
        document_embeddings = filtered_df[self.embedding_column].tolist()
        similarities = self.dot_product_similarity(document_embeddings, query_embedding)
        if norm_score:
            similarities = self.sigmoid_similarity(similarities)
            
        filtered_df["score"] = similarities
        filtered_df.drop_duplicates(subset=[self.text_column], keep='first', inplace=True)
        filtered_df["search_type"] = "splade"
        ranked_df = filtered_df.sort_values(
            by="score", ascending=False
        ).head(top_n)
        return ranked_df

    def generate_expansion_terms(self, input_string: str) -> list:
        """
        Generates a list of expansion terms based on the input string.

        Args:
            input_string (str): The string to generate expansion terms for.

        Returns:
            list: A list of expansion terms generated by SPLADE.
        """
        tokens = self.tokenizer(
            input_string, return_tensors="pt", padding=True, truncation=True
        )
        output = self.sparse_model(**tokens)

        query_embedding = torch.max(
            torch.log(1 + torch.relu(output.logits))
            * tokens.attention_mask.unsqueeze(-1),
            dim=1,
        )[0].squeeze()

        idx2token = {idx: token for token, idx in self.tokenizer.get_vocab().items()}

        cols = query_embedding.nonzero().squeeze().cpu().tolist()
        weights = query_embedding[cols].cpu().tolist()
        sparce_dict = dict(zip(cols, weights))
        # map token IDs to readable tokens
        sparce_dict_tokens = {
            idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
        }
        # sort to most relevant tokens first
        sparce_dict_tokens = {
            k: v
            for k, v in sorted(
                sparce_dict_tokens.items(), key=lambda item: item[1], reverse=True
            )
        }
        sparce_dict_tokens = self.strip_hash_from_dict_keys(sparce_dict_tokens)
        filtered_keys = self.filter_dict_by_string(sparce_dict_tokens, input_string)

        print(f"SPLADE generated {len(filtered_keys)} expansion terms")
        print(f"Top expansion terms: {filtered_keys[:10]}")
        return filtered_keys

    @staticmethod
    def strip_hash_from_dict_keys(input_dict: dict) -> dict:
        """Helper to remove pound sign from word piece tokens"""
        return {k.replace("#", ""): v for k, v in input_dict.items()}

    @staticmethod
    def filter_dict_by_string(input_dict: dict, input_string: str) -> list:
        """Helper to filter out tokens that appear in the input string"""
        return [key for key in input_dict if key.lower() not in input_string.lower()]

