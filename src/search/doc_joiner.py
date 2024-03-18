import itertools
from collections import defaultdict
from math import inf
from typing import List, Optional
import numpy as np

import pandas as pd

from src.types import Document

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


class DocJoinerDF:
    """
    A class that joins multiple DataFrames into a single DataFrame.

    It supports different joins modes:
    - concatenate: Keeps the highest scored document in case of duplicates.
        Takes 'best of' while maintaining the original score.
    - merge: Merge and calculate a weighted sum of the scores of duplicate documents.
        Considers all instances of duplicate records and updates their scores.
    - reciprocal_rank_fusion: Merge and assign scores based on reciprocal rank fusion.
        Ignores original scores focusing only on ranks across search results.
    """

    def __init__(
        self,
        join_mode: str = "concatenate",
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        sort_by_score: bool = True,
    ):

        if join_mode not in ["concatenate", "merge", "reciprocal_rank_fusion"]:
            raise ValueError(f"DocJoinerDF component does not support '{join_mode}' join_mode.")
        self.join_mode = join_mode
        self.weights = weights
        self.top_k = top_k
        self.sort_by_score = sort_by_score

    def run(self, dataframes: List[pd.DataFrame]):
        """
        Joins multiple DataFrames of documents into a single DataFrame depending on the `join_mode` parameter.
        """
        output_df = pd.DataFrame()
        if self.join_mode == "concatenate":
            output_df = self._concatenate(dataframes)
        if self.join_mode == "reciprocal_rank_fusion":
            output_df = self._reciprocal_rank_fusion(dataframes)
        elif self.join_mode == "merge":
            output_df = self._merge(dataframes)
        if self.sort_by_score:
            output_df = output_df.sort_values(by='score', ascending=False)
        if self.top_k:
            output_df = output_df.head(self.top_k)
        return output_df


    def _concatenate(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate multiple DataFrames and retain the record with the highest score for duplicate records.

        Args:
            dataframes (List[pd.DataFrame]): A list of pandas DataFrames to be concatenated.

        Returns:
            pd.DataFrame: A DataFrame containing the concatenated 'documents'.

        Raises:
            ValueError: If any DataFrame in `dataframes` does not contain a 'score' column.
        """
        
        # Check if all DataFrames have a 'score' column
        for df in dataframes:
            if 'score' not in df.columns:
                raise ValueError("All DataFrames must contain a 'score' column.")

        concatenated_df = pd.concat(dataframes, ignore_index=True)
        # Remove potential embeddings so that the df can be deduplicated
        array_columns = [col for col in concatenated_df.columns if isinstance(concatenated_df[col].iloc[0], np.ndarray)]
        array_df = concatenated_df[array_columns]
        concatenated_df.drop(columns=array_columns, inplace=True)
        # Select the highest scored document for duplicates
        concatenated_df = concatenated_df.loc[concatenated_df.groupby('index')['score'].idxmax()]
        final_df = pd.concat([concatenated_df, array_df.loc[concatenated_df.index]], axis=1)
        return final_df
        
        
    def _reciprocal_rank_fusion(self, dataframes, id_column='index') -> pd.DataFrame:
        """
        Merge a list of DataFrames and assign scores based on reciprocal rank fusion.
        The constant k is set to 61 (60 was suggested by the original paper,
        plus 1 as python lists are 0-based and the paper used 1-based ranking).
        """
        k = 61
        scores_map = defaultdict(float)
        rank_map = defaultdict(lambda: float('inf'))  # Default to infinity for comparison
        documents_map = {}

        if self.weights is None:
            self.weights = [1 / len(dataframes)] * len(dataframes)

        for df, weight in zip(dataframes, self.weights):
            for index, row in df.iterrows():
                doc_id = row[id_column]
                rank = index  # Assuming the DataFrame is sorted by relevance
                
                # Update score
                scores_map[doc_id] += (weight * len(dataframes)) / (k + rank)
                
                # Check if this is the best (lowest) rank seen for this doc_id
                if rank < rank_map[doc_id]:
                    rank_map[doc_id] = rank
                    documents_map[doc_id] = row

        # Calculate normalized scores and update documents_map
        max_possible_score = len(dataframes) / k
        for doc_id, row in documents_map.items():
            normalized_score = scores_map[doc_id] / max_possible_score
            documents_map[doc_id]['score'] = normalized_score

        # Convert documents_map back to a DataFrame
        result_df = pd.DataFrame.from_dict(documents_map, orient='index')

        # Sort by new score and return
        return result_df.sort_values(by=['score'], ascending=False)
    

    def _merge(self, dataframes) -> pd.DataFrame:
        """
        Merge a list of DataFrames and calculate a weighted sum of the scores to deduplicate records.

        Args:
            dataframes (List[pd.DataFrame]): A list of pandas DataFrames to be merged.

        Returns:
            pd.DataFrame: A DataFrame containing the merged 'documents'.

        Raises:
            ValueError: If any DataFrame in `dataframes` does not contain a 'score' column.
        """
        # Check if all DataFrames have a 'score' column
        for df in dataframes:
            if 'score' not in df.columns:
                raise ValueError("All DataFrames must contain a 'score' column.")
        
        scores_map = defaultdict(float)
        documents_map = {}

        if self.weights is None:
            self.weights = [1 / len(dataframes)] * len(dataframes)

        for df, weight in zip(dataframes, self.weights):
            for index, row in df.iterrows():
                doc_id = index  # Using DataFrame index as unique ID
                score = row['score'] if 'score' in row and pd.notnull(row['score']) else 0
                scores_map[doc_id] += score * weight
                if doc_id not in documents_map:
                    documents_map[doc_id] = row.to_dict()

        # Update the scores in documents_map based on scores_map
        for doc_id, info in documents_map.items():
            info['score'] = scores_map[doc_id]

        # Convert documents_map back to a DataFrame
        result_df = pd.DataFrame.from_dict(documents_map, orient='index')

        return result_df
    
    
    


