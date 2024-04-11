import hashlib
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
    A class designed to join and deduplicate multiple pandas DataFrames into a single consolidated DataFrame.

    This class supports various join modes, which determine how duplicate documents are handled and how their scores are aggregated or recalculated.
    Each join mode applies a different strategy for dealing with document scores during the joining process:

    - concatenate: This mode concatenates all documents across the input DataFrames. In the case of duplicate documents (identified by the text),
        only the document with the highest score is retained. This approach ensures that the best score of each document is preserved in the final DataFrame.

    - merge: In this mode, documents are merged based on their identification (text content). The scores of duplicate documents are combined into a weighted sum,
        reflecting the aggregate score across all occurrences. This method considers a document's relevance by its scores across multiple DataFrames.

    - reciprocal_rank_fusion: This mode also merges documents, but it calculates new scores using the reciprocal rank fusion algorithm.
        Original scores are not considered; instead, the method focuses on the rank of each document within its original DataFrame.
        The final score for each document is determined by its ranks across all input DataFrames, making it well suited when scoring metrics vary.

    Parameters:
    - join_mode (str): Specifies the join mode to be used. Must be one of 'concatenate', 'merge', or 'reciprocal_rank_fusion'.
    - weights (Optional[List[float]]): A list of weights corresponding to each DataFrame. Used in the 'merge' and 'reciprocal_rank_fusion' modes to
        adjust the influence of each DataFrame's scores on the final results. If not provided, equal weights are assumed. One use case is to assign more weight
        to the original query when applying query expansion or transformation.
    - top_k (Optional[int]): Limits the output to the top-k documents based on their final scores. If not provided, all documents are included in the results.

    """

    def __init__(
        self,
        join_mode: str = "concatenate",
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
    ):
        if join_mode not in ["concatenate", "merge", "reciprocal_rank_fusion"]:
            raise ValueError(
                f"DocJoinerDF component does not support '{join_mode}' join_mode."
            )
        self.join_mode = join_mode
        self.weights = weights
        self.top_k = top_k

    def run(self, dataframes: List[pd.DataFrame], text_column: str = "text"):
        """
        Joins multiple DataFrames of documents into a single DataFrame depending on the `join_mode` parameter.

        Args:
            dataframes (List[pd.DataFrame]): A list of pandas DataFrames to be joined.
            text_column (str): The name of the column containing the document text used for deduplication.
        """
        # Assign unique IDs based on document text
        for df in dataframes:
            df["_internal_id"] = df[text_column].apply(
                lambda x: hashlib.md5(x.encode()).hexdigest()
            )

        if self.join_mode == "concatenate":
            output_df = self._concatenate(dataframes, "_internal_id")
        elif self.join_mode == "reciprocal_rank_fusion":
            output_df = self._reciprocal_rank_fusion(dataframes, "_internal_id")
        elif self.join_mode == "merge":
            output_df = self._merge(dataframes, "_internal_id")

        if self.top_k:
            output_df = output_df.nlargest(self.top_k, "score")

        # Drop the internal ID column before returning the final DataFrame
        output_df.drop(columns=["_internal_id"], inplace=True)

        return output_df

    def _concatenate(
        self, dataframes: List[pd.DataFrame], id_column: str = "_internal_id"
    ) -> pd.DataFrame:
        """
        Concatenate multiple DataFrames and retain the record with the highest score for duplicate records.

        Args:
            dataframes (List[pd.DataFrame]): A list of pandas DataFrames to be concatenated.

        Returns:
            pd.DataFrame: A DataFrame containing the concatenated 'documents'.

        Raises:
            ValueError: If any DataFrame in `dataframes` does not contain a 'score' column.
        """
        # Validate 'score' column presence
        for df in dataframes:
            if "score" not in df.columns:
                raise ValueError("All DataFrames must contain a 'score' column.")

        # Concatenate all dataframes
        concatenated_df = pd.concat(dataframes, ignore_index=True)

        # Handling array columns by temporarily removing them
        array_columns = [
            col
            for col in concatenated_df.columns
            if isinstance(concatenated_df[col].iloc[0], np.ndarray)
        ]
        non_array_df = concatenated_df.drop(columns=array_columns)

        # Deduplicate by keeping the highest score for each unique document
        deduplicated_df = (
            non_array_df.sort_values(by="score", ascending=False)
            .drop_duplicates(subset=id_column, keep="first")
            .sort_index()
        )
        # Re-attach array columns
        final_df = pd.concat([deduplicated_df, concatenated_df[array_columns]], axis=1)
        final_df.sort_values(by="score", ascending=False, inplace=True)
        return final_df

    def _reciprocal_rank_fusion(
        self, dataframes: List[pd.DataFrame], id_column: str = "_internal_id"
    ) -> pd.DataFrame:
        """
        Merge a list of DataFrames and assign scores based on reciprocal rank fusion.
        The constant k is set to 61 (60 was suggested by the original paper,
        plus 1 as python lists are 0-based and the paper used 1-based ranking).
        """
        k = 61
        scores_map = defaultdict(float)
        documents_map = {}

        # Ensure weights are properly initialized
        if self.weights is None:
            self.weights = [1.0 / len(dataframes)] * len(dataframes)

        for df_idx, (df, weight) in enumerate(zip(dataframes, self.weights)):
            for rank, (_, row) in enumerate(df.iterrows(), start=1):
                doc_id = row[id_column]
                scores_map[doc_id] += (weight * len(dataframes)) / (k + rank - 1)
                if doc_id not in documents_map:
                    documents_map[doc_id] = row.to_dict()

        # Normalize the scores. Note: len(results) / k is the maximum possible score,
        # achieved by being ranked first in all doc lists with non-zero weight.
        max_possible_score = len(dataframes) / k
        for doc_id in documents_map.keys():
            documents_map[doc_id]["score"] = scores_map[doc_id] / max_possible_score

        # Convert the documents_map to a DataFrame
        result_df = pd.DataFrame.from_dict(documents_map, orient="index").reset_index(
            drop=True
        )

        # Handle duplicates based on text content, retaining the highest-ranking instance
        result_df = result_df.sort_values(by="score", ascending=False).drop_duplicates(
            subset=id_column, keep="first"
        )

        return result_df

    def _merge(
        self, dataframes: List[pd.DataFrame], id_column: str = "_internal_id"
    ) -> pd.DataFrame:
        """
        Merge a list of DataFrames and calculate a weighted sum of the scores to deduplicate records.

        Args:
            dataframes (List[pd.DataFrame]): A list of pandas DataFrames to be merged.

        Returns:
            pd.DataFrame: A DataFrame containing the merged 'documents'.

        Raises:
            ValueError: If any DataFrame in `dataframes` does not contain a 'score' column.
        """
        # Validate 'score' column and initialize scores_map and documents_map
        scores_map = defaultdict(float)
        documents_map = {}

        if self.weights is None:
            self.weights = [1.0 / len(dataframes)] * len(dataframes)

        for df, weight in zip(dataframes, self.weights):
            for _, row in df.iterrows():
                doc_id = row[id_column]
                score = row["score"]
                scores_map[doc_id] += score * weight
                if doc_id not in documents_map:
                    documents_map[doc_id] = row.to_dict()

        # Update scores in the documents_map
        for doc_id, info in documents_map.items():
            info["score"] = scores_map[doc_id]

        result_df = pd.DataFrame.from_dict(documents_map, orient="index")
        result_df.sort_values(by="score", ascending=False, inplace=True)
        result_df.reset_index(drop=True, inplace=True)
        return result_df
