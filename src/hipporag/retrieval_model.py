import sys
from typing import List

import _pickle as pickle
import argparse
from glob import glob
import os.path
import pandas as pd

import pickle
import numpy as np
import os
from tqdm import tqdm

import faiss

from transformers import AutoModel, AutoTokenizer
from src.hipporag.processing import *
from src.hipporag.config_manager import ConfigManager


class RetrievalModule:
    """
    Class designed to retrieve potential synonymy candidates for a set of UMLS terms from a set of entities.
    """

    def __init__(
        self,
        config: ConfigManager,
    ):
        """
        Args:
            retriever_name: Retrieval names can be one of 3 types
                2) The name of a pickle file mapping AUIs to precomputed vectors
                3) A huggingface transformer model
        """
        self.config = config
        self.retriever_name = config.retriever_name
        self.pool_method = "cls"
        self.retrieval_name_dir = None

        # Search for pickle file
        print("No Pre-Computed Vectors. Confirming PLM Model.")

        try:
            if "ckpt" in self.retriever_name:
                self.plm = AutoModel.load_from_checkpoint(self.retriever_name)
            else:
                self.plm = AutoModel.from_pretrained(self.retriever_name)
        except:
            assert False, print(
                "{} is an invalid retriever name. Check Documentation.".format(
                    self.retriever_name
                )
            )

        # If not pre-computed, create vectors
        self.retrieval_name_dir = self.config.vector_directory / self.retriever_name.replace('/', '_').replace('.', '')
        if not self.retrieval_name_dir.exists():
            self.retrieval_name_dir.mkdir(parents=True, exist_ok=True)
        print(self.retrieval_name_dir)
        # Get previously computed vectors
        precomp_strings, precomp_vectors = self.get_precomputed_plm_vectors()

        # Get AUI Strings to be Encoded
        string_df = pd.read_csv(self.config.kb_to_kb_path, sep="\t")
        string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]
        sorted_df = self.create_sorted_df(string_df.strings.values)

        # Identify Missing Strings
        missing_strings = self.find_missing_strings(
            sorted_df.strings.unique(), precomp_strings
        )

        # Encode Missing Strings
        if len(missing_strings) > 0:
            print("Encoding {} Missing Strings".format(len(missing_strings)))
            (
                new_vectors,
                new_strings,
            ) = self.encode_strings(missing_strings, "cls")

            precomp_strings = list(precomp_strings)
            precomp_vectors = list(precomp_vectors)

            precomp_strings.extend(list(new_strings))
            precomp_vectors.extend(list(new_vectors))

            precomp_vectors = np.array(precomp_vectors)

            self.save_vecs(precomp_strings, precomp_vectors)

        self.vector_dict = self.make_dictionary(
            sorted_df, precomp_strings, precomp_vectors
        )

        print("Vectors Loaded.")

        queries = string_df[string_df.type == "query"]
        kb = string_df[string_df.type == "kb"]

        nearest_neighbors = self.retrieve_knn(queries.strings.values, kb.strings.values)
        
        # Use pathlib to handle path operations
        output_path = self.config.nearest_neighbors_path
        
        # Use 'with' statement for proper file handling
        with open(output_path, "wb") as f:
            pickle.dump(nearest_neighbors, f)

    def get_precomputed_plm_vectors(self):
        # Load or Create a DataFrame sorted by phrase length for efficient PLM computation
        strings = self.load_precomp_strings()
        vectors = self.load_plm_vectors()
        return strings, vectors

    def create_sorted_df(self, strings):
        lengths = []

        for string in tqdm(strings):
            lengths.append(len(str(string)))

        lengths_df = pd.DataFrame(lengths)
        lengths_df["strings"] = strings

        return lengths_df.sort_values(0)

    def save_vecs(self, strings: List[str], vectors: np.ndarray, bin_size: int = 50000) -> None:
        """
        Save the encoded strings and their corresponding vectors to files.

        Args:
            strings (List[str]): List of encoded strings to save.
            vectors (np.ndarray): Array of vectors corresponding to the strings.
            bin_size (int): Number of vectors to save in each file. Defaults to 50000.

        Raises:
            TypeError: If bin_size is not an integer.

        Example usage:
            retrieval_model.save_vecs(encoded_strings, encoded_vectors)
        """
        # Save encoded strings
        with open(self.retrieval_name_dir / "encoded_strings.txt", "w") as f:
            for string in strings:
                f.write(string + "\n")

        # Ensure bin_size is an integer
        if not isinstance(bin_size, int):
            raise TypeError("bin_size must be an integer")

        # Split vectors into bins
        num_bins = (len(vectors) + bin_size - 1) // bin_size  # Use integer division
        split_vecs = np.array_split(vectors, num_bins)

        # Save vector bins
        for i, vecs in tqdm(enumerate(split_vecs), desc="Saving vector bins"):
            with open(self.retrieval_name_dir / f"vecs_{i}.p", "wb") as f:
                pickle.dump(vecs, f)

    def load_precomp_strings(self):
        filename = self.retrieval_name_dir / "encoded_strings.txt"

        if not filename.exists():
            return []

        with open(filename, "r") as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]

        return lines

    def load_plm_vectors(self):
        vectors = []

        print("Loading PLM Vectors.")
        files = glob(str(self.retrieval_name_dir / "vecs_*.p"))

        if len(files) == 0:
            return vectors

        for i in tqdm(range(len(files))):
            i_files = glob(str(self.retrieval_name_dir / f"vecs_{i}.p"))
            if len(i_files) != 1:
                break
            else:
                with open(i_files[0], "rb") as f:
                    vectors.append(pickle.load(f))

        if vectors:
            vectors = np.vstack(vectors)
        else:
            vectors = np.array([])

        return vectors

    def find_missing_strings(self, relevant_strings, precomputed_strings):
        return list(set(relevant_strings).difference(set(precomputed_strings)))

    def make_dictionary(self, sorted_df, precomp_strings, precomp_vectors):
        print("Populating Vector Dict")
        precomp_string_ids = {}

        for i, string in enumerate(precomp_strings):
            precomp_string_ids[string] = i

        vector_dict = {}

        for i, row in tqdm(sorted_df.iterrows(), total=len(sorted_df)):
            string = row.strings

            try:
                vector_id = precomp_string_ids[string]
                vector_dict[string] = precomp_vectors[vector_id]
            except Exception as e:
                print(e)

        return vector_dict

    def encode_strings(self, strs_to_encode, pool_method):
        tokenizer = AutoTokenizer.from_pretrained(self.retriever_name)

        # Sorting Strings by length
        sorted_missing_strings = [len(s) for s in strs_to_encode]
        strs_to_encode = list(
            np.array(strs_to_encode)[np.argsort(sorted_missing_strings)]
        )

        all_cls = []
        all_strings = []
        num_strings_proc = 0

        with torch.no_grad():
            batch_sizes = []

            text_batch = []
            max_pad_size = 0

            for i, string in tqdm(enumerate(strs_to_encode), total=len(strs_to_encode)):
                length = len(tokenizer.tokenize(string))

                text_batch.append(string)
                num_strings_proc += 1

                if length > max_pad_size:
                    max_pad_size = length

                if max_pad_size * len(text_batch) > 50000 or num_strings_proc == len(
                    strs_to_encode
                ):
                    text_batch = list(text_batch)
                    encoding = tokenizer(
                        text_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.plm.config.max_length,
                    )
                    input_ids = encoding["input_ids"]
                    attention_mask = encoding["attention_mask"]

                    outputs = self.plm(input_ids, attention_mask=attention_mask)

                    if pool_method == "cls":
                        embeddings = outputs[0][:, 0, :]

                    elif pool_method == "mean":
                        embeddings = mean_pooling(outputs[0], attention_mask)

                    all_cls.append(embeddings.cpu().numpy())
                    all_strings.extend(text_batch)

                    batch_sizes.append(len(text_batch))

                    text_batch = []
                    max_pad_size = 0

        all_cls = np.vstack(all_cls)

        assert len(all_cls) == len(all_strings)
        assert all(
            [all_strings[i] == strs_to_encode[i] for i in range(len(all_strings))]
        )

        return all_cls, all_strings

    def retrieve_knn(self, queries, knowledge_base, k=2047):
        """
        Retrieve k-nearest neighbors for queries from the knowledge base using CPU-only FAISS.

        Args:
            queries: List of query strings.
            knowledge_base: List of strings in the knowledge base.
            k: Number of nearest neighbors to retrieve.

        Returns:
            A dictionary mapping each query to its nearest neighbors and distances.
        """
        import os

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        original_vecs = []
        new_vecs = []

        for string in knowledge_base:
            original_vecs.append(self.vector_dict[string])

        for string in queries:
            new_vecs.append(self.vector_dict[string])

        if len(original_vecs) == 0 or len(new_vecs) == 0:
            return {}

        original_vecs = np.vstack(original_vecs)
        new_vecs = np.vstack(new_vecs)

        print("Building Index")
        dim = original_vecs.shape[1]
        print(f"dim: {dim}")

        index = faiss.IndexFlatIP(dim)
        index.add(original_vecs.astype("float32"))

        print("Searching")
        D, I = index.search(new_vecs.astype("float32"), k)
        print("Index done!")

        sorted_candidate_dictionary = {}

        for query, nn_inds, nn_dists in zip(queries, I, D):
            nns = [knowledge_base[i] for i in nn_inds]
            sorted_candidate_dictionary[query] = (nns, nn_dists)

        return sorted_candidate_dictionary
