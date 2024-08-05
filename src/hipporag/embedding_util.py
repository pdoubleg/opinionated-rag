import os
from typing import Union, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from colbert.data import Queries

from src.hipporag.processing import (
    mean_pooling_embedding_with_normalization,
    mean_pooling_embedding,
)


class HuggingFaceWrapper:
    """
    A wrapper class for Hugging Face models to handle text encoding and similarity scoring.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize the HuggingFaceWrapper.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            device (str, optional): The device to run the model on. Defaults to 'cpu'.
        """
        self.model_name: str = model_name
        self.model_name_processed: str = model_name.replace("/", "_").replace(".", "_")
        self.model: AutoModel = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device: str = device

    def encode_text(
        self,
        text: Union[str, List[str]],
        norm: bool = True,
        return_cpu: bool = False,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode the input text using the Hugging Face model.

        Args:
            text (Union[str, List[str]]): The text to encode.
            norm (bool, optional): Whether to normalize the embeddings. Defaults to True.
            return_cpu (bool, optional): Whether to return the result on CPU. Defaults to False.
            return_numpy (bool, optional): Whether to return the result as a numpy array. Defaults to False.

        Returns:
            Union[torch.Tensor, np.ndarray]: The encoded text.
        """
        encoding_func = (
            mean_pooling_embedding_with_normalization
            if norm
            else mean_pooling_embedding
        )
        with torch.no_grad():
            if isinstance(text, str):
                text = [text]
            res = []
            if len(text) > 1:
                for t in tqdm(
                    text, total=len(text), desc=f"HF model {self.model_name} encoding"
                ):
                    res.append(
                        encoding_func(t, self.tokenizer, self.model, self.device)
                    )
            else:
                res = [encoding_func(text[0], self.tokenizer, self.model, self.device)]
            res = torch.stack(res)
            res = torch.squeeze(res, dim=1)

        if return_cpu:
            res = res.cpu()
        if return_numpy:
            res = res.numpy()
        return res

    def get_query_doc_scores(
        self, query_vec: np.ndarray, doc_vecs: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the similarity scores between a query vector and document vectors.

        Args:
            query_vec (np.ndarray): The query vector.
            doc_vecs (np.ndarray): The document vectors.

        Returns:
            np.ndarray: A matrix of query-document similarity scores.
        """
        return np.dot(doc_vecs, query_vec.T)


def init_embedding_model(model_name: str) -> HuggingFaceWrapper:
    """
    Initialize and return a HuggingFaceWrapper embedding model.

    Args:
        model_name (str): The name of the Hugging Face model to initialize.

    Returns:
        HuggingFaceWrapper: An instance of the HuggingFaceWrapper class initialized with the specified model.
    """
    return HuggingFaceWrapper(model_name)


def colbertv2_index(
    corpus: list,
    dataset_name: str,
    exp_name: str,
    index_name="nbits_2",
    checkpoint_path="exp/colbertv2.0",
    overwrite="reuse",
):
    """
    Indexing corpus and phrases using colbertv2
    @param corpus:
    @return:
    """
    from src.hipporag.config_manager import ConfigManager

    config = ConfigManager()

    corpus_processed = [x.replace("\n", "\t") for x in corpus]

    corpus_tsv_file_path = (
        config.vector_directory
        / "colbert"
        / f"{dataset_name}_{exp_name}_{len(corpus_processed)}.tsv"
    )
    with open(corpus_tsv_file_path, "w") as f:  # save to tsv
        for pid, p in enumerate(corpus_processed):
            f.write(f'{pid}\t"{p}"' + "\n")
    root_path = config.vector_directory / "colbert" / f"{dataset_name}"

    # indexing corpus
    with Run().context(RunConfig(nranks=1, experiment=exp_name, root=root_path)):
        Config = ColBERTConfig(
            nbits=2,
            root=root_path,
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=Config)
        indexer.index(
            name=index_name, collection=corpus_tsv_file_path, overwrite=overwrite
        )


def retrieve_knn(kb, queries, duplicate=True, nns=100):
    checkpoint_path = "exp/colbertv2.0"

    from src.hipporag.config_manager import ConfigManager

    config = ConfigManager()
    vector_dir = config.vector_directory

    if duplicate:
        kb = list(
            set(list(kb) + list(queries))
        )  # Duplicating queries to obtain score of query to query and normalize

    corpus_path = vector_dir / "colbert" / "corpus.tsv"
    with open(corpus_path, "w") as f:  # save to tsv
        for pid, p in enumerate(kb):
            f.write(f'{pid}\t"{p}"' + "\n")

    queries_path = vector_dir / "colbert" / "queries.tsv"
    with open(queries_path, "w") as f:  # save to tsv
        for qid, q in enumerate(queries):
            f.write(f"{qid}\t{q}" + "\n")

    # index
    with Run().context(RunConfig(nranks=1, experiment="colbert", root=str(vector_dir))):
        Config = ColBERTConfig(nbits=2, root=str(vector_dir / "colbert"))
        indexer = Indexer(checkpoint=checkpoint_path, config=Config)
        indexer.index(
            name="nbits_2",
            collection=str(corpus_path),
            overwrite=True,
        )

    # retrieval
    with Run().context(RunConfig(nranks=1, experiment="colbert", root="")):
        searcher = Searcher(index="nbits_2", config=config)
        queries = Queries(
            os.path.join(config.vector_directory, "colbert", "queries.tsv")
        )
        ranking = searcher.search_all(queries, k=nns)

    ranking_dict = {}

    for i in range(len(queries)):
        query = queries[i]
        rank = ranking.data[i]
        max_score = rank[0][2]
        if duplicate:
            rank = rank[1:]
        ranking_dict[query] = (
            [kb[r[0]] for r in rank],
            [r[2] / max_score for r in rank],
        )

    return ranking_dict
