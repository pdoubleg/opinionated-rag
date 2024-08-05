import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import _pickle as pickle
from collections import defaultdict
from glob import glob

import igraph as ig
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from colbert.data import Queries

from src.hipporag.langchain_util import init_langchain_model, LangChainModel
from src.hipporag.embedding_util import colbertv2_index, init_embedding_model
from src.hipporag.named_entity_extraction_parallel import named_entity_recognition
from src.hipporag.processing import get_output_path, processing_phrases, min_max_normalize
from src.hipporag.config_manager import ConfigManager

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"


class HippoRAG:
    def __init__(
        self,
        config: ConfigManager,
        linking_retriever_name=None,
    ):
        # Initialize core attributes
        self.config = config
        self.corpus_path = "./hipporag_data/hippo_test_corpus.json"
        self.corpus_name = self.config.dataset_name
        self.extraction_model_name = self.config.llm_model
        self.extraction_model_name_processed = self.config.llm_model.replace("/", "_")
        self.client = init_langchain_model(self.config.llm_model)

        # Set up retriever names
        assert self.config.retriever_name
        if linking_retriever_name is None:
            linking_retriever_name = self.config.retriever_name
        self.graph_creating_retriever_name = (
            self.config.retriever_name
        )  # 'colbertv2', 'facebook/contriever', or other HuggingFace models
        self.graph_creating_retriever_name_processed = (
            self.config.retriever_name.replace("/", "_").replace(".", "")
        )
        self.linking_retriever_name = linking_retriever_name
        self.linking_retriever_name_processed = linking_retriever_name.replace(
            "/", "_"
        ).replace(".", "")

        # Set up graph and extraction parameters
        self.extraction_type = self.config.extraction_type
        self.graph_type = self.config.graph_type
        self.sim_threshold = self.config.sim_threshold
        self.node_specificity = self.config.node_specificity

        # Configure ColBERT
        if "colbert" in self.config.retriever_name.lower():
            # Use default configuration for ColBERT
            root_path = self.config.vector_directory / "colbert" / f"{self.corpus_name}"
            self.colbert_config = {
                "root": root_path,
                "doc_index_name": "nbits_2",
                "phrase_index_name": "nbits_2",
            }
        else:
            # For non-ColBERT retrievers, set colbert_config to None
            self.colbert_config = None

        # Set up graph algorithm parameters
        self.graph_alg = self.config.graph_alg
        self.damping = self.config.damping
        self.recognition_threshold = self.config.recognition_threshold

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Load or initialize named entity cache
        try:
            self.named_entity_cache = pd.read_csv(
                self.config.output_directory
                / f"{self.corpus_name}_queries.named_entity_output.tsv",
                sep="\t",
            )
        except Exception as e:
            # If file not found, initialize an empty DataFrame
            self.named_entity_cache = pd.DataFrame([], columns=["query", "triples"])

        # Process named entity cache based on column names
        if "query" in self.named_entity_cache:
            self.named_entity_cache = {
                row["query"]: eval(row["triples"])
                for i, row in self.named_entity_cache.iterrows()
            }
        elif "question" in self.named_entity_cache:
            self.named_entity_cache = {
                row["question"]: eval(row["triples"])
                for i, row in self.named_entity_cache.iterrows()
            }

        # Initialize embedding model and other attributes
        self.embed_model = init_embedding_model(self.graph_creating_retriever_name)
        self.dpr_only = self.config.dpr_only
        self.doc_ensemble = self.config.doc_ensemble

        self.graph_plus = {}  # Initialize as an empty dictionary
        self.statistics = {}

        # Conditional path: Load index files and build graph if not using DPR only
        if not self.dpr_only:
            self.load_index_files()
            self.build_graph()
            self.load_node_vectors()
        else:
            # If using DPR only, just load the corpus
            self.load_corpus()

        # Conditional path: Load document embeddings for certain configurations
        if (self.doc_ensemble or self.dpr_only) and self.config.retriever_name not in [
            "colbertv2",
            "bm25",
        ]:
            self.get_dpr_doc_embedding()

        # Conditional path: Set up ColBERTv2 if it's the chosen retriever
        if self.config.retriever_name == "colbertv2":
            if self.dpr_only is False or self.doc_ensemble:
                # Index phrases and set up phrase searcher
                colbertv2_index(
                    self.phrases.tolist(),
                    self.corpus_name,
                    "phrase",
                    self.colbert_config["phrase_index_name"],
                    overwrite=True,
                )
                with Run().context(
                    RunConfig(
                        nranks=1, experiment="phrase", root=self.colbert_config["root"]
                    )
                ):
                    config = ColBERTConfig(
                        root=self.colbert_config["root"],
                    )
                    self.phrase_searcher = Searcher(
                        index=self.colbert_config["phrase_index_name"],
                        config=config,
                        verbose=0,
                    )
            if self.doc_ensemble or self.dpr_only:
                # Index corpus and set up corpus searcher
                colbertv2_index(
                    self.dataset_df[self.config.text_to_embed_column].tolist(),
                    self.corpus_name,
                    "corpus",
                    self.colbert_config["doc_index_name"],
                    overwrite=True,
                )
                with Run().context(
                    RunConfig(
                        nranks=1, experiment="corpus", root=self.colbert_config["root"]
                    )
                ):
                    config = ColBERTConfig(
                        root=self.colbert_config["root"],
                    )
                    self.corpus_searcher = Searcher(
                        index=self.colbert_config["doc_index_name"],
                        config=config,
                        verbose=0,
                    )


    def get_shortest_distance_between_nodes(self, node1: str, node2: str) -> float:
        """
        Get the shortest distance between two nodes in the graph.

        This function calculates the shortest path distance between two nodes
        in the graph representation of the knowledge base. It uses the igraph
        library's shortest_paths method to compute the distance.

        Args:
            node1 (str): The phrase representing the first node.
            node2 (str): The phrase representing the second node.

        Returns:
            float: The shortest distance between the two nodes. Returns -1 if
                   an error occurs (e.g., nodes not found in the graph).

        Depends on:
            - self.phrases: An array of all phrases in the graph.
            - self.g: An igraph Graph object representing the knowledge graph.

        Note:
            This function assumes that the graph (self.g) has been properly
            initialized and that the phrases are indexed in the same order
            as the graph nodes.
        """
        try:
            node1_id = np.where(self.phrases == node1)[0][0]
            node2_id = np.where(self.phrases == node2)[0][0]

            return self.g.shortest_paths(node1_id, node2_id)[0][0]
        except Exception as e:
            return -1

    def rank_docs(self, query: str, top_k=10):
        """
        Rank documents based on the input query using a combination of retrieval methods and graph algorithms.

        This function performs document ranking using a hybrid approach that may include dense retrieval,
        named entity recognition, and graph-based algorithms depending on the configuration.

        Args:
            query (str): The input query string.
            top_k (int, optional): The number of top-ranked documents to return. Defaults to 10.

        Returns:
            Tuple[List[int], List[float], Dict]: A tuple containing:
                - List of top-k document IDs (sorted by rank)
                - List of corresponding document scores
                - Dictionary of additional logging information

        Depends on:
            - self.query_ner: Method for named entity recognition
            - self.linking_retriever_name: Configuration for retrieval method
            - self.doc_ensemble, self.dpr_only: Flags for retrieval strategy
            - self.corpus_searcher: For ColBERTv2 retrieval
            - self.embed_model: For dense retrieval
            - self.doc_embedding_mat: Pre-computed document embeddings
            - self.link_node_by_colbertv2, self.link_node_by_dpr: Entity linking methods
            - self.graph_alg: Specified graph algorithm
            - self.run_pagerank_igraph_chunk, self.get_neighbors: Graph processing methods
            - Various matrices: self.facts_to_phrases_mat, self.docs_to_facts_mat, etc.

        Note:
            The function uses different strategies based on the configuration and the presence of named entities
            in the query. It may combine scores from dense retrieval and graph-based methods.
        """
        assert isinstance(query, str), "Query must be a string"
        query_ner_list = self.query_ner(query)

        if "colbertv2" in self.linking_retriever_name:
            # Get Query Doc Scores
            queries = Queries(path=None, data={0: query})
            if self.doc_ensemble:
                query_doc_scores = np.zeros(self.doc_to_phrases_mat.shape[0])
                ranking = self.corpus_searcher.search_all(
                    queries, k=self.doc_to_phrases_mat.shape[0]
                )
                # max_query_score = self.get_colbert_max_score(query)
                for doc_id, rank, score in ranking.data[0]:
                    query_doc_scores[doc_id] = score
            elif self.dpr_only:
                query_doc_scores = np.zeros(len(self.dataset_df))
                ranking = self.corpus_searcher.search_all(
                    queries, k=len(self.dataset_df)
                )
                for doc_id, rank, score in ranking.data[0]:
                    query_doc_scores[doc_id] = score

            if (
                len(query_ner_list) > 0
            ):  # if no entities are found, assign uniform probability to documents
                all_phrase_weights, linking_score_map = self.link_node_by_colbertv2(
                    query_ner_list
                )
        else:  # dense retrieval model
            # Get Query Doc Scores
            if self.doc_ensemble or self.dpr_only:
                query_embedding = self.embed_model.encode_text(
                    query, return_cpu=True, return_numpy=True, norm=True
                )
                query_doc_scores = np.dot(self.doc_embedding_mat, query_embedding.T)
                query_doc_scores = query_doc_scores.T[0]

            if (
                len(query_ner_list) > 0
            ):  # if no entities are found, assign uniform probability to documents
                all_phrase_weights, linking_score_map = self.link_node_by_dpr(
                    query_ner_list
                )

        # Run Personalized PageRank (PPR) or other Graph Algorithm Doc Scores
        if not self.dpr_only:
            if len(query_ner_list) > 0:
                combined_vector = np.max([all_phrase_weights], axis=0)

                if self.graph_alg == "ppr":
                    ppr_phrase_probs = self.run_pagerank_igraph_chunk(
                        [all_phrase_weights]
                    )[0]
                elif self.graph_alg == "none":
                    ppr_phrase_probs = combined_vector
                elif self.graph_alg == "neighbor_2":
                    ppr_phrase_probs = self.get_neighbors(combined_vector, 2)
                elif self.graph_alg == "neighbor_3":
                    ppr_phrase_probs = self.get_neighbors(combined_vector, 3)
                elif self.graph_alg == "paths":
                    ppr_phrase_probs = self.get_neighbors(combined_vector, 3)
                else:
                    assert False, f"Graph Algorithm {self.graph_alg} Not Implemented"

                fact_prob = self.facts_to_phrases_mat.dot(ppr_phrase_probs)
                ppr_doc_prob = self.docs_to_facts_mat.dot(fact_prob)
                ppr_doc_prob = min_max_normalize(ppr_doc_prob)
            else:  # dpr_only or no entities found
                ppr_doc_prob = np.ones(len(self.extracted_triples)) / len(
                    self.extracted_triples
                )

        # Combine Query-Doc and PPR Scores
        if self.doc_ensemble or self.dpr_only:
            # doc_prob = ppr_doc_prob * 0.5 + min_max_normalize(query_doc_scores) * 0.5
            if len(query_ner_list) == 0:
                doc_prob = query_doc_scores
                self.statistics["doc"] = self.statistics.get("doc", 0) + 1
            elif (
                np.min(list(linking_score_map.values())) > self.recognition_threshold
            ):  # high confidence in named entities
                doc_prob = ppr_doc_prob
                self.statistics["ppr"] = self.statistics.get("ppr", 0) + 1
            else:  # relatively low confidence in named entities, combine the two scores
                # the higher threshold, the higher chance to use the doc ensemble
                doc_prob = (
                    ppr_doc_prob * 0.5 + min_max_normalize(query_doc_scores) * 0.5
                )
                query_doc_scores = min_max_normalize(query_doc_scores)

                top_ppr = np.argsort(ppr_doc_prob)[::-1][:10]
                top_ppr = [(top, ppr_doc_prob[top]) for top in top_ppr]

                top_doc = np.argsort(query_doc_scores)[::-1][:10]
                top_doc = [(top, query_doc_scores[top]) for top in top_doc]

                top_hybrid = np.argsort(doc_prob)[::-1][:10]
                top_hybrid = [(top, doc_prob[top]) for top in top_hybrid]

                self.statistics["ppr_doc_ensemble"] = (
                    self.statistics.get("ppr_doc_ensemble", 0) + 1
                )
        else:
            doc_prob = ppr_doc_prob

        # Return ranked docs and ranked scores
        sorted_doc_ids = np.argsort(doc_prob, kind="mergesort")[::-1]
        sorted_scores = doc_prob[sorted_doc_ids]

        if not (self.dpr_only) and len(query_ner_list) > 0:
            # logs
            phrase_one_hop_triples = []
            for phrase_id in np.where(all_phrase_weights > 0)[0]:
                # get all the triples that contain the phrase from self.graph_plus
                for t in list(self.kg_adj_list[phrase_id].items())[:20]:
                    phrase_one_hop_triples.append([self.phrases[t[0]], t[1]])
                for t in list(self.kg_inverse_adj_list[phrase_id].items())[:20]:
                    phrase_one_hop_triples.append([self.phrases[t[0]], t[1], "inv"])

            # get top ranked nodes from doc_prob and self.doc_to_phrases_mat
            nodes_in_retrieved_doc = []
            for doc_id in sorted_doc_ids[:5]:
                node_id_in_doc = list(
                    np.where(self.doc_to_phrases_mat[[doc_id], :].toarray()[0] > 0)[0]
                )
                nodes_in_retrieved_doc.append(
                    [self.phrases[node_id] for node_id in node_id_in_doc]
                )

            # get top ppr_phrase_probs
            top_pagerank_phrase_ids = np.argsort(ppr_phrase_probs, kind="mergesort")[
                ::-1
            ][:20]

            # get phrases for top_pagerank_phrase_ids
            top_ranked_nodes = [
                self.phrases[phrase_id] for phrase_id in top_pagerank_phrase_ids
            ]
            logs = {
                "named_entities": query_ner_list,
                "linked_node_scores": [
                    list(k) + [float(v)] for k, v in linking_score_map.items()
                ],
                "1-hop_graph_for_linked_nodes": phrase_one_hop_triples,
                "top_ranked_nodes": top_ranked_nodes,
                "nodes_in_retrieved_doc": nodes_in_retrieved_doc,
            }
        else:
            logs = {}

        return sorted_doc_ids.tolist()[:top_k], sorted_scores.tolist()[:top_k], logs

    def query_ner(self, query: str) -> List[str]:
        """
        Perform Named Entity Recognition (NER) on the given query.

        This function extracts named entities from the input query. If DPR (Dense Passage Retrieval) is enabled,
        it returns an empty list. Otherwise, it attempts to retrieve named entities from a cache or performs
        NER using an external function.

        Args:
            query (str): The input query string to perform NER on.

        Returns:
            List[str]: A list of processed named entities extracted from the query.

        Depends on:
            - self.dpr_only: A boolean flag indicating if DPR is the only method used.
            - self.named_entity_cache: A cache storing previously extracted named entities.
            - named_entity_recognition(): An external function for NER.
            - processing_phrases(): A function to process extracted phrases.
            - self.client: A client object used for NER (passed to named_entity_recognition).
            - self.logger: A logging object for error reporting.

        Note:
            If an error occurs during NER, it logs the error and returns an empty list.
        """
        client = init_langchain_model(self.config.llm_model)
        if self.dpr_only:
            query_ner_list = []
        else:
            # Extract Entities
            try:
                if query in self.named_entity_cache:
                    query_ner_list = self.named_entity_cache[query]["named_entities"]
                else:
                    query_ner_json, total_tokens = named_entity_recognition(
                        client=client,
                        text=query,
                    )
                    query_ner_list = eval(query_ner_json)["named_entities"]

                query_ner_list = [processing_phrases(p) for p in query_ner_list]
            except Exception as e:
                self.logger.error(f"Error in Query NER: {e}")
                query_ner_list = []
        return query_ner_list

    def get_neighbors(self, prob_vector: np.ndarray, max_depth: int = 1) -> np.ndarray:
        """
        Expand the probability vector by including neighboring nodes in the graph.

        This function takes a probability vector and expands it by including
        neighboring nodes up to a specified depth in the graph. It modifies
        the probabilities of neighboring nodes based on the initial probabilities.

        Args:
            prob_vector (np.ndarray): Initial probability vector for nodes in the graph.
            max_depth (int, optional): Maximum depth to explore in the graph. Defaults to 1.

        Returns:
            np.ndarray: Modified probability vector including neighboring nodes.

        Depends on:
            - self.g: A graph object (e.g., NetworkX graph) with a neighbors() method.

        Note:
            This function modifies the input prob_vector in-place.
        """
        initial_nodes = prob_vector.nonzero()[0]
        min_prob = np.min(prob_vector[initial_nodes])

        for initial_node in initial_nodes:
            all_neighborhood = []

            current_nodes = [initial_node]

            for depth in range(max_depth):
                next_nodes = []

                for node in current_nodes:
                    next_nodes.extend(self.g.neighbors(node))
                    all_neighborhood.extend(self.g.neighbors(node))

                current_nodes = list(set(next_nodes))

            for i in set(all_neighborhood):
                prob_vector[i] += 0.5 * min_prob

        return prob_vector

    def load_corpus(self):
        """
        Load the corpus from a JSON file and create a DataFrame with paragraphs.

        This function loads the corpus from a JSON file specified by self.corpus_path.
        If self.corpus_path is None, it constructs a default path based on self.corpus_name.
        It then creates a pandas DataFrame with a 'paragraph' column containing the
        title and text of each document in the corpus.

        Depends on:
            - self.corpus_path: Path to the corpus JSON file
            - self.corpus_name: Name of the corpus (used if corpus_path is None)

        Modifies:
            - self.corpus_path: Sets the path to the corpus file
            - self.corpus: Stores the loaded corpus data
            - self.dataset_df: Creates a DataFrame with paragraphs from the corpus

        Raises:
            AssertionError: If the corpus file is not found at the specified path

        Returns:
            None
        """
        if self.corpus_path is None:
            self.corpus_path = (
                self.config.data_directory / f"{self.corpus_name}_corpus.json"
            )

        assert os.path.isfile(self.corpus_path), "Corpus file not found"
        with open(self.corpus_path, "r") as f:
            self.corpus = json.load(f)
        self.dataset_df = pd.DataFrame()
        self.dataset_df["paragraph"] = [
            f"{p['title']}\n{p['text']}" for p in self.corpus
        ]

    def load_index_files(self):
        """
        Load index files and initialize various data structures for the HippoRAG system.

        This function performs the following tasks:
        1. Loads extraction results from JSON files
        2. Initializes DataFrames and dictionaries for graph representation
        3. Loads graph components such as nodes, edges, and relations
        4. Sets up matrices for efficient graph operations

        Depends on:
            - self.corpus_name: Name of the corpus
            - self.extraction_type: Type of extraction used
            - self.extraction_model_name_processed: Processed name of the extraction model
            - self.graph_type: Type of graph being used
            - self.graph_creating_retriever_name_processed: Processed name of the graph-creating retriever

        Modifies:
            - self.extracted_triples: Stores extracted triples from the corpus
            - self.dataset_df: DataFrame containing passage information
            - self.kb_node_phrase_to_id: Dictionary mapping phrases to node IDs
            - self.lose_fact_dict: Dictionary of facts with their importance scores
            - self.relations_dict: Dictionary of relations in the graph
            - self.lose_facts: List of facts sorted by importance
            - self.phrases: Array of all phrases in the graph
            - self.docs_to_facts: Dictionary mapping documents to facts
            - self.facts_to_phrases: Dictionary mapping facts to phrases
            - self.docs_to_facts_mat: Sparse matrix of document-fact relationships
            - self.facts_to_phrases_mat: Sparse matrix of fact-phrase relationships
            - self.doc_to_phrases_mat: Sparse matrix of document-phrase relationships
            - self.phrase_to_num_doc: Array of document counts for each phrase
            - self.graph_plus: Dictionary containing the full graph structure

        Raises:
            - Logs critical error if no extraction files are found
            - Logs warnings if specific graph components are not found

        Returns:
            None
        """
        index_file_pattern = os.path.join(
            self.config.output_directory,
            "openie_{}_results_{}_{}_*.json".format(
                self.corpus_name,
                self.extraction_type,
                self.extraction_model_name_processed,
            ),
        )
        possible_files = glob(index_file_pattern)
        if len(possible_files) == 0:
            self.logger.critical(
                f"No extraction files found: {index_file_pattern} ; please check if working directory is correct or if the extraction has been done."
            )
            return
        max_samples = np.max(
            [
                int(
                    file.split("{}_".format(self.extraction_model_name_processed))[
                        1
                    ].split(".json")[0]
                )
                for file in possible_files
            ]
        )
        extracted_file = json.load(
            open(
                os.path.join(
                    self.config.output_directory,
                    "openie_{}_results_{}_{}_{}.json".format(
                        self.corpus_name,
                        self.extraction_type,
                        self.extraction_model_name_processed,
                        max_samples,
                    ),
                ),
                "r",
            )
        )

        self.extracted_triples = extracted_file["docs"]

        self.dataset_df = pd.DataFrame([p["passage"] for p in self.extracted_triples])
        self.dataset_df["paragraph"] = [s["passage"] for s in self.extracted_triples]

        self.extraction_type = (
            self.extraction_type + "_" + self.extraction_model_name_processed
        )

        self.kb_node_phrase_to_id = pickle.load(
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_phrase_dict_{phrase_type}_{extraction_type}.subset.p",
                    dataset=self.config.dataset_name,
                    graph_type=self.config.graph_type,
                    phrase_type=self.config.phrase_type,
                    extraction_type=self.config.extraction_type,
                ),
                "rb",
            )
        )
        self.lose_fact_dict = pickle.load(
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_fact_dict_{phrase_type}_{extraction_type}.subset.p",
                    dataset=self.config.dataset_name,
                    graph_type=self.config.graph_type,
                    phrase_type=self.config.phrase_type,
                    extraction_type=self.config.extraction_type,
                ),
                "rb",
            )
        )

        try:
            self.relations_dict = pickle.load(
                open(
                    get_output_path(
                        "{dataset}_{graph_type}_graph_relation_dict_{phrase_type}_{extraction_type}_{retriever}.subset.p",
                        dataset=self.config.dataset_name,
                        graph_type=self.config.graph_type,
                        phrase_type=self.config.phrase_type,
                        extraction_type=self.config.extraction_type,
                        retriever=self.graph_creating_retriever_name_processed,
                    ),
                    "rb",
                )
            )
        except:
            pass

        self.lose_facts = list(self.lose_fact_dict.keys())
        self.lose_facts = [
            self.lose_facts[i] for i in np.argsort(list(self.lose_fact_dict.values()))
        ]
        self.phrases = np.array(list(self.kb_node_phrase_to_id.keys()))[
            np.argsort(list(self.kb_node_phrase_to_id.values()))
        ]

        self.docs_to_facts = pickle.load(
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_doc_to_facts_{phrase_type}_{extraction_type}.subset.p",
                    dataset=self.config.dataset_name,
                    graph_type=self.config.graph_type,
                    phrase_type=self.config.phrase_type,
                    extraction_type=self.config.extraction_type,
                ),
                "rb",
            )
        )
        self.facts_to_phrases = pickle.load(
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_facts_to_phrases_{phrase_type}_{extraction_type}.subset.p",
                    dataset=self.config.dataset_name,
                    graph_type=self.config.graph_type,
                    phrase_type=self.config.phrase_type,
                    extraction_type=self.config.extraction_type,
                ),
                "rb",
            )
        )

        self.docs_to_facts_mat = pickle.load(
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_doc_to_facts_csr_{phrase_type}_{extraction_type}.subset.p",
                    dataset=self.config.dataset_name,
                    graph_type=self.config.graph_type,
                    phrase_type=self.config.phrase_type,
                    extraction_type=self.config.extraction_type,
                ),
                "rb",
            )
        )  # (num docs, num facts)
        self.facts_to_phrases_mat = pickle.load(
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_facts_to_phrases_csr_{phrase_type}_{extraction_type}.subset.p",
                    dataset=self.config.dataset_name,
                    graph_type=self.config.graph_type,
                    phrase_type=self.config.phrase_type,
                    extraction_type=self.config.extraction_type,
                ),
                "rb",
            )
        )  # (num facts, num phrases)

        self.doc_to_phrases_mat = self.docs_to_facts_mat.dot(self.facts_to_phrases_mat)
        self.doc_to_phrases_mat[self.doc_to_phrases_mat.nonzero()] = 1
        self.phrase_to_num_doc = self.doc_to_phrases_mat.sum(0).T

        graph_file_path = os.path.join(
            self.config.output_directory,
            f"{self.config.dataset_name}_facts_and_sim_graph_plus_{self.config.sim_threshold}_thresh_{self.config.phrase_type}_{self.config.extraction_type}_{self.graph_creating_retriever_name_processed}.subset.p"
        )
        if os.path.isfile(graph_file_path):
            self.graph_plus = pickle.load(open(graph_file_path, "rb"))
            self.logger.info(f"Loaded graph from {graph_file_path}")
        else:
            self.logger.warning(f"Graph file not found: {graph_file_path}")
            self.logger.info("Attempting to load individual graph components...")

            # Load individual components
            components = [
                "phrase_dict",
                "fact_dict",
                "doc_to_facts",
                "facts_to_phrases",
            ]
            for component in components:
                component_path = os.path.join(
                    self.config.output_directory,
                    f"{self.config.dataset_name}_{self.config.graph_type}_graph_{component}_{self.config.phrase_type}_{self.config.extraction_type}.subset.p",
                )
                if os.path.isfile(component_path):
                    with open(component_path, "rb") as f:
                        setattr(self, f"{component}_dict", pickle.load(f))
                    self.logger.info(f"Loaded {component} from {component_path}")
                else:
                    self.logger.warning(
                        f"{component.capitalize()} file not found: {component_path}"
                    )

    def get_phrases_in_doc_str(self, doc: str) -> List[str]:
        """
        Retrieves the phrases present in a given document string.

        This function searches for phrases in the provided document string by first
        finding the document ID in the dataset and then using the document-to-phrases
        matrix to identify relevant phrases.

        Args:
            doc (str): The document string to search for phrases.

        Returns:
            List[str]: A list of phrases found in the document. Returns an empty list
                       if the document is not found or if an error occurs.

        Depends on:
            - self.dataset_df: DataFrame containing document information.
            - self.doc_to_phrases_mat: Matrix mapping documents to phrases.
            - self.phrases: Array of all available phrases.

        Note:
            This function assumes that the necessary data structures (dataset_df,
            doc_to_phrases_mat, and phrases) have been properly initialized.
        """
        try:
            # Find doc id from self.dataset_df
            doc_id = self.dataset_df[self.dataset_df.paragraph == doc].index[0]
            phrase_ids = self.doc_to_phrases_mat[[doc_id], :].nonzero()[1].tolist()
            return [self.phrases[phrase_id] for phrase_id in phrase_ids]
        except:
            return []

    def get_passage_by_idx(self, passage_idx: int) -> str:
        """
        Retrieves the passage text for a given passage index.

        This function depends on the self.dataset_df DataFrame, which should contain
        a 'paragraph' column with the passage texts.

        Args:
            passage_idx (int): The index of the passage to retrieve.

        Returns:
            str: The text of the passage corresponding to the given index.

        Raises:
            IndexError: If the passage_idx is out of bounds for the DataFrame.

        Note:
            Ensure that self.dataset_df is properly initialized before calling this function.
        """
        return self.dataset_df.iloc[passage_idx]["paragraph"]

    def get_extraction_by_passage_idx(
        self, passage_idx: Union[str, int], chunk: bool = False
    ) -> Optional[Dict]:
        """
        Retrieves the extraction for a given passage index from self.extracted_triples.

        This function searches through self.extracted_triples to find an item that matches
        the given passage_idx. It can handle both chunked and non-chunked extractions.

        Args:
            passage_idx (Union[str, int]): The index of the passage to search for.
            chunk (bool, optional): If True, also matches partial indices for chunked passages.
                Defaults to False.

        Returns:
            Optional[Dict]: The matching extraction item if found, None otherwise.

        Depends on:
            - self.extracted_triples: A list of dictionaries containing extracted triples.

        Note:
            When chunk is True, it will match both exact indices and indices that start
            with the given passage_idx followed by an underscore (for chunked passages).
        """
        for item in self.extracted_triples:
            if not chunk and item["idx"] == passage_idx:
                return item
            elif chunk and (
                item["idx"] == passage_idx
                or item["idx"].startswith(str(passage_idx) + "_")
            ):
                return item
        return None

    def build_graph(self):
        """
        Builds a graph representation of the knowledge base.

        This function constructs a graph from the existing `self.graph_plus` dictionary,
        creating adjacency lists and an igraph Graph object. It depends on the following
        class attributes:
        - self.graph_plus: A dictionary of edges and their weights
        - self.kb_node_phrase_to_id: A mapping of phrases to node IDs
        - self.logger: A logging object for information output

        The function updates or creates the following attributes:
        - self.graph_plus: A refined version of the input graph
        - self.kg_adj_list: An adjacency list representation of the graph
        - self.kg_inverse_adj_list: An inverse adjacency list
        - self.g: An igraph Graph object representing the final graph

        The function does not return any value but modifies the class state.
        """
        edges = set()

        new_graph_plus = {}
        self.kg_adj_list = defaultdict(dict)
        self.kg_inverse_adj_list = defaultdict(dict)

        for edge, weight in tqdm(
            self.graph_plus.items(), total=len(self.graph_plus), desc="Building Graph"
        ):
            edge1 = edge[0]
            edge2 = edge[1]

            if (edge1, edge2) not in edges and edge1 != edge2:
                new_graph_plus[(edge1, edge2)] = self.graph_plus[(edge[0], edge[1])]
                edges.add((edge1, edge2))
                self.kg_adj_list[edge1][edge2] = self.graph_plus[(edge[0], edge[1])]
                self.kg_inverse_adj_list[edge2][edge1] = self.graph_plus[
                    (edge[0], edge[1])
                ]

        self.graph_plus = new_graph_plus

        edges = list(edges)

        n_vertices = len(self.kb_node_phrase_to_id)
        self.g = ig.Graph(n_vertices, edges)

        self.g.es["weight"] = [self.graph_plus[(v1, v3)] for v1, v3 in edges]
        self.logger.info(
            f"Graph built: num vertices: {n_vertices}, num_edges: {len(edges)}"
        )

    def load_node_vectors(self):
        """
        Load node vectors for the knowledge graph.

        This function attempts to load node vectors from a pre-encoded string file.
        If that file doesn't exist, it generates and saves new embeddings.

        The function depends on:
        - self.linking_retriever_name_processed: Processed name of the linking retriever
        - self.linking_retriever_name: Name of the linking retriever
        - self.phrases: List of phrases to be encoded
        - self.embed_model: Model used for text encoding

        The function modifies:
        - self.kb_node_phrase_embeddings: Matrix of phrase embeddings

        Outputs:
        - Logs information about the loading or saving process

        Note:
        - For 'colbertv2' linking retriever, the function returns early without loading vectors
        """
        retriever_name_processed = self.config.retriever_name.replace("/", "-")
        encoded_string_path = os.path.join(
            self.config.vector_directory,
            f"{retriever_name_processed}_",
            "encoded_strings.txt",
        )
        if os.path.isfile(encoded_string_path):
            self.load_node_vectors_from_string_encoding_cache(encoded_string_path)
        else:  # use another way to load node vectors
            if self.config.retriever_name == "colbertv2":
                return
            kb_node_phrase_embeddings_path = os.path.join(
                self.config.vector_directory,
                f"{retriever_name_processed}_kb_node_phrase_embeddings.p",
            )

            self.kb_node_phrase_embeddings = self.embed_model.encode_text(
                self.phrases, return_cpu=True, return_numpy=True, norm=True
            )
            pickle.dump(
                self.kb_node_phrase_embeddings,
                open(kb_node_phrase_embeddings_path, "wb"),
            )
            self.logger.info(
                "Saved phrase embeddings to: "
                + kb_node_phrase_embeddings_path
                + ", shape: "
                + str(self.kb_node_phrase_embeddings.shape)
            )

    def load_node_vectors_from_string_encoding_cache(
        self, string_file_path: str
    ) -> None:
        """
        Load node vectors from a pre-encoded string file and process them for use in the knowledge graph.

        This function loads pre-encoded vectors, normalizes them, and maps them to the phrases in the knowledge base.

        Args:
            string_file_path (str): Path to the file containing encoded strings.

        Depends on:
            - self.linking_retriever_name_processed: Processed name of the linking retriever.
            - self.phrases: List of phrases in the knowledge base.
            - self.kb_node_phrase_to_id: Dictionary mapping phrases to their IDs in the knowledge base.

        Modifies:
            - self.strings: List of encoded strings.
            - self.string_to_id: Dictionary mapping strings to their indices.
            - self.kb_node_phrase_embeddings: NumPy array of phrase embeddings for the knowledge base.

        Outputs:
            - Logs information about the loading process and any phrases without vectors.

        Note:
            This function assumes that the vector files are stored in a specific directory structure
            and naming convention based on the linking_retriever_name_processed.
        """
        self.logger.info("Loading node vectors from: " + string_file_path)
        kb_vectors = []
        vector_dir = os.path.join(
            self.config.vector_directory,
            f"{self.linking_retriever_name_processed}_mean",
        )

        for i in range(len(glob(os.path.join(vector_dir, "vecs_*")))):
            vector_file = os.path.join(vector_dir, f"vecs_{i}.p")
            kb_vectors.append(torch.Tensor(pickle.load(open(vector_file, "rb"))))
        self.strings = open(string_file_path, "r").readlines()
        # for i in range(
        #     len(
        #         glob(
        #             "../data/lm_vectors/{}_mean/vecs_*".format(
        #                 self.linking_retriever_name_processed
        #             )
        #         )
        #     )
        # ):
        #     kb_vectors.append(
        #         torch.Tensor(
        #             pickle.load(
        #                 open(
        #                     "../data/lm_vectors/{}_mean/vecs_{}.p".format(
        #                         self.linking_retriever_name_processed, i
        #                     ),
        #                     "rb",
        #                 )
        #             )
        #         )
        #     )
        kb_mat = torch.cat(kb_vectors)  # a matrix of phrase vectors
        self.strings = [s.strip() for s in self.strings]
        self.string_to_id = {string: i for i, string in enumerate(self.strings)}
        kb_mat = kb_mat.T.divide(torch.linalg.norm(kb_mat, dim=1)).T
        kb_mat = kb_mat.to("cuda")
        kb_only_indices = []
        num_non_vector_phrases = 0
        for i in range(len(self.kb_node_phrase_to_id)):
            phrase = self.phrases[i]
            if phrase not in self.string_to_id:
                num_non_vector_phrases += 1

            phrase_id = self.string_to_id.get(phrase, 0)
            kb_only_indices.append(phrase_id)
        self.kb_node_phrase_embeddings = kb_mat[
            kb_only_indices
        ]  # a matrix of phrase vectors
        self.kb_node_phrase_embeddings = self.kb_node_phrase_embeddings.cpu().numpy()
        self.logger.info(
            "{} phrases did not have vectors.".format(num_non_vector_phrases)
        )

    def get_dpr_doc_embedding(self):
        """
        Retrieve or compute document embeddings for the corpus.

        This function either loads pre-computed document embeddings from a cache file
        or computes new embeddings using the current embedding model. The embeddings
        are stored in the `self.doc_embedding_mat` attribute.

        Depends on:
            - self.embed_model: The embedding model used to encode text.
            - self.dataset_df: DataFrame containing the corpus documents.
            - self.logger: Logger object for recording information and errors.

        Modifies:
            - self.doc_embedding_mat: Matrix of document embeddings.

        Outputs:
            - Logs information about loading or saving embeddings.

        Note:
            The function attempts to save newly computed embeddings to a cache file
            for future use, creating the necessary directory structure if it doesn't exist.
        """
        cache_filename = os.path.join(
            self.config.vector_directory, f"{self.linking_retriever_name_processed}_cls/cache.p"
        )
        if os.path.exists(cache_filename):
            self.doc_embedding_mat = pickle.load(open(cache_filename, "rb"))
            self.logger.info(
                f"Loaded doc embeddings from {cache_filename}, shape: {self.doc_embedding_mat.shape}"
            )
        else:
            self.doc_embedding_mat = self.embed_model.encode_text(
                self.dataset_df["paragraph"].tolist(),
                return_cpu=True,
                return_numpy=True,
                norm=True,
            )

            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)

            try:
                pickle.dump(self.doc_embedding_mat, open(cache_filename, "wb"))
                self.logger.info(
                    f"Saved doc embeddings to {cache_filename}, shape: {self.doc_embedding_mat.shape}"
                )
            except IOError as e:
                self.logger.error(f"Error saving doc embeddings: {e}")

    def run_pagerank_igraph_chunk(self, reset_prob_chunk):
        """
        Run personalized PageRank on the graph for a chunk of reset probabilities.

        This function computes personalized PageRank scores for each reset probability
        in the given chunk. It uses the igraph library's implementation of PageRank.

        Args:
            reset_prob_chunk (list): A list of reset probability vectors, where each
                                     vector corresponds to a personalized PageRank computation.

        Returns:
            np.ndarray: A 2D array of PageRank probabilities, where each row corresponds
                        to the PageRank scores for one reset probability vector.

        Depends on:
            - self.g: An igraph Graph object representing the knowledge graph.
            - self.kb_node_phrase_to_id: A dictionary mapping phrases to node IDs.
            - self.damping: The damping factor for PageRank (should be between 0 and 1).

        Note:
            This function uses tqdm to display a progress bar during computation.
            The PageRank computation is performed using the 'prpack' implementation,
            which is typically faster for larger graphs.
        """
        pageranked_probabilities = []

        for reset_prob in tqdm(reset_prob_chunk, desc="pagerank chunk"):
            pageranked_probs = self.g.personalized_pagerank(
                vertices=range(len(self.kb_node_phrase_to_id)),
                damping=self.damping,
                directed=False,
                weights="weight",
                reset=reset_prob,
                implementation="prpack",
            )

            pageranked_probabilities.append(np.array(pageranked_probs))

        return np.array(pageranked_probabilities)

    def get_colbert_max_score(self, query: str) -> float:
        """
        Calculate the maximum ColBERT score for a given query.

        This function computes the maximum possible ColBERT score for a query
        by encoding the query and comparing it with itself.

        Args:
            query (str): The input query string.

        Returns:
            float: The maximum ColBERT score for the query.

        Depends on:
            - self.phrase_searcher: A ColBERT searcher object for encoding queries.

        Note:
            This function uses the ColBERT model to encode the query and compute
            the maximum score. It's useful for normalizing ColBERT scores.
        """
        queries_ = [query]
        encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)
        encoded_doc = self.phrase_searcher.checkpoint.docFromText(queries_).float()
        max_score = (
            encoded_query[0]
            .matmul(encoded_doc[0].T)
            .max(dim=1)
            .values.sum()
            .detach()
            .cpu()
            .numpy()
        )

        return max_score

    def get_colbert_real_score(self, query: str, doc: str) -> float:
        """
        Calculate the ColBERT score between a query and a document.

        This function computes the ColBERT score for a given query-document pair
        by encoding both the query and the document and calculating their similarity.

        Args:
            query (str): The input query string.
            doc (str): The document string to compare against the query.

        Returns:
            float: The ColBERT score representing the similarity between the query and the document.

        Depends on:
            - self.phrase_searcher: A ColBERT searcher object for encoding queries and documents.

        Note:
            This function uses the ColBERT model to encode both the query and the document,
            then computes their similarity score. It's useful for ranking documents
            with respect to a given query.
        """
        queries_ = [query]
        encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)

        docs_ = [doc]
        encoded_doc = self.phrase_searcher.checkpoint.docFromText(docs_).float()

        real_score = (
            encoded_query[0]
            .matmul(encoded_doc[0].T)
            .max(dim=1)
            .values.sum()
            .detach()
            .cpu()
            .numpy()
        )

        return real_score

    def link_node_by_colbertv2(
        self, query_ner_list: List[str]
    ) -> Tuple[np.ndarray, Dict[Tuple[str, str], float]]:
        """
        Link named entities in the query to nodes in the knowledge graph using ColBERTv2.

        This function takes a list of named entities extracted from a query and links them
        to the most relevant phrases in the knowledge graph using the ColBERTv2 model.

        Args:
            query_ner_list (List[str]): A list of named entities extracted from the query.

        Returns:
            Tuple[np.ndarray, Dict[Tuple[str, str], float]]:
                - A numpy array representing the relevance of each phrase in the knowledge graph.
                - A dictionary mapping (query, linked_phrase) pairs to their relevance scores.

        Depends on:
            - self.phrase_searcher: A ColBERT searcher object for encoding and searching phrases.
            - self.phrases: An array of all phrases in the knowledge graph.
            - self.node_specificity: A boolean indicating whether to use node specificity weighting.
            - self.phrase_to_num_doc: An array containing the document frequency of each phrase.
            - self.get_colbert_max_score(): A method to get the maximum ColBERT score for normalization.

        Note:
            This function uses ColBERTv2 to encode queries and phrases, compute similarity scores,
            and link named entities to the most relevant phrases in the knowledge graph.
            It also applies node specificity weighting if enabled.
        """
        phrase_ids = []
        max_scores = []

        for query in query_ner_list:
            queries = Queries(path=None, data={0: query})

            queries_ = [query]
            encoded_query = self.phrase_searcher.encode(
                queries_, full_length_search=False
            )

            max_score = self.get_colbert_max_score(query)

            ranking = self.phrase_searcher.search_all(queries, k=1)
            for phrase_id, rank, score in ranking.data[0]:
                phrase = self.phrases[phrase_id]
                phrases_ = [phrase]
                encoded_doc = self.phrase_searcher.checkpoint.docFromText(
                    phrases_
                ).float()
                real_score = (
                    encoded_query[0]
                    .matmul(encoded_doc[0].T)
                    .max(dim=1)
                    .values.sum()
                    .detach()
                    .cpu()
                    .numpy()
                )

                phrase_ids.append(phrase_id)
                max_scores.append(real_score / max_score)

        # Create a vector (num_doc) with weights at the indices of the retrieved documents and 0s elsewhere
        top_phrase_vec = np.zeros(len(self.phrases))

        for phrase_id in phrase_ids:
            if self.node_specificity:
                if self.phrase_to_num_doc[phrase_id] == 0:
                    weight = 1
                else:
                    weight = 1 / self.phrase_to_num_doc[phrase_id]
                top_phrase_vec[phrase_id] = weight
            else:
                top_phrase_vec[phrase_id] = 1.0

        return top_phrase_vec, {
            (query, self.phrases[phrase_id]): max_score
            for phrase_id, max_score, query in zip(
                phrase_ids, max_scores, query_ner_list
            )
        }

    def link_node_by_dpr(
        self, query_ner_list: List[str]
    ) -> Tuple[np.ndarray, Dict[Tuple[str, str], float]]:
        """
        Links named entities from the query to the most similar phrases in the knowledge graph using DPR (Dense Passage Retrieval).

        This function embeds the query named entities, computes similarity scores with knowledge graph phrases,
        and returns a weighted vector of phrase importances along with a mapping of linked phrases and their scores.

        Args:
            query_ner_list (List[str]): A list of named entities extracted from the query.

        Returns:
            Tuple[np.ndarray, Dict[Tuple[str, str], float]]:
                - np.ndarray: A vector of weights for all phrases in the knowledge graph.
                - Dict[Tuple[str, str], float]: A mapping of (query_phrase, linked_phrase) pairs to their similarity scores.

        Depends on:
            - self.embed_model: An embedding model to encode text.
            - self.kb_node_phrase_embeddings: Pre-computed embeddings for knowledge graph phrases.
            - self.phrases: A list of all phrases in the knowledge graph.
            - self.node_specificity: A boolean flag indicating whether to use node specificity weighting.
            - self.phrase_to_num_doc: A mapping of phrases to their document frequency.

        Note:
            This function uses dense embeddings to link query entities to knowledge graph phrases.
            It can optionally apply node specificity weighting to adjust the importance of frequent phrases.
        """
        query_ner_embeddings = self.embed_model.encode_text(
            query_ner_list, return_cpu=True, return_numpy=True, norm=True
        )

        # Get Closest Entity Nodes
        prob_vectors = np.dot(
            query_ner_embeddings, self.kb_node_phrase_embeddings.T
        )  # (num_ner, dim) x (num_phrases, dim).T -> (num_ner, num_phrases)

        linked_phrase_ids = []
        max_scores = []

        for prob_vector in prob_vectors:
            phrase_id = np.argmax(prob_vector)  # the phrase with the highest similarity
            linked_phrase_ids.append(phrase_id)
            max_scores.append(prob_vector[phrase_id])

        # create a vector (num_phrase) with 1s at the indices of the linked phrases and 0s elsewhere
        # if node_specificity is True, it's not one-hot but a weight
        all_phrase_weights = np.zeros(len(self.phrases))

        for phrase_id in linked_phrase_ids:
            if self.node_specificity:
                if (
                    self.phrase_to_num_doc[phrase_id] == 0
                ):  # just in case the phrase is not recorded in any documents
                    weight = 1
                else:  # the more frequent the phrase, the less weight it gets
                    weight = 1 / self.phrase_to_num_doc[phrase_id]

                all_phrase_weights[phrase_id] = weight
            else:
                all_phrase_weights[phrase_id] = 1.0

        linking_score_map = {
            (query_phrase, self.phrases[linked_phrase_id]): max_score
            for linked_phrase_id, max_score, query_phrase in zip(
                linked_phrase_ids, max_scores, query_ner_list
            )
        }
        return all_phrase_weights, linking_score_map
