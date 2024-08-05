"""
This module creates a graph representation of extracted information from documents.

Key functionalities:
1. Processes extracted triples and entities from documents.
2. Creates knowledge bases for phrases and relations.
3. Builds a graph structure representing relationships between phrases.
4. Generates sparse matrices to efficiently represent:
   - Document-to-fact relationships (docs_to_facts_mat):
     - Creates a sparse matrix where rows represent documents and columns represent facts.
     - Shape is (number of documents, number of facts).
     - A value of 1 in cell (i, j) indicates that fact j appears in document i.
   - Fact-to-phrase relationships (facts_to_phrases_mat):
     - Creates a sparse matrix where rows represent facts and columns represent phrases.
     - Shape is (number of facts, number of unique phrases).
     - A value of 1 in cell (i, j) indicates that phrase j appears in fact i.

Note:
Documents are rows from our corpus
Facts are extracted triples from documents
Phrases are entities in the triples

"""

import copy
import re
import numpy as np

import pandas as pd
from scipy.sparse import csr_array
from glob import glob
import os
import json
from tqdm import tqdm
import pickle

from src.hipporag.processing import get_output_path, processing_phrases
from src.hipporag.config_manager import ConfigManager

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"


def create_graph(
    config: ConfigManager,
):
    """
    Create a graph representation of extracted information from documents.

    This function processes extracted triples and entities from documents to build a graph structure
    representing relationships between phrases. It also generates sparse matrices to efficiently
    represent document-to-fact and fact-to-phrase relationships.

    The graph creation process involves the following key steps:
    1. Load and process extracted triples and entities from JSON files.
    2. Clean and normalize phrases and triples.
    3. Build knowledge bases for phrases and relations.
    4. Create graph structure with nodes (phrases) and edges (relationships between phrases).
    5. Generate sparse matrices for efficient representation of relationships.
    6. Optionally augment the graph with cosine similarity-based synonymy edges.

    Entity and Triple Handling:
    - Entities are extracted from triples and stored in a set for each document.
    - Triples are cleaned and normalized, with each component (subject, predicate, object) processed.
    - Valid triples are added to the graph structure, while incorrectly formatted triples are logged.
    - Entities not found in the Named Entity Recognition (NER) results are also logged.

    Graph Building:
    - Nodes represent unique phrases (entities) extracted from triples.
    - Edges represent relationships between phrases, derived from triples.
    - Additional edges may be added based on cosine similarity between phrases.
    - The graph is represented using dictionaries and sparse matrices for efficiency.

    Outputs:
    - JSON files containing nodes, facts, and overall graph structure.
    - Pickle files with various dictionaries and sparse matrices representing the graph.
    - Statistics about the created graph (number of phrases, triples, entities, etc.).

    Args:
        config (ConfigManager): Configuration object containing settings for graph creation.

    Returns:
        None. The function saves the created graph and related data to files.
    """
    possible_files = glob(
        get_output_path(
            "openie_{dataset}_results_{extraction_type}_{extraction_model}_*.json",
            dataset=config.dataset_name,
            extraction_type=config.extraction_type,
            extraction_model=config.llm_model,
        )
    )
    print(possible_files)
    assert len(possible_files) > 0, "No files found"
    max_samples = np.max(
        [
            int(file.split(f"{config.llm_model}_")[1].split(".json")[0])
            for file in possible_files
        ]
    )
    extracted_file = json.load(
        open(
            get_output_path(
                "openie_{dataset}_results_{extraction_type}_{extraction_model}_{max_samples}.json",
                dataset=config.dataset_name,
                extraction_type=config.extraction_type,
                extraction_model=config.llm_model,
                max_samples=max_samples,
            ),
            "r",
        )
    )

    # Extract triples from the loaded file
    extracted_triples = extracted_file["docs"]
    
    # Process the retriever name for file naming
    processed_retriever_name = config.retriever_name.replace(
        "/", "_"
    ).replace(".", "")

    # Initialize data structures for storing processed information
    passage_json = []
    phrases = []
    entities = []
    relations = {}
    incorrectly_formatted_triples = []
    triples_wo_ner_entity = []
    triple_tuples = []
    full_neighborhoods = {}
    # for logging
    incorrectly_formatted_triples = []
    triples_wo_ner_entity = []
    correct_wiki_format = 0

    # Process each row in the extracted triples
    for i, row in tqdm(enumerate(extracted_triples), total=len(extracted_triples)):
        # Normalize phrase: lowercase, remove non-alphanumeric chars, strip whitespace
        ner_entities = [processing_phrases(p) for p in row["extracted_entities"]]

        triples = row["extracted_triples"]
        doc_json = row

        clean_triples = []
        unclean_triples = []
        doc_entities = set()

        # Process each triple in the current row
        for triple in triples:
            triple = [str(s) for s in triple]

            if len(triple) > 1:
                # Check if the triple has the correct format (subject[0], predicate/relation[1], object[2])
                if len(triple) != 3:
                    # Handle incorrectly formatted triples
                    clean_triple = [processing_phrases(p) for p in triple]
                    incorrectly_formatted_triples.append(triple)
                    unclean_triples.append(triple)
                else:
                    # Process correctly formatted triples
                    clean_triple = [processing_phrases(p) for p in triple]

                    clean_triples.append(clean_triple)
                    phrases.extend(clean_triple)
                    correct_wiki_format += 1

                    head_ent = clean_triple[0]
                    tail_ent = clean_triple[2]

                    # Check if both entities are not in NER entities
                    if head_ent not in ner_entities and tail_ent not in ner_entities:
                        triples_wo_ner_entity.append(triple)

                    # Store relation between head and tail entities
                    relations[(head_ent, tail_ent)] = clean_triple[1]

                    raw_head_ent = triple[0]
                    raw_tail_ent = triple[2]

                    # Build entity neighborhoods
                    # A neighborhood is a set of triples that share an entity
                    # Here we're building neighborhoods for both head and tail entities
                    entity_neighborhood = full_neighborhoods.get(raw_head_ent, set())
                    entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                    full_neighborhoods[raw_head_ent] = entity_neighborhood

                    entity_neighborhood = full_neighborhoods.get(raw_tail_ent, set())
                    entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                    full_neighborhoods[raw_tail_ent] = entity_neighborhood

                    # Add entities to the overall entity list and document-specific set
                    for triple_entity in [clean_triple[0], clean_triple[2]]:
                        entities.append(triple_entity)
                        doc_entities.add(triple_entity)

        # Update document JSON with processed information
        doc_json["entities"] = list(set(doc_entities))
        doc_json["clean_triples"] = clean_triples
        doc_json["noisy_triples"] = unclean_triples
        triple_tuples.append(clean_triples)

        passage_json.append(doc_json)

    # Print statistics about correct Wiki format
    print(f"Correct Wiki Format: {correct_wiki_format} out of {len(extracted_triples)}")

    # Create lists of unique phrases and relations
    unique_phrases = list(np.unique(entities))
    unique_relations = np.unique(list(relations.values()) + ["equivalent"])

    # Create knowledge base (kb) DataFrame for phrases
    kb = pd.DataFrame(unique_phrases, columns=["strings"])
    kb2 = copy.deepcopy(kb)
    kb["type"] = "query"
    kb2["type"] = "kb"
    kb_full = pd.concat([kb, kb2])
    kb_full.to_csv(config.kb_to_kb_path, sep="\t")

    # Create knowledge base (kb) DataFrame for relations
    rel_kb = pd.DataFrame(unique_relations, columns=["strings"])
    rel_kb2 = copy.deepcopy(rel_kb)
    rel_kb["type"] = "query"
    rel_kb2["type"] = "kb"
    rel_kb_full = pd.concat([rel_kb, rel_kb2])
    rel_kb_full.to_csv(config.rel_kb_to_kb_path, sep="\t")

    if config.create_graph_flag:
        print("Creating Graph")

        # Create a list of dictionaries for nodes (phrases)
        node_json = [{"idx": i, "name": p} for i, p in enumerate(unique_phrases)]
        # Create a dictionary mapping phrases to their indices
        kb_phrase_dict = {p: i for i, p in enumerate(unique_phrases)}

        # Flatten the list of triples
        lose_facts = []
        for triples in triple_tuples:
            lose_facts.extend([tuple(t) for t in triples])

        # Create a dictionary mapping facts to their indices
        lose_fact_dict = {f: i for i, f in enumerate(lose_facts)}
        # Create a list of dictionaries for facts
        fact_json = [
            {"idx": i, "head": t[0], "relation": t[1], "tail": t[2]}
            for i, t in enumerate(lose_facts)
        ]

        json.dump(
            passage_json,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_passage_chatgpt_openIE.{phrase_type}_{extraction_type}.subset.json",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "w",
            ),
        )

        json.dump(
            node_json,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_nodes_chatgpt_openIE.{phrase_type}_{extraction_type}.subset.json",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "w",
            ),
        )
        json.dump(
            fact_json,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_clean_facts_chatgpt_openIE.{phrase_type}_{extraction_type}.subset.json",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "w",
            ),
        )

        pickle.dump(
            kb_phrase_dict,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_phrase_dict_{phrase_type}_{extraction_type}.subset.p",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "wb",
            ),
        )
        pickle.dump(
            lose_fact_dict,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_fact_dict_{phrase_type}_{extraction_type}.subset.p",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "wb",
            ),
        )

        graph_json = {}

        docs_to_facts = {}  # (Num Docs, Num Facts)
        facts_to_phrases = {}  # (Num Facts, Num Phrases)
        graph = {}  # (Num Phrases, Num Phrases)

        num_triple_edges = 0

        # Creating Adjacency and Document to Phrase Matrices
        for doc_id, triples in tqdm(enumerate(triple_tuples), total=len(triple_tuples)):
            doc_phrases = []
            fact_edges = []

            # Iterate over triples
            for triple in triples:
                triple = tuple(triple)

                fact_id = lose_fact_dict[triple]

                if len(triple) == 3:
                    relation = triple[1]
                    triple = np.array(triple)[[0, 2]]

                    docs_to_facts[(doc_id, fact_id)] = 1

                    for i, phrase in enumerate(triple):
                        phrase_id = kb_phrase_dict[phrase]
                        doc_phrases.append(phrase_id)

                        facts_to_phrases[(fact_id, phrase_id)] = 1

                        for phrase2 in triple[i + 1 :]:
                            phrase2_id = kb_phrase_dict[phrase2]

                            fact_edge_r = (phrase_id, phrase2_id)
                            fact_edge_l = (phrase2_id, phrase_id)

                            fact_edges.append(fact_edge_r)
                            fact_edges.append(fact_edge_l)

                            graph[fact_edge_r] = (
                                graph.get(fact_edge_r, 0.0) + config.inter_triple_weight
                            )
                            graph[fact_edge_l] = (
                                graph.get(fact_edge_l, 0.0) + config.inter_triple_weight
                            )

                            phrase_edges = graph_json.get(phrase, {})
                            edge = phrase_edges.get(phrase2, ("triple", 0))
                            phrase_edges[phrase2] = ("triple", edge[1] + 1)
                            graph_json[phrase] = phrase_edges

                            phrase_edges = graph_json.get(phrase2, {})
                            edge = phrase_edges.get(phrase, ("triple", 0))
                            phrase_edges[phrase] = ("triple", edge[1] + 1)
                            graph_json[phrase2] = phrase_edges

                            num_triple_edges += 1

        pickle.dump(
            docs_to_facts,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_doc_to_facts_{phrase_type}_{extraction_type}.subset.p",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "wb",
            ),
        )
        pickle.dump(
            facts_to_phrases,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_facts_to_phrases_{phrase_type}_{extraction_type}.subset.p",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "wb",
            ),
        )

        docs_to_facts_mat = csr_array(
            (
                [int(v) for v in docs_to_facts.values()],
                (
                    [int(e[0]) for e in docs_to_facts.keys()],
                    [int(e[1]) for e in docs_to_facts.keys()],
                ),
            ),
            shape=(len(triple_tuples), len(lose_facts)),
        )
        facts_to_phrases_mat = csr_array(
            (
                [int(v) for v in facts_to_phrases.values()],
                (
                    [e[0] for e in facts_to_phrases.keys()],
                    [e[1] for e in facts_to_phrases.keys()],
                ),
            ),
            shape=(len(lose_facts), len(unique_phrases)),
        )

        pickle.dump(
            docs_to_facts_mat,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_doc_to_facts_csr_{phrase_type}_{extraction_type}.subset.p",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "wb",
            ),
        )
        pickle.dump(
            facts_to_phrases_mat,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_facts_to_phrases_csr_{phrase_type}_{extraction_type}.subset.p",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "wb",
            ),
        )

        pickle.dump(
            graph,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_fact_doc_edges_{phrase_type}_{extraction_type}.subset.p",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "wb",
            ),
        )

        print("Loading Vectors")

        # Expanding OpenIE triples with cosine similarity-based synonymy edges
        if config.cosine_sim_edges:
            """
            Augment the graph with cosine similarity-based synonymy edges.
            
            This process involves:
            1. Loading pre-computed nearest neighbors (similarity data)
            2. Creating a new graph with additional similarity edges
            3. Processing phrases and identifying synonym candidates
            4. Adding new edges to the graph based on similarity scores
            
            Args:
                config (ConfigManager): Configuration object containing settings
            
            Returns:
                None (modifies graph_plus, relations, and graph_json in place)
            """
            # Load pre-computed nearest neighbors based on retriever type
            if "colbert" in config.retriever_name:
                kb_similarity = pickle.load(open(config.nearest_neighbors_path, "rb"))
            else:
                kb_similarity = pickle.load(open(config.nearest_neighbors_path, "rb"))

            print("Augmenting Graph from Similarity")

            # Create a deep copy of the original graph to add similarity edges
            graph_plus = copy.deepcopy(graph)

            # Process phrases in the similarity data
            kb_similarity = {processing_phrases(k): v for k, v in kb_similarity.items()}

            synonym_candidates = []

            # Iterate through all phrases in the similarity data
            for phrase in tqdm(kb_similarity.keys(), total=len(kb_similarity)):
                synonyms = []

                # Only process phrases with more than 2 alphanumeric characters
                if len(re.sub("[^A-Za-z0-9]", "", phrase)) > 2:
                    phrase_id = kb_phrase_dict.get(phrase, None)

                    if phrase_id is not None:
                        nns = kb_similarity[phrase]

                        num_nns = 0
                        # Iterate through nearest neighbors and their similarity scores
                        for nn, score in zip(nns[0], nns[1]):
                            nn = processing_phrases(nn)
                            # Stop if score is below threshold or we've processed 100 neighbors
                            if score < config.sim_threshold or num_nns > 100:
                                break
                                
                            if nn != phrase:
                                phrase2_id = kb_phrase_dict.get(nn)

                                if phrase2_id is not None:
                                    phrase2 = nn

                                    # Create a new edge between similar phrases
                                    sim_edge = (phrase_id, phrase2_id)
                                    synonyms.append((nn, score))

                                    # Add the new edge to relations and graph_plus
                                    relations[(phrase, phrase2)] = "equivalent"
                                    graph_plus[sim_edge] = config.similarity_max * score

                                    num_nns += 1

                                    # Update graph_json with the new similarity edge
                                    phrase_edges = graph_json.get(phrase, {})
                                    edge = phrase_edges.get(phrase2, ("similarity", 0))
                                    if edge[0] == "similarity":
                                        phrase_edges[phrase2] = (
                                            "similarity",
                                            edge[1] + score,
                                        )
                                        graph_json[phrase] = phrase_edges

                # Add the phrase and its synonyms to the candidate list
                synonym_candidates.append((phrase, synonyms))

            pickle.dump(
                synonym_candidates,
                open(
                    get_output_path(
                        "{dataset}_similarity_edges_mean_{threshold}_thresh_{phrase_type}_{extraction_type}_{retriever}.subset.p",
                        dataset=config.dataset_name,
                        threshold=config.sim_threshold,
                        phrase_type=config.phrase_type,
                        extraction_type=config.extraction_type,
                        retriever=processed_retriever_name,
                    ),
                    "wb",
                ),
            )
        else:
            graph_plus = graph

        pickle.dump(
            relations,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_relation_dict_{phrase_type}_{extraction_type}_{retriever}.subset.p",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                    retriever=processed_retriever_name,
                ),
                "wb",
            ),
        )

        print("Saving Graph")

        synonymy_edges = set(
            [edge for edge in relations.keys() if relations[edge] == "equivalent"]
        )

        stat_df = [
            ("Total Phrases", len(phrases)),
            ("Unique Phrases", len(unique_phrases)),
            ("Number of Individual Triples", len(lose_facts)),
            (
                "Number of Incorrectly Formatted Triples (ChatGPT Error)",
                len(incorrectly_formatted_triples),
            ),
            (
                "Number of Triples w/o NER Entities (ChatGPT Error)",
                len(triples_wo_ner_entity),
            ),
            ("Number of Unique Individual Triples", len(lose_fact_dict)),
            ("Number of Entities", len(entities)),
            ("Number of Relations", len(relations)),
            ("Number of Unique Entities", len(np.unique(entities))),
            ("Number of Synonymy Edges", len(synonymy_edges)),
            ("Number of Unique Relations", len(unique_relations)),
        ]

        print(pd.DataFrame(stat_df).set_index(0))

        if config.similarity_max == 1.0:
            pickle.dump(
                graph_plus,
                open(
                    get_output_path(
                        "{dataset}_{graph_type}_graph_plus_{threshold}_thresh_{phrase_type}_{extraction_type}_{retriever}.subset.p",
                        dataset=config.dataset_name,
                        graph_type=config.graph_type,
                        threshold=config.sim_threshold,
                        phrase_type=config.phrase_type,
                        extraction_type=config.extraction_type,
                        retriever=processed_retriever_name,
                    ),
                    "wb",
                ),
            )
        else:
            pickle.dump(
                graph_plus,
                open(
                    get_output_path(
                        "{dataset}_{graph_type}_graph_plus_{threshold}_thresh_{phrase_type}_{extraction_type}_sim_max_{similarity_max}_{retriever}.subset.p",
                        dataset=config.dataset_name,
                        graph_type=config.graph_type,
                        threshold=config.sim_threshold,
                        phrase_type=config.phrase_type,
                        extraction_type=config.extraction_type,
                        similarity_max=config.similarity_max,
                        retriever=processed_retriever_name,
                    ),
                    "wb",
                ),
            )

        json.dump(
            graph_json,
            open(
                get_output_path(
                    "{dataset}_{graph_type}_graph_openAI_openIE.{phrase_type}_{extraction_type}.subset.json",
                    dataset=config.dataset_name,
                    graph_type=config.graph_type,
                    phrase_type=config.phrase_type,
                    extraction_type=config.extraction_type,
                ),
                "w",
            ),
        )
