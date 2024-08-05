"""
This module creates a graph representation of extracted information from documents.

Key functionalities:
1. Processes extracted triples and entities from documents.
2. Creates knowledge bases for phrases and relations.
3. Builds a graph structure representing relationships between phrases.
4. Generates sparse matrices to efficiently represent document-to-fact and fact-to-phrase relationships.
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

def load_extracted_data(config: ConfigManager) -> dict:
    """
    Load extracted data from JSON files based on configuration.

    Args:
        config (ConfigManager): Configuration object containing settings.

    Returns:
        dict: Loaded extracted data.
    """
    possible_files = glob(
        get_output_path(
            "openie_{dataset}_results_{extraction_type}_{extraction_model}_*.json",
            dataset=config.dataset_name,
            extraction_type=config.extraction_type,
            extraction_model=config.llm_model,
        )
    )
    assert len(possible_files) > 0, "No files found"
    max_samples = np.max(
        [
            int(file.split(f"{config.llm_model}_")[1].split(".json")[0])
            for file in possible_files
        ]
    )
    return json.load(
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

def process_triples(extracted_triples: list) -> tuple:
    """
    Process extracted triples and entities.

    Args:
        extracted_triples (list): List of extracted triples.

    Returns:
        tuple: Processed data including passage_json, phrases, entities, relations, etc.
    """
    passage_json = []
    phrases = []
    entities = []
    relations = {}
    incorrectly_formatted_triples = []
    triples_wo_ner_entity = []
    triple_tuples = []
    full_neighborhoods = {}
    correct_wiki_format = 0

    for _, row in tqdm(enumerate(extracted_triples), total=len(extracted_triples)):
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

        doc_json["entities"] = list(set(doc_entities))
        doc_json["clean_triples"] = clean_triples
        doc_json["noisy_triples"] = unclean_triples
        triple_tuples.append(clean_triples)

        passage_json.append(doc_json)

    return (passage_json, phrases, entities, relations, incorrectly_formatted_triples,
            triples_wo_ner_entity, triple_tuples, full_neighborhoods, correct_wiki_format)

def create_knowledge_bases(unique_phrases: list, unique_relations: list, config: ConfigManager):
    """
    Create knowledge base DataFrames for phrases and relations.

    Args:
        unique_phrases (list): List of unique phrases.
        unique_relations (list): List of unique relations.
        config (ConfigManager): Configuration object.

    Returns:
        None
    """
    kb = pd.DataFrame(unique_phrases, columns=["strings"])
    kb2 = copy.deepcopy(kb)
    kb["type"] = "query"
    kb2["type"] = "kb"
    kb_full = pd.concat([kb, kb2])
    kb_full.to_csv(config.kb_to_kb_path, sep="\t")

    rel_kb = pd.DataFrame(unique_relations, columns=["strings"])
    rel_kb2 = copy.deepcopy(rel_kb)
    rel_kb["type"] = "query"
    rel_kb2["type"] = "kb"
    rel_kb_full = pd.concat([rel_kb, rel_kb2])
    rel_kb_full.to_csv(config.rel_kb_to_kb_path, sep="\t")

def create_graph_structure(unique_phrases: list, triple_tuples: list) -> tuple:
    """
    Create graph structure including nodes, facts, and dictionaries.

    Args:
        unique_phrases (list): List of unique phrases.
        triple_tuples (list): List of triple tuples.

    Returns:
        tuple: node_json, kb_phrase_dict, lose_facts, lose_fact_dict, fact_json
    """
    node_json = [{"idx": i, "name": p} for i, p in enumerate(unique_phrases)]
    kb_phrase_dict = {p: i for i, p in enumerate(unique_phrases)}

    lose_facts = []
    for triples in triple_tuples:
        lose_facts.extend([tuple(t) for t in triples])

    lose_fact_dict = {f: i for i, f in enumerate(lose_facts)}
    fact_json = [
        {"idx": i, "head": t[0], "relation": t[1], "tail": t[2]}
        for i, t in enumerate(lose_facts)
    ]

    return node_json, kb_phrase_dict, lose_facts, lose_fact_dict, fact_json

def create_adjacency_matrices(triple_tuples: list, lose_fact_dict: dict, kb_phrase_dict: dict, config: ConfigManager) -> tuple:
    """
    Create adjacency matrices for the graph.

    Args:
        triple_tuples (list): List of triple tuples.
        lose_fact_dict (dict): Dictionary mapping facts to indices.
        kb_phrase_dict (dict): Dictionary mapping phrases to indices.
        config (ConfigManager): Configuration object.

    Returns:
        tuple: docs_to_facts, facts_to_phrases, graph, graph_json
    """
    docs_to_facts = {}
    facts_to_phrases = {}
    graph = {}
    graph_json = {}

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

    return docs_to_facts, facts_to_phrases, graph, graph_json

def augment_graph_with_similarity(graph: dict, kb_phrase_dict: dict, config: ConfigManager) -> tuple:
    """
    Augment the graph with cosine similarity-based synonymy edges.

    Args:
        graph (dict): Original graph.
        kb_phrase_dict (dict): Dictionary mapping phrases to indices.
        config (ConfigManager): Configuration object.

    Returns:
        tuple: graph_plus, relations, graph_json, synonym_candidates
    """
    if not config.cosine_sim_edges:
        return graph, {}, {}, []

    # Load pre-computed nearest neighbors
    kb_similarity = pickle.load(open(config.nearest_neighbors_path, "rb"))
    kb_similarity = {processing_phrases(k): v for k, v in kb_similarity.items()}

    graph_plus = copy.deepcopy(graph)
    relations = {}
    graph_json = {}
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
                    if score < config.threshold or num_nns > 100:
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

    return graph_plus, relations, graph_json, synonym_candidates

def save_graph_data(graph_data: dict, config: ConfigManager):
    """
    Save graph data to files.

    Args:
        graph_data (dict): Dictionary containing various graph data to be saved.
        config (ConfigManager): Configuration object.

    Returns:
        None
    """
    for key, data in graph_data.items():
        file_path = get_output_path(
            f"{{dataset}}_{{graph_type}}_graph_{key}_{{phrase_type}}_{{extraction_type}}.subset.p",
            dataset=config.dataset_name,
            graph_type=config.graph_type,
            phrase_type=config.phrase_type,
            extraction_type=config.extraction_type,
        )
        pickle.dump(data, open(file_path, "wb"))

def create_graph(config: ConfigManager):
    """
    Create a graph representation of extracted information from documents.

    Args:
        config (ConfigManager): Configuration object containing settings for graph creation.

    Returns:
        None. The function saves the created graph and related data to files.
    """
    extracted_file = load_extracted_data(config)
    extracted_triples = extracted_file["docs"]

    processed_data = process_triples(extracted_triples)
    (passage_json, phrases, entities, relations, incorrectly_formatted_triples,
     triples_wo_ner_entity, triple_tuples, full_neighborhoods, correct_wiki_format) = processed_data

    unique_phrases = list(np.unique(entities))
    unique_relations = np.unique(list(relations.values()) + ["equivalent"])

    create_knowledge_bases(unique_phrases, unique_relations, config)

    if config.create_graph_flag:
        node_json, kb_phrase_dict, lose_facts, lose_fact_dict, fact_json = create_graph_structure(unique_phrases, triple_tuples)

        docs_to_facts, facts_to_phrases, graph, graph_json = create_adjacency_matrices(triple_tuples, lose_fact_dict, kb_phrase_dict, config)

        graph_plus, relations, graph_json, synonym_candidates = augment_graph_with_similarity(graph, kb_phrase_dict, config)

        # Save various graph data
        graph_data = {
            "doc_to_facts": docs_to_facts,
            "facts_to_phrases": facts_to_phrases,
            "fact_doc_edges": graph,
            "relation_dict": relations,
            "graph_plus": graph_plus,
        }
        save_graph_data(graph_data, config)

        # Print statistics
        print_graph_statistics(phrases, unique_phrases, lose_facts, incorrectly_formatted_triples,
                               triples_wo_ner_entity, lose_fact_dict, entities, relations)

def print_graph_statistics(phrases, unique_phrases, lose_facts, incorrectly_formatted_triples,
                           triples_wo_ner_entity, lose_fact_dict, entities, relations):
    """
    Print statistics about the created graph.

    Args:
        Various graph-related data.

    Returns:
        None
    """
    stat_df = [
        ("Total Phrases", len(phrases)),
        ("Unique Phrases", len(unique_phrases)),
        ("Number of Individual Triples", len(lose_facts)),
        ("Number of Incorrectly Formatted Triples (ChatGPT Error)", len(incorrectly_formatted_triples)),
        ("Number of Triples w/o NER Entities (ChatGPT Error)", len(triples_wo_ner_entity)),
        ("Number of Unique Individual Triples", len(lose_fact_dict)),
        ("Number of Entities", len(entities)),
        ("Number of Relations", len(relations)),
        ("Number of Unique Entities", len(np.unique(entities))),
        ("Number of Synonymy Edges", len([edge for edge in relations.keys() if relations[edge] == "equivalent"])),
        ("Number of Unique Relations", len(np.unique(list(relations.values())))),
    ]
    print(pd.DataFrame(stat_df).set_index(0))

# Main execution
if __name__ == "__main__":
    config = ConfigManager()  # Assume this is properly initialized
    create_graph(config)