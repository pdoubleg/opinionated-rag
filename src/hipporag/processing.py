import json
import re
import os
import numpy as np
import torch

from src.hipporag.config_manager import ConfigManager

config = ConfigManager()
DEFAULT_BASE_DIRECTORY = config.output_directory


def get_file_name(path):
    return path.split("/")[-1].replace(".jsonl", "").replace(".json", "")


def get_output_path(filename, **kwargs):
    """
    Generate a standardized output file path.

    Args:
        filename (str): Base filename
        **kwargs: Additional parameters to format the filename

    Returns:
        str: Formatted output file path
    """
    return os.path.join(DEFAULT_BASE_DIRECTORY, filename.format(**kwargs))


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def mean_pooling_embedding(input_str: str, tokenizer, model, device="cuda"):
    inputs = tokenizer(
        input_str, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    outputs = model(**inputs)

    embedding = (
        mean_pooling(outputs[0], inputs["attention_mask"]).to("cpu").detach().numpy()
    )
    return embedding


def mean_pooling_embedding_with_normalization(
    input_str, tokenizer, model, device="cuda"
):
    encoding = tokenizer(input_str, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = mean_pooling(outputs[0], attention_mask)
    embeddings = embeddings.T.divide(torch.linalg.norm(embeddings, dim=1)).T

    return embeddings


def processing_phrases(phrase):
    """Normalize phrase: lowercase, remove non-alphanumeric chars, strip whitespace"""
    return re.sub("[^A-Za-z0-9 ]", " ", phrase.lower()).strip()


def extract_json_dict(text):
    pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}"
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return ""
    else:
        return ""


def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def print_formatted_dict(data: dict, max_examples: int = 5):
    
    # Print Named Entities
    print("Named Entities:")
    for entity in data['named_entities']:
        print(f"- {entity}")
    print()

    # Print Linked Node Scores
    print("Linked Node Scores:")
    for node, linked, score in data['linked_node_scores']:
        print(f"- {node} -> {linked} (Score: {score:.2f})")
    print()

    # Print 1-hop Graph for Linked Nodes
    print(f"1-hop Graph for Linked Nodes (top {max_examples}):")
    for item, score in data['1-hop_graph_for_linked_nodes'][:max_examples]:
        if isinstance(score, (int, float)):
            print(f"- {item} (Score: {score:.2f})")
        else:
            print(f"- {item}")
    print()

    # Print Top Ranked Nodes
    print(f"Top Ranked Nodes (top {max_examples}):")
    for node in data['top_ranked_nodes'][:max_examples]:
        print(f"- {node}")
    print()

    # Print Nodes in Retrieved Doc
    print(f"Nodes in Retrieved Doc (top {max_examples} from each document):")
    for i, doc_nodes in enumerate(data['nodes_in_retrieved_doc']):
        print(f"Document {i}:")
        for node in doc_nodes[:max_examples]:
            print(f"- {node}")
        if i < len(data['nodes_in_retrieved_doc']):
            print()
