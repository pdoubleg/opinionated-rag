import json
from glob import glob
import os

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from src.hipporag.langchain_util import init_langchain_model
from src.hipporag.openie_extraction_instructions import (
    ner_prompts,
    openie_post_ner_prompts,
)
from src.hipporag.config_manager import ConfigManager


def named_entity_recognition(passage: str, config: ConfigManager = ConfigManager()):
    client = init_langchain_model(config.llm_model)
    ner_messages = ner_prompts.format_prompt(user_input=passage)

    not_done = True

    total_tokens = 0
    response_content = "{}"

    while not_done:
        try:
            chat_completion = client.invoke(
                ner_messages.to_messages(),
                temperature=config.ner_temperature,
                response_format={"type": "json_object"},
            )
            response_content = chat_completion.content
            response_content = eval(response_content)
            total_tokens += chat_completion.response_metadata["token_usage"][
                "total_tokens"
            ]
            if "named_entities" not in response_content:
                response_content = []
            else:
                response_content = response_content["named_entities"]

            not_done = False
        except Exception as e:
            print("Passage NER exception")
            print(e)

    return response_content, total_tokens


def openie_post_ner_extract(
    passage: str, entities: list, model_name: str
) -> tuple[str, int]:
    """
    Extract OpenIE triples from a passage using named entities.

    Args:
        passage (str): The input passage.
        entities (list): List of named entities.
        model_name (str): Name of the model to use.

    Returns:
        tuple[str, int]: A tuple containing the response content and total tokens used.
    """
    client = init_langchain_model(model_name)
    named_entity_json = {"named_entities": entities}
    openie_messages = openie_post_ner_prompts.format_prompt(
        passage=passage, named_entity_json=json.dumps(named_entity_json)
    )
    try:
        chat_completion = client.invoke(
            openie_messages.to_messages(),
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.content
        total_tokens = chat_completion.response_metadata["token_usage"]["total_tokens"]
    except Exception as e:
        print("OpenIE exception", e)
        return "", 0

    return response_content, total_tokens


def extract_openie_from_triples(
    args, existing_json, ents_by_doc, auxiliary_file_exists, model_name
):
    """
    Extract OpenIE triples from the given passages.

    Args:
        args (list): A list of tuples containing (index, passage_dict) pairs.
        existing_json (list): List of existing JSON data.
        ents_by_doc (list): List of entities by document.
        auxiliary_file_exists (bool): Whether an auxiliary file exists.
        model_name (str): Name of the model to use.

    Returns:
        tuple: A tuple containing new_json, all_entities, and chatgpt_total_tokens.
    """
    new_json = []
    all_entities = []
    chatgpt_total_tokens = 0

    for i, r in args:
        passage = r["passage"]

        if i < len(existing_json):
            new_json.append(existing_json[i])
        else:
            if auxiliary_file_exists:
                doc_entities = ents_by_doc[i]
            else:
                doc_entities, total_ner_tokens = named_entity_recognition(passage)
                doc_entities = list(np.unique(doc_entities))
                chatgpt_total_tokens += total_ner_tokens
                ents_by_doc.append(doc_entities)
            all_entities.extend(doc_entities)
            triples, total_tokens = openie_post_ner_extract(
                passage, doc_entities, model_name
            )
            chatgpt_total_tokens += total_tokens

            r["extracted_entities"] = doc_entities

            try:
                r["extracted_triples"] = eval(triples)["triples"]
            except:
                print("ERROR")
                print(triples)
                r["extracted_triples"] = []

            new_json.append(r)

    return new_json, all_entities, chatgpt_total_tokens


def load_existing_data(dataset: str, arg_str: str, num_passages: int) -> tuple:
    """
    Load existing data from previous extractions.

    Args:
        dataset (str): The name of the dataset.
        arg_str (str): The argument string used for file naming.
        num_passages (int): The number of passages to process.

    Returns:
        tuple: A tuple containing existing_json, ents_by_doc, and already_done flag.
    """
    try:
        arg_str_regex = arg_str.replace(str(num_passages), "*")
        prev_num_passages = 0
        new_json_temp = None

        for file in glob(
            f"./hipporag_data/output/openie{dataset}_results_{arg_str_regex}.json"
        ):
            possible_json = json.load(open(file, "r"))
            if prev_num_passages < len(possible_json["docs"]):
                prev_num_passages = len(possible_json["docs"])
                new_json_temp = possible_json

        if new_json_temp is None:
            return [], [], False

        existing_json = new_json_temp["docs"]
        if "ents_by_doc" in new_json_temp:
            ents_by_doc = new_json_temp["ents_by_doc"]
        elif "non_dedup_ents_by_doc" in new_json_temp:
            ents_by_doc = new_json_temp["non_dedup_ents_by_doc"]
        else:
            ents_by_doc = []

        already_done = num_passages < len(existing_json)
        return existing_json, ents_by_doc, already_done

    except Exception as e:
        print(f"Error loading existing data: {e}")
        return [], [], False


def check_auxiliary_file(
    dataset: str, flags_present: list, model_name: str, num_passages: int
) -> tuple:
    """
    Check for and load auxiliary files to reduce API consumption.

    Args:
        dataset (str): The name of the dataset.
        flags_present (list): List of flags present in the extraction.
        model_name (str): The name of the model used.
        num_passages (int): The number of passages to process.

    Returns:
        tuple: A tuple containing auxiliary_file_exists flag and ents_by_doc.
    """
    aux_file_str = "_".join(flags_present) + f"*_{model_name}_{num_passages}"
    aux_file_str = aux_file_str.replace(f"{num_passages}", "*")
    auxiliary_files = glob(
        f"./hipporag_data/output/openie{dataset}_results_{aux_file_str}.json"
    )

    for auxiliary_file in auxiliary_files:
        try:
            aux_info_json = json.load(open(auxiliary_file, "r"))
            if len(aux_info_json["docs"]) >= num_passages:
                ents_by_doc = aux_info_json["ents_by_doc"]
                print(f"Using Auxiliary File: {auxiliary_file}")
                return True, ents_by_doc
        except Exception as e:
            print(f"Error processing auxiliary file {auxiliary_file}: {e}")

    return False, []


def run_openie_extraction(
    config: ConfigManager,
    corpus,
) -> str:
    """
    Run OpenIE extraction on the specified dataset.

    """
    model_name = config.llm_model
    # TODO: make pydantic model for corpus/chunks
    retrieval_corpus = corpus
    for document in retrieval_corpus:
        document["passage"] = document[config.doc_column]

    dataset = "_" + str(config.dataset_name)

    # Process num_passages
    if config.num_passages == "all":
        num_passages = len(retrieval_corpus)
    else:
        try:
            num_passages = int(num_passages)
        except ValueError:
            raise AssertionError("Set 'num_passages' to an integer or 'all'")

    # Set up extraction parameters
    flag_names = ["ner"]
    flags_present = [flag_names[i] for i, flag in enumerate([config.run_ner]) if flag]
    if len(flags_present) > 0:
        arg_str = (
            "_".join(flags_present)
            + "_"
            + model_name.replace("/", "_")
            + f"_{num_passages}"
        )
    else:
        arg_str = model_name.replace("/", "_") + f"_{num_passages}"

    print(arg_str)

    # Initialize language model
    client = init_langchain_model(config.llm_model)

    # Load existing data or initialize new
    existing_json, ents_by_doc, already_done = load_existing_data(
        dataset, arg_str, num_passages
    )

    if not already_done:
        auxiliary_file_exists, aux_ents_by_doc = check_auxiliary_file(
            dataset, flags_present, config.llm_model, num_passages
        )
        if auxiliary_file_exists:
            ents_by_doc = aux_ents_by_doc

    try:
        # Get incomplete extraction output with same settings
        arg_str_regex = arg_str.replace(str(num_passages), "*")

        prev_num_passages = 0
        new_json_temp = None

        for file in glob(
            os.path.join(
                config.output_directory,
                "openie{}_results_{}.json".format(dataset, arg_str_regex),
            )
        ):
            possible_json = json.load(open(file, "r"))
            if prev_num_passages < len(possible_json["docs"]):
                prev_num_passages = len(possible_json["docs"])
                new_json_temp = possible_json

        existing_json = new_json_temp["docs"]
        if "ents_by_doc" in new_json_temp:
            ents_by_doc = new_json_temp["ents_by_doc"]
        elif "non_dedup_ents_by_doc" in new_json_temp:
            ents_by_doc = new_json_temp["non_dedup_ents_by_doc"]
        else:
            ents_by_doc = []

        if num_passages < len(existing_json):
            already_done = True
    except:
        existing_json = []
        ents_by_doc = []

    # Load files to reduce API consumption
    aux_file_str = "_".join(flags_present) + "*_" + model_name + f"_{num_passages}"
    aux_file_str = aux_file_str.replace("{}".format(num_passages), "*")
    auxiliary_files = glob(
        os.path.join(
            config.output_directory,
            "openie{}_results_{}.json".format(dataset, aux_file_str),
        )
    )

    auxiliary_file_exists = False

    if len(auxiliary_files) > 0:
        for auxiliary_file in auxiliary_files:
            aux_info_json = json.load(open(auxiliary_file, "r"))
            if len(aux_info_json["docs"]) >= num_passages:
                ents_by_doc = aux_info_json["ents_by_doc"]
                auxiliary_file_exists = True
                print("Using Auxiliary File: {}".format(auxiliary_file))
                break

    extracted_triples_subset = retrieval_corpus[:num_passages]

    num_processes = config.num_processes

    splits = np.array_split(range(len(extracted_triples_subset)), num_processes)

    args = []
    for split in splits:
        args.append(
            (
                [(i, extracted_triples_subset[i]) for i in split],
                existing_json,
                ents_by_doc,
                auxiliary_file_exists,
                model_name,
            )
        )
    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            outputs = pool.starmap(extract_openie_from_triples, args)
    else:
        outputs = [extract_openie_from_triples(*arg) for arg in args]

    new_json = []
    all_entities = []
    lm_total_tokens = 0

    for output in outputs:
        new_json.extend(output[0])
        all_entities.extend(output[1])
        lm_total_tokens += output[2]

    # Save results
    if not already_done:
        avg_ent_chars = np.mean([len(e) for e in all_entities])
        avg_ent_words = np.mean([len(e.split()) for e in all_entities])

        approx_total_tokens = (len(retrieval_corpus) / num_passages) * lm_total_tokens

        extra_info_json = {
            "docs": new_json,
            "ents_by_doc": ents_by_doc,
            "avg_ent_chars": round(avg_ent_chars, 2),
            "avg_ent_words": round(avg_ent_words, 2),
            "num_tokens": lm_total_tokens,
            "approx_total_tokens": approx_total_tokens,
        }
        output_path = os.path.join(
            config.output_directory, f"openie{dataset}_results_{arg_str}.json"
        )
        json.dump(extra_info_json, open(output_path, "w"))
        print("OpenIE saved to", output_path)

    return output_path
