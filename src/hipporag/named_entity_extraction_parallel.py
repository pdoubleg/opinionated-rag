import sys
from functools import partial

sys.path.append('.')

from src.hipporag.processing import extract_json_dict

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from tqdm import tqdm

from src.hipporag.langchain_util import init_langchain_model

query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

query_prompt_template = """
Question: {}

"""


def named_entity_recognition(client, text: str):
    
    query_ner_prompts = ChatPromptTemplate.from_messages([SystemMessage("You're a very effective entity extraction system."),
                                                          HumanMessage(query_prompt_one_shot_input),
                                                          AIMessage(query_prompt_one_shot_output),
                                                          HumanMessage(query_prompt_template.format(text))])
    query_ner_messages = query_ner_prompts.format_prompt()

    chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=300, stop=['\n\n'], response_format={"type": "json_object"})
    response_content = chat_completion.content
    total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']

    return response_content, total_tokens


def run_ner_on_texts(client, texts):
    ner_output = []
    total_cost = 0

    for text in tqdm(texts):
        ner, cost = named_entity_recognition(client, text)
        ner_output.append(ner)
        total_cost += cost

    return ner_output, total_cost
