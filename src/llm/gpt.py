import os
import sys
import logging

from openai import OpenAI
import instructor
from pydantic import BaseModel, Field

client = instructor.patch(OpenAI())
import tiktoken

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    after_log,
)  # for exponential backoff

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_PRICING = {
    "gpt-3.5-turbo-0125": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-16k-0613": {"prompt": 0.003, "completion": 0.004},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
    "embedding": {"hugging_face": 0, "text-embedding-ada-002": 0.0001},
}


OPENAI_MODEL_CONTEXT_LENGTH = {
    "gpt-3.5-turbo-0125": 4097,
    "gpt-3.5-turbo-16k-0613": 16385,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
}


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    after=after_log(logger, logging.INFO),
)
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def llm_call_cost(response) -> float:
    """
    Returns the total cost of one or more LLM calls in dollars.
    
    Args:
        response: A single response object or a list of response objects from the LLM call.
    
    Returns:
        The total cost of the LLM call(s) in dollars.
    """
    if isinstance(response, list):
        total_cost = 0
        for resp in response:
            total_cost += _calculate_response_cost(resp)
        return total_cost
    else:
        return _calculate_response_cost(response)

def _calculate_response_cost(response) -> float:
    """
    Calculates the cost of a single LLM call in dollars.
    
    Args:
        response: A single response object from the LLM call.
    
    Returns:
        The cost of the LLM call in dollars.
    """
    model = response._raw_response.model
    usage = response._raw_response.usage
    prompt_cost = OPENAI_PRICING[model]["prompt"]
    completion_cost = OPENAI_PRICING[model]["completion"]
    prompt_token_cost = (usage.prompt_tokens * prompt_cost) / 1000
    completion_token_cost = (usage.completion_tokens * completion_cost) / 1000
    return prompt_token_cost + completion_token_cost


def llm_call(
    model,
    response_model,
    system_prompt="You are an AI assistant that answers user questions using the context provided.",
    user_prompt="Please help me answer the following question:",
    few_shot_examples=None,
):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot_examples is not None:
        messages.extend(few_shot_examples)
    if user_prompt is not None:
        messages.append({"role": "user", "content": user_prompt})

    response = completion_with_backoff(
        model=model,
        response_model=response_model,
        temperature=0,
        messages=messages,
    )

    # print cost of call
    # call_cost = llm_call_cost(response)
    # logger.info(f"🤑 LLM call cost: ${call_cost:.4f}")
    return response


def get_num_tokens_simple(model, prompt):
    """Estimate the number of tokens in the prompt using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


# class HelloWorld(BaseModel):
#     """A sarcastic hello world"""

#     hello: str = Field(
#         ...,
#         description="A sarcastic but wacky hello 'word based' phrase based on user provided term.",
#     )

#     def say_hello(self):
#         return self.hello


# hi, cost = llm_call(
#     model="gpt-3.5-turbo-16k",
#     user_prompt="A pirate office worker",
#     response_model=HelloWorld,
# )
# logger.info(hi.say_hello())
# logger.info(f"That joke cost us ${cost:.4f}")
