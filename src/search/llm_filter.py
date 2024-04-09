
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain.schema.messages import AIMessage
from langchain.schema.messages import BaseMessage
from langchain.schema.messages import HumanMessage
from langchain.schema.messages import SystemMessage
from llama_index_client import TextNode
from pydantic import BaseModel

from src.search.threadpool import run_functions_tuples_in_parallel

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


USEFUL_PAT = "Yes useful"
NONUSEFUL_PAT = "Not useful"

CHUNK_FILTER_PROMPT = f"""
Determine if the reference section is USEFUL for answering the user query.
It is good enough for the section to be related or similar to the query; \
it should be relevant information that is USEFUL for comparing to the query.
If the section contains ANY useful information, that is good enough, \
it does not need to fully answer the user query, but it \
should at least address a component to be USEFUL.

Reference Section:
```
{{chunk_text}}
```

User Query:
```
{{user_query}}
```

Respond with EXACTLY AND ONLY: "{USEFUL_PAT}" or "{NONUSEFUL_PAT}"
""".strip()


def dict_based_prompt_to_langchain_prompt(
    messages: list[dict[str, str]]
) -> list[BaseMessage]:
    prompt: list[BaseMessage] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not role:
            raise ValueError(f"Message missing `role`: {message}")
        if not content:
            raise ValueError(f"Message missing `content`: {message}")
        elif role == "user":
            prompt.append(HumanMessage(content=content))
        elif role == "system":
            prompt.append(SystemMessage(content=content))
        elif role == "assistant":
            prompt.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unknown role: {role}")
    return prompt


def llm_eval_chunk(query: str, chunk_content: str) -> bool:
    def _get_usefulness_messages() -> list[dict[str, str]]:
        messages = [
            {
                "role": "user",
                "content": CHUNK_FILTER_PROMPT.format(
                    chunk_text=chunk_content, user_query=query
                ),
            },
        ]

        return messages

    def _extract_usefulness(model_output: str) -> bool:
        """Default 'useful' if the LLM doesn't match pattern exactly.
        This is because it's better to trust the (re)ranking if LLM fails"""
        if model_output.content.strip().strip('"').lower() == NONUSEFUL_PAT.lower():
            return False
        return True

    llm = ChatOpenAI(model='gpt-3.5-turbo')

    messages = _get_usefulness_messages()
    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(messages)
    model_output = llm.invoke(filled_llm_prompt)

    return _extract_usefulness(model_output)


def llm_batch_eval_chunks(
    query: str, chunk_contents: list[str], use_threads: bool = True
) -> list[bool]:
    if use_threads:
        functions_with_args: list[tuple[Callable, tuple]] = [
            (llm_eval_chunk, (query, chunk_content)) for chunk_content in chunk_contents
        ]

        print(
            "Running LLM usefulness eval in parallel (following logging may be out of order)"
        )
        parallel_results = run_functions_tuples_in_parallel(
            functions_with_args, allow_failures=True
        )

        # In case of failure/timeout, don't throw out the chunk
        return [True if item is None else item for item in parallel_results]

    else:
        return [
            llm_eval_chunk(query, chunk_content) for chunk_content in chunk_contents
        ]
        

def filter_chunks(
    query: str,
    chunks_to_filter: list[BaseModel],
    max_llm_filter_chunks: int = 20,
) -> list[BaseModel]:
    """Filters chunks based on whether the LLM thought they were relevant to the query.

    Args:
        query (str): The query to filter chunks against.
        chunks_to_filter (list[BaseModel]): A list of BaseModel objects to filter.
        max_llm_filter_chunks (int, optional): The maximum number of chunks to consider. Defaults to 20.

    Returns:
        list[BaseModel]: A list of BaseModel objects that were marked as relevant.
    """
    chunks_to_filter = chunks_to_filter[: max_llm_filter_chunks]
    llm_chunk_selection = llm_batch_eval_chunks(
        query=query,
        chunk_contents=[chunk.text for chunk in chunks_to_filter],
    )
    return [
        chunk
        for ind, chunk in enumerate(chunks_to_filter)
        if llm_chunk_selection[ind]
    ]

