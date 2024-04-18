import asyncio
import re
import warnings
import instructor
import openai
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm
from tenacity import AsyncRetrying, Retrying, stop_after_attempt, wait_fixed

from src.utils.gen_utils import count_tokens


pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel, ConfigDict, Field, model_validator, root_validator, validator

from typing import Optional, Type, Generic, TypeVar, List, Dict, Any
from typing_extensions import Self
from datetime import datetime, timedelta


def clean_string(string):
    # Remove spaces and special characters using regex
    string = re.sub("[^A-Za-z0-9]+", "", string)
    # Convert the string to lowercase
    string = string.lower()
    return string


def extract_citation_numbers_in_brackets(text: str) -> List[str]:
    """Extracts a list of citation integer-strings, e.g. ['1', '4', '5']"""
    matches = re.finditer(r"\[(\d+)\]", text)
    citations = []
    seen = set()
    for match in matches:
        citation = match.group(1)
        if citation not in seen:
            citations.append(citation)
            seen.add(citation)
    return citations


def generate_citation_strings(
    citation_numbers: List[str],
    df: pd.DataFrame,
    opinion_claim_numbers: Optional[List[str]] = None,
) -> List[str]:
    result = []
    for citation in citation_numbers:
        i = int(citation) - 1  # convert string to int and adjust for 0-indexing
        title = df.iloc[i]["llm_title"]
        claim_number = df.iloc[i]["id"]
        # Add footnote style links. These go to the opinion text which has a back link to the citation
        footnote = f'<sup class="footnote-ref" id="fnref-{i+1}"><a href="#fn-{i+1}">{i+1}</a></sup>'
        # Don't link to non-existent opinion text
        if claim_number not in opinion_claim_numbers:
            footnote = ""
        # Note: Current code links all cases to USRM PL Navigator Claims system.
        # TODO: Update links here and in input context string

        claim_number_clean = clean_string(str(claim_number))
        link = f"={claim_number_clean}"
        claim_number_formatted = f"[{claim_number}]({link})"

        venue = str(df.iloc[i]["State"])
        date = "2022 Jan"
        result.append(
            f"**{[i+1]}** {footnote} *{title}* - {venue}, {date}, Claim Number: {claim_number_formatted}\n\n"
        )
    return result


def get_claim_numbers_from_citations(
    citation_numbers: List[str], df: pd.DataFrame
) -> pd.DataFrame:
    result = []
    for citation in citation_numbers:
        i = int(citation) - 1  # convert string to int and adjust for 0-indexing
        temp_df = (
            df.iloc[i][["id", "llm_title", "body", "full_link", "State", "topic_title"]]
            .to_frame()
            .T
        )
        result.append(temp_df)
    result_df = pd.concat(result, axis=0, ignore_index=True)
    result_df["citation"] = [str(c) for c in citation_numbers]
    result_df["footnote_content"] = "[^" + result_df["citation"] + "]: "
    result_df["Citation"] = pd.Categorical(
        result_df["citation"], ordered=True, categories=citation_numbers
    )
    return result_df


class FactSummary(BaseModel):
    """A detailed factual summarization."""

    summary: str = Field(
        ...,
        description="An information dense factual summary with emphasis on details without any mention of specific entities.",
    )


def get_llm_fact_pattern_summary(query: str) -> FactSummary:
    """Get a fact pattern summary from the LLM.

    Args:
        query (str): The user's query to summarize.

    Returns:
        FactSummary: A detailed factual summarization of the query.
    """
    client = instructor.patch(openai.OpenAI())
    return client.chat.completions.create(
        model="gpt-4-turbo",
        response_model=FactSummary,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "You are a world class query understanding AI. Your task is to generate information dense summaries of user queries such that key information is retained and emphasized to support downstream information retrieval.",
            },
            {
                "role": "user",
                "content": f"\n\n# Original query:\n{query}",
            },
        ],
    )


async def aget_llm_fact_pattern_summary(query: str, id_value: str) -> Dict:
    """Asynchronously generates an information dense summary of a user query using OpenAI's chat completion API.

    Args:
        query (str): The user's query to summarize.
        id_value (str): A temporary identifier for the query.

    Returns:
        Dict: The summarized query result along with the temporary identifier.
    """
    client = instructor.patch(openai.AsyncOpenAI())
    response = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_model=FactSummary,
        max_retries=AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "You are a world-class query understanding AI. Your task is to generate information dense summaries of user queries such that key information is retained and emphasized to support downstream information retrieval.",
            },
            {
                "role": "user",
                "content": f"\n\n# Original query:\n{query}",
            },
        ],
    )
    result = response.model_dump()
    result["temp_id"] = id_value
    return result


async def aget_fact_patterns_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Asynchronously generates fact pattern summaries for a DataFrame using OpenAI's chat completion API.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data to summarize.
        text_col (str): The name of the column in the DataFrame containing the text to summarize.

    Returns:
        pd.DataFrame: The original DataFrame merged with the generated fact pattern summaries.
    """
    df["temp_id"] = [str(i) for i in range(len(df))]
    tasks = [
        aget_llm_fact_pattern_summary(row[text_col], str(row["temp_id"]))
        for _, row in df.iterrows()
    ]
    results = []
    # Wrap asyncio.as_completed with tqdm for progress tracking
    for future in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing summaries"
    ):
        result = await future
        results.append(result)
    results_df = pd.DataFrame(results)
    merged_df = df.merge(results_df, left_on="temp_id", right_on="temp_id")
    return merged_df


from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


def parallel_get_llm_fact_pattern_summaries(
    queries: List[str], max_workers: int = 5
) -> List[FactSummary]:
    """
    Executes get_llm_fact_pattern_summary in parallel for a list of queries.

    Args:
        queries (List[str]): A list of queries to process.
        max_workers (int): Maximum number of threads to use.

    Returns:
        List[FactSummary]: A list of FactSummary objects corresponding to each query.
    """
    summaries = [None] * len(queries)  # Pre-allocate list for results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each query
        future_to_index = {
            executor.submit(get_llm_fact_pattern_summary, query): i
            for i, query in enumerate(queries)
        }

        # As each future completes, store the result in the corresponding position
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                summaries[index] = future.result()
            except Exception as e:
                print(f"Query at index {index} failed: {e}")
                summaries[index] = str(e)  # Or handle the error as appropriate

    return summaries


# Create context for LLM prompt
def create_context(
    df: pd.DataFrame,
    query: str,
    text_column: str = "body",
    url_column: str = "full_link",
    context_token_limit: int = 3000,
) -> str:
    """Creates a context string from a DataFrame based on a query.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        query (str): The query string.
        text_column (str, optional): The name of the column containing the text data. Defaults to 'body'.
        url_column (str, optional): The name of the column containing the URL data. Defaults to 'full_link'.
        context_token_limit (int, optional): The maximum number of tokens allowed in the context. Defaults to 3000.

    Returns:
        str: The context string created from the DataFrame.
    """
    df.reset_index(drop=True, inplace=True)
    returns = []
    count = 1
    total_tokens = count_tokens(query)  # Start with tokens from the query
    # Add the text to the context until the context is too long
    for _, row in df.iterrows():
        text = "[" + str(count) + "] " + row[text_column] + "\nURL: " + row[url_column]
        text_tokens = count_tokens(text)
        if total_tokens + text_tokens > context_token_limit:
            break
        returns.append(text)
        total_tokens += text_tokens
        count += 1
    return "\n\n".join(returns)


def create_formatted_input(
    df: pd.DataFrame,
    query: str,
    text_column: str = "body",
    url_column: str = "full_link",
    context_token_limit: int = 3000,
    instructions: str = """Instructions: Using only the provided search results that are relevant, and starting with the most relevant, write a detailed comparative analysis for a new query. If there are no relevant cases say so, and use one example from the search results to illustrate the uniquness of the new query. ALWAYS cite search results using [[number](URL)] notation after the reference.\n\nNew Query:""",
) -> str:
    """Creates a formatted input string for the model based on a DataFrame and query.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        query (str): The query string.
        text_column (str, optional): The name of the column containing the text data. Defaults to 'body'.
        url_column (str, optional): The name of the column containing the URL data. Defaults to 'full_link'.
        context_token_limit (int, optional): The maximum number of tokens allowed in the context. Defaults to 3000.
        instructions (str, optional): The instructions to include in the formatted input. Defaults to a predefined string.

    Returns:
        str: The formatted input string.
    """
    context = create_context(df, query, text_column, url_column, context_token_limit)

    try:
        prompt = f"""{context}\n\n{instructions}\n{query}\n\nAnalysis:"""
        return prompt
    except Exception as e:
        print(e)
        return ""


class ResearchReport(BaseModel):
    """A comparative analysis legal research report focusing on relevancy."""

    research_report: str = Field(
        ...,
        description="A legal style research report comparing a new issue or question with similar past issues.",
    )
    citations: Optional[Any] = None
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )
    
    @model_validator(mode='after')
    def extract_citations(self) -> Self:
        cites = extract_citation_numbers_in_brackets(self.research_report)
        self.citations = cites if cites is not None else "No Citations Found"
        return self
           

def get_final_answer(formatted_input: str, model_name: str) -> ResearchReport:
    """Gets the final answer from the model based on the formatted input.

    Args:
        formatted_input (str): The formatted input string containing the search results and query.
        model_name (str): The name of the model to use for generating the answer.

    Returns:
        ResearchReport: The generated research report containing the comparative analysis.
    """
    client = instructor.patch(openai.OpenAI())
    return client.chat.completions.create(
        model=model_name,
        response_model=ResearchReport,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "You are helpful legal research assistant. Analyze the current legal question, and compare it to the search results of past cases. Using only the provided context, offer insights into how the researcher can reference the relevant past questions to address the new outstanding issue. Do not answer the question or provide opinions, only draw helpful comparisons to the relevant search results. Remember to use markdown links when citing the context, for example [[number](URL)].",
            },
            {"role": "user", "content": f"Remember to use inline markdown links when citing the context, for example [[number](URL)]. Search Results:\n\n{formatted_input}"},
        ],
    )
