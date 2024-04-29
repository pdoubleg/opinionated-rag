import asyncio
import re
import warnings
from annotated_types import Gt, Lt
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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
    root_validator,
    validator,
)

from typing import Optional, Type, Generic, TypeVar, List, Dict, Any
from typing_extensions import Annotated, Self
from datetime import datetime, timedelta


def convert_timestamp_to_datetime(timestamp: str) -> str:
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d")


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


def clean_string(string):
    # Remove spaces and special characters using regex
    string = re.sub("[^A-Za-z0-9]+", "", string)
    # Convert the string to lowercase
    string = string.lower()
    return string


def generate_citation_strings(
    citation_numbers: List[str],
    df: pd.DataFrame,
    name_column: str,
    location_column: str,
    date_column: str,
    url_column: str,
) -> List[str]:
    result = []
    for citation in citation_numbers:
        i = int(citation) - 1  # convert string to int and adjust for 0-indexing
        title = df.iloc[i][name_column]
        claim_number = df.iloc[i]["id"]
        claim_number = clean_string(str(claim_number))
        link = df.iloc[i][url_column]
        claim_number_formatted = f"[{claim_number}]({link})"
        venue = str(df.iloc[i][location_column])
        date = str(df.iloc[i][date_column])
        result.append(
            f"**{[i+1]}** *{title}* - {venue}, {date}, Claim Number: {claim_number_formatted}\n\n"
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
                "content": "You are a world-class document understanding AI. Your task is to generate highly detailed and information dense summaries of legal text such that key information is retained and emphasized to support downstream information retrieval. Include verbatim substring quotes and specific case law citations as often as possible to support the summaries.",
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


async def aget_fact_patterns_df(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Asynchronously generates fact pattern summaries for a DataFrame using OpenAI's chat completion API.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data to summarize.
        text_column (str): The name of the column in the DataFrame containing the text to summarize.

    Returns:
        pd.DataFrame: The original DataFrame merged with the generated fact pattern summaries.
    """
    df["temp_id"] = [str(i) for i in range(len(df))]
    tasks = [
        aget_llm_fact_pattern_summary(row[text_column], str(row["temp_id"]))
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
    merged_df.drop(columns=["temp_id"], inplace=True)
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
    instructions: str = """Instructions: Using only the provided search results that are relevant, and starting with the most relevant, write a detailed comparative analysis for a new query. If there are no relevant cases say so, and use one example from the search results to illustrate the uniqueness of the new query. ALWAYS cite search results using [[number](URL)] notation after the reference.\n\nNew Query:""",
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

    @model_validator(mode="after")
    def extract_citations(self) -> Self:
        cites = extract_citation_numbers_in_brackets(self.research_report)
        self.citations = cites if cites is not None else "No Citations Found"
        if self.citations == "No Citations Found":
            raise ValueError(
                f"Output should contain at least on citation string from the provided context, e.g., `['2']`"
            )
        return self

    @property
    def generate_citation_strings(
        self,
        df: pd.DataFrame,
        id_column: str,
        name_column: str,
        location_column: str,
        date_column: str,
        url_column: str,
    ) -> List[str]:
        result = []
        for citation in self.citations:
            i = int(citation) - 1  # convert string to int and adjust for 0-indexing
            title = df.iloc[i][name_column]
            id_number = df.iloc[i][id_column]
            id_number = clean_string(str(id_number))
            link = df.iloc[i][url_column]
            id_number_formatted = f"[{id_number}]({link})"
            venue = str(df.iloc[i][location_column])
            date = str(df.iloc[i][date_column])
            result.append(
                f"**{[i+1]}** *{title}* - {venue}, {date}, Claim Number: {id_number_formatted}\n\n"
            )
        return result


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
                "content": "You are helpful legal research assistant. Analyze the current legal question, and compare it to the search results of past cases. Using only the provided context, offer insights into how the relevant past cases can help address the new question or issue. Do not answer the question or provide opinions, only draw helpful comparisons to help guide research efforts. Remember to ALWAYS use markdown links when citing the context, for example [[number](URL)].",
            },
            {
                "role": "user",
                "content": f"Remember to use ALWAYS include markdown citation links when referencing the context, for example [[number](URL)]. Search Results:\n\n{formatted_input}",
            },
        ],
    )


class ContextSummary(BaseModel):
    """A summary analysis of context based on user instructions."""

    summary_analysis: str = Field(
        ...,
        description="An information dense factual summary with emphasis on details and nuanced considerations.",
    )
    relevance_score: Annotated[int, Gt(0), Lt(11)] = Field(
        ...,
        description="An integer score from 1-10 indicating relevance to question.",
    )


summary_prompt = (
    "Summarize the excerpt below to help determine if it is a relevant reference for a question.\n\n"
    "Excerpt:----\n\n{context}\n\n----\n\n"
    "Question: {question}\n\n"
    "Do not answer the question, instead summarize to give evidence to help "
    "answer the question. Stay detailed; report specific citations, laws, or "
    'direct quotes (marked with quotation marks). Reply "Not applicable" if the '
    "excerpt is irrelevant. At the end of your response, provide an integer score "
    "from 1-10 indicating relevance to question. Do not explain your score."
    "\n\nRelevant Information Summary:"
)


def evaluate_context(context: str, question: str, model_name: str) -> ContextSummary:
    client = instructor.patch(openai.OpenAI())
    return client.chat.completions.create(
        model=model_name,
        response_model=ContextSummary,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "You are helpful legal research AI.",
            },
            {
                "role": "user",
                "content": summary_prompt.format(context=context, question=question),
            },
        ],
    )


async def aget_context_evaluation(
    context: str, question: str, id_value: str, model_name: str
) -> Dict:
    """Asynchronously evaluates the relevance of a context to a question using OpenAI's chat completion API.

    Args:
        context (str): The context to evaluate.
        question (str): The question to evaluate the context against.
        id_value (str): A temporary identifier for the context.
        model_name (str): The name of the model to use for evaluation.

    Returns:
        Dict: The context evaluation result along with the temporary identifier.
    """
    client = instructor.patch(openai.AsyncOpenAI())
    response = await client.chat.completions.create(
        model=model_name,
        response_model=ContextSummary,
        max_retries=AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful legal research AI.",
            },
            {
                "role": "user",
                "content": summary_prompt.format(context=context, question=question),
            },
        ],
    )
    result = response.model_dump()
    result["temp_id"] = id_value
    return result


async def aget_context_evaluations_df(
    df: pd.DataFrame, question: str, text_column: str, model_name: str
) -> pd.DataFrame:
    """Asynchronously evaluates the relevance of contexts in a DataFrame to a question using OpenAI's chat completion API.

    Args:
        df (pd.DataFrame): The DataFrame containing the context data to evaluate.
        question (str): The question to evaluate the contexts against.
        text_column (str): The name of the column in the DataFrame containing the contexts to evaluate.
        model_name (str): The name of the model to use for evaluation.

    Returns:
        pd.DataFrame: The original DataFrame merged with the generated context evaluations.
    """
    df["temp_id"] = [str(i) for i in range(len(df))]
    tasks = [
        aget_context_evaluation(
            row[text_column], question, str(row["temp_id"]), model_name
        )
        for _, row in df.iterrows()
    ]
    results = []
    # Wrap asyncio.as_completed with tqdm for progress tracking
    for future in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Processing context evaluations",
    ):
        result = await future
        results.append(result)
    results_df = pd.DataFrame(results)
    merged_df = df.merge(results_df, left_on="temp_id", right_on="temp_id")
    merged_df.drop(columns=["temp_id"], inplace=True)
    merged_df.sort_values(by='relevance_score', ascending=False, inplace=True)
    return merged_df
