import enum
import functools
import re
from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from tenacity import Retrying, stop_after_attempt, wait_fixed
import instructor
import openai
import enum
from typing import List
import numpy as np
from textwrap import fill

from src.agent.tools.semantic_search import SemanticSearch
from src.agent.tools.splade_search import SparseEmbeddingsSplade


class SubQuestionList(BaseModel):
    """List of sub-questions related to a high level question"""

    questions: List[str] = Field(
        description="Sub-questions related to the main question."
    )


def generate_subquestions(query: str, n: str = "2 to 3") -> SubQuestionList:
    client = instructor.patch(openai.OpenAI())

    template = f"""
    Your task is to decompose an original user question into {n} distinct atomic sub-questions, \
    such that when they are resolved, the high level question will be answered.
    \n\n# Original question:\n{query}
    """

    return client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_model=SubQuestionList,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "You are a world class query understanding AI.",
            },
            {
                "role": "user",
                "content": template,
            },
        ],
    )


# df = pd.read_parquet("./data/splade_embeds.parquet")
# # Convert column names to snake_case for compatibility with LanceDB
# original_columns = df.columns
# snake_case_columns = {
#     col: re.sub(r"(?<!^)(?=[A-Z])", "_", col).lower() for col in original_columns
# }
# df.rename(columns=snake_case_columns, inplace=True)

# splade_search = SparseEmbeddingsSplade(
#     df=df,
#     text_column="body",
#     splade_column="splade_embeddings",
# )

# semantic_search = SemanticSearch(
#     df=df,
#     embedding_col_name="embeddings",
# )


class SearchType(enum.Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"


import enum
import instructor

from typing import List
import numpy as np
from openai import OpenAI
from pydantic import Field, BaseModel
from tenacity import Retrying, stop_after_attempt, wait_fixed

from src.agent.tools.semantic_search import SemanticSearch
from src.agent.tools.splade_search import SparseEmbeddingsSplade
from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

client = instructor.patch(OpenAI())


class SubQuestionList(BaseModel):
    """List of sub-questions related to a high level question"""

    questions: List[str] = Field(
        description="Sub-questions related to the main question."
    )

@functools.cache
def generate_subquestions(query: str, n: str = "2 to 4") -> SubQuestionList:
    client = instructor.patch(openai.OpenAI())

    template = f"""
    Your task is to decompose an original user question into {n} distinct atomic sub-questions, \
    such that when they are resolved, the high level question will be answered.
    \n\n# Original question:\n{query}
    """

    return client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_model=SubQuestionList,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "You are a world class query understanding AI.",
            },
            {
                "role": "user",
                "content": template,
            },
        ],
    )


class SubQuestion(BaseModel):
    """
    Class representing a single sub-question research topic related to a higher level question.

    """

    chain_of_thought: str = Field(
        description="Reasoning behind the sub-question with respect to its role in answering the primary question.", 
        exclude=True
    )
    sub_question_topic: str = Field(
        ...,
        description="A concise topic title for the sub-question.",
    )
    sub_question_query: str = Field(
        ...,
        description="A distinct and context rich sub-question that can be directly entered into a vector database, which does similarity search for retrieving documents, in order to help answer a higher level user question.",
    )
    sub_question_keywords: List[str] = Field(
        default_factory=list,
        description="Extracted keywords suitable for exact-match text search that support the sub-question topic. Can be technical domain-specific items such as form numbers, formal references, codified rules, citations, etc.",
    )

    def execute(
        self, semantic_search: SemanticSearch, splade_search: SparseEmbeddingsSplade
    ) -> pd.DataFrame:
        """
        Executes both vector and keyword searches based on the sub-question query and keywords.

        Returns:
            pd.DataFrame: A DataFrame containing the results from both searches.
        """
        wrapped_thought = fill(self.chain_of_thought, width=100)
        logger.info(
            f"\n\nThought: {wrapped_thought}\nSearch topic: {self.sub_question_topic}"
        )
        logger.info(f"Running vector (OpenAI) search: {self.sub_question_query}")
        vector_results = semantic_search.query_similar_documents(
            self.sub_question_query, top_n=10
        )

        # Initialize an empty DataFrame for SPLADE results
        splade_results = pd.DataFrame()

        # Check if sub_question_keywords is not None and has valid keywords
        if self.sub_question_keywords and any(self.sub_question_keywords):
            keywords = ", ".join(self.sub_question_keywords)
            logger.info(f"Running keyword (SPLADE) search: {keywords}")
            splade_results = splade_search.query_similar_documents(keywords, top_n=10)

        # Concatenate results only if splade_results is not empty
        if not splade_results.empty:
            results_df = pd.concat([vector_results, splade_results], ignore_index=True)
        else:
            results_df = vector_results

        logger.info(
            f"Returning {len(vector_results)} records from vector search and {len(splade_results)} from keywords"
        )
        logger.info("-" * 75)
        return vector_results, splade_results


class MultiSearch(BaseModel):
    """
    Class representing multiple sub-questions that are mutually exclusive and
    collectively exhaustive in relation to an input user question.

    Args:
        searches (List[SubQuestion]): The list of sub-questions.
    """

    searches: List[SubQuestion] = Field(
        ...,
        description="List of sub-questions and searches to perform.",
    )

    def execute(
        self, semantic_search: SemanticSearch, splade_search: SparseEmbeddingsSplade
    ) -> pd.DataFrame:
        """Helper method to run and combine vector &/or splade searches."""
        vector_results = []
        splade_results = []
        for search in self.searches:
            vector_res, splade_res = search.execute(semantic_search, splade_search)
            vector_results.append(vector_res)
            splade_results.append(splade_res)
        return vector_results, splade_results


def segment_search_query(user_query: str, n: str = "2 to 4") -> MultiSearch:
    """
    Convert a string into multiple search queries.

    Args:
        data (str): The string to convert into search queries.
        n (str): A number range in string format to 'ask' the llm to generate.

    Returns:
        MultiSearch: An object representing the multiple search queries.
    """

    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        temperature=0.1,
        response_model=MultiSearch,
        max_retries=Retrying(stop=stop_after_attempt(5), wait=wait_fixed(1)),
        messages=[
            {
                "role": "system",
                "content": "You are a legal AI assistant that specializes in breaking down nuanced legal questions into simple, manageable sub-questions. When presented with a user legal question or research topic, your role is to generate a list of sub-questions that, when answered, will comprehensively address the original query.",
            },
            {
                "role": "user",
                "content": f"Please generate {n} sub-questions to help research a user question.",
            },
            {"role": "user", "content": f"Here is the user question: {user_query}"},
        ],
        # max_tokens=2000,
    )
    return completion
