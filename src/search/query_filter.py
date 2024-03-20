import os
import re
import instructor
import openai
import pandas as pd
import lancedb
from typing import Optional, List
from pydantic import BaseModel, Field
from tenacity import Retrying, stop_after_attempt, wait_fixed

from src.embedding_models.models import OpenAIEmbeddings
from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


class QueryFilterPlan(BaseModel):
    """A revised user query, possibly improved by filtering."""
    
    original_query: str = Field(
        ..., 
        description="The original user query."
    )
    filter: Optional[str] | None = Field(
        None, 
        description="An SQL-like filter inferred from the user query."
    )
    rephrased_query: str = Field(
        ...,
        description="A rephrased query based on the FILTER fields, or an empty string if no filter is needed.",
    )
    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the given DataFrame based on the (LLM) defined filter, using LanceDB for querying.

        Args:
            df (pd.DataFrame): The DataFrame to be filtered.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if not self.filter:
                logger.info(f"No filters have been set! Returning input DataFrame.")
                return df
        else:
            logger.info(f"Input DataFrame has {len(df):,} rows")
            logger.info(f"Applying filter(s): {self.filter}")

        uri = "../temp-lancedb/pd_table.lance"
        db = lancedb.connect(uri)
        try:
            table = db.create_table("temp_lance", data=df, mode="create")
        except:
            table = db.create_table("temp_lance", data=df, mode="overwrite")
            
        result_df = (
            table.search()
            .where(self.filter)
            .limit(None)
            .to_df()
        )
        db.drop_database()
        logger.info(f"Filtered DataFrame has {len(result_df):,} rows")
        return result_df


system_message = f"""
You will receive a QUERY, to be answered based on an EXTREMELY LARGE collection
of documents you DO NOT have access to, but your ASSISTANT does.
You only know that these documents contain text content and FILTERABLE fields in the SCHEMA below:  

{{doc_schema}}

Based on the QUERY and the above SCHEMA, your task is to determine a QUERY PLAN,
consisting of:
-  a FILTER (can be None) that would help the ASSISTANT to answer the query.
    Remember the FILTER can refer to ANY fields in the above SCHEMA. 
    To get good results, for STRING MATCHES, consider using LIKE instead of =, e.g.
    "CEO LIKE '%Jobs%'" instead of "CEO = 'Steve Jobs'"
- a possibly REPHRASED QUERY to be answerable given the FILTER.
    Keep in mind that the ASSISTANT does NOT know anything about the FILTER fields,
    so the REPHRASED QUERY should NOT mention ANY FILTER fields.

EXAMPLE:
------- 
Suppose there is a document-set about crime reports, where:
    CONTENT = crime report,
    Filterable SCHEMA consists of City, Year, num_deaths.

Then given this ORIGINAL QUERY: 

    What were the total deaths in shoplifting crimes in Los Angeles in 2023?

A POSSIBLE QUERY PLAN could be:

FILTER: "City LIKE '%Los Angeles%' AND Year = 2023"
REPHRASED QUERY: "shoplifting crime" --> this will be used to MATCH content of docs
        [NOTE: we dropped the FILTER fields City and Year since the 
        ASSISTANT does not know about them and only uses the query to 
        match the CONTENT of the docs.]

------------- END OF EXAMPLE ----------------

The FILTER must be a SQL-like condition, e.g. 
"year > 2000 AND genre = 'ScienceFiction'".
To ensure you get useful results, you should make your FILTER 
NOT TOO STRICT, e.g. look for approximate match using LIKE, etc.
E.g. "CEO LIKE '%Jobs%'" instead of "CEO = 'Steve Jobs'"
            
"""


def describe_dataframe(
    input_df: pd.DataFrame, filter_fields: List[str] = [], n_vals: int = 10
) -> str:
    """
    Generates a description of the columns in the dataframe,
    along with a listing of up to `n_vals` unique values for each column.
    Intended to be used to insert into an LLM context so it can generate
    appropriate queries or filters on the df.

    Args:
    df (pd.DataFrame): The dataframe to describe.
    filter_fields (list): A list of fields that can be used for filtering.
        When non-empty, the values-list will be restricted to these.
    n_vals (int): How many unique values to show for each column.

    Returns:
    str: A description of the dataframe.
    """
    # Convert column names to snake_case for compatibility with LanceDB
    df = input_df[filter_fields]
    
    description = []
    for column in df.columns.to_list():
        unique_values = df[column].dropna().unique()
        unique_count = len(unique_values)
        if unique_count > n_vals:
            displayed_values = unique_values[:n_vals]
            more_count = unique_count - n_vals
            values_desc = f" Values - {displayed_values}, ... {more_count} more"
        else:
            values_desc = f" Values - {unique_values}"
        col_type = "string" if df[column].dtype == "object" else df[column].dtype
        col_desc = f"* {column} ({col_type}); {values_desc}"
        description.append(col_desc)

    all_cols = "\n".join(description)

    return f"""
Name of each field, its type and unique values (up to {n_vals}):
{all_cols}
        """


def generate_query_plan(
    input_df: pd.DataFrame, query: str, filter_fields: List[str], n_vals: int = 20
) -> QueryFilterPlan:
    client = instructor.patch(openai.OpenAI())

    df = input_df[filter_fields]

    filter_string = describe_dataframe(
        input_df=df,
        filter_fields=filter_fields,
        n_vals=n_vals,
    )
    logger.info(f"Schema shown to LLM: {filter_string}")
    return client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_model=QueryFilterPlan,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": system_message.format(doc_schema=filter_string),
            },
            {
                "role": "user",
                "content": f"Here is the QUERY: {query}",
            },
        ],
    )
    

def auto_filter_vector_search(
    df: pd.DataFrame,
    query: str,
    text_column: str,
    embeddings_column: str,
    filter_fields: List[str],
    top_k: int = 20
) -> pd.DataFrame:
    
    query_plan = generate_query_plan(
        input_df=df,
        query=query,
        filter_fields=filter_fields,
        )
    
    if not query_plan.filter:
        logger.info(f"No filters were identified for query: {query}")
        search_query = query_plan.original_query
    else:
        logger.info(f"Applying filter(s): {query_plan.filter}")
        logger.info(f"Revised query: {query_plan.rephrased_query}")
        search_query = query_plan.rephrased_query
    
    df.rename(columns={embeddings_column: "vector"}, inplace=True)
    uri = "../temp-lancedb/pd_table.lance"
    db = lancedb.connect(uri)
    filter_fields.append(text_column)
    filter_fields.append("vector")
    try:
        table = db.create_table("temp_lance", data=df[filter_fields], mode="create")
    except:
        table = db.create_table("temp_lance", data=df[filter_fields], mode="overwrite")
        
    embeddings_model = OpenAIEmbeddings()
    embedder = embeddings_model.embedding_fn()
    query_vector = embedder(search_query)
    
    if query_plan.filter:
        result = (
            table.search(query_vector[0]) \
            .metric("cosine") \
            .where(query_plan.filter, prefilter=True) \
            .limit(top_k) \
            .to_df()
        )
    else:
        result = (
            table.search(query_vector[0]) \
            .metric("cosine") \
            .limit(top_k) \
            .to_df()
        )

    result.rename(columns={"vector": embeddings_column}, inplace=True)
    logger.info(f"Vector search yielded a DataFrame with {len(result):,} rows")
    return result
    

def auto_filter_fts_search(
    df: pd.DataFrame,
    query: str,
    text_column: str,
    embeddings_column: str,
    filter_fields: List[str],
    top_k: int = 20
) -> pd.DataFrame:
    """
    Performs a full-text search (FTS) on the given DataFrame based on the specified query and filter fields.

    Args:
        df (pd.DataFrame): The DataFrame to search.
        query (str): The query string to search for.
        text_column (str): The name of the column containing text to search.
        embeddings_column (str): The name of the column containing embeddings.
        filter_fields (List[str]): A list of fields to filter the search results.
        top_k (int): The maximum number of search results to return.

    Returns:
        pd.DataFrame: A DataFrame containing the top_k search results.
    """
    
    query_plan = generate_query_plan(
        input_df=df,
        query=query,
        filter_fields=filter_fields,
        )
    if not query_plan.filter:
        logger.info(f"No filters were identified for query: {query}")
    else:
        logger.info(f"Applying filter(s): {query_plan.filter}")
    
    # Check if a revised query exists and is not an empty string, use it if so
    if query_plan.rephrased_query and query_plan.rephrased_query.strip():
        logger.info(f"Revised query: {query_plan.rephrased_query}")
        search_query = query_plan.rephrased_query
    else:
        search_query = query_plan.original_query
    
    df.rename(columns={embeddings_column: "vector"}, inplace=True)
    uri = "../temp-lancedb/pd_table.lance"
    db = lancedb.connect(uri)
    filter_fields.append(text_column)
    filter_fields.append("vector")
    try:
        table = db.create_table("temp_lance", data=df[filter_fields], mode="create")
    except:
        table = db.create_table("temp_lance", data=df[filter_fields], mode="overwrite")
        
    table.create_fts_index(text_column, replace=True)
    
    # Clean up query: replace all newlines with spaces in query,
    # force special search keywords to lower case, remove quotes,
    # so it's not interpreted as search syntax
    query_clean = (
        search_query.replace("\n", " ")
        .replace("AND", "and")
        .replace("OR", "or")
        .replace("NOT", "not")
        .replace("'", "")
        .replace('"', "")
    )
    
    if query_plan.filter:
        result = (
            table.search(query_clean, query_type="fts") \
            .where(query_plan.filter) \
            .limit(top_k) \
            .to_df()
        )
    else:
        result = (
            table.search(query_clean, query_type="fts") \
            .limit(top_k) \
            .to_df()
        )
    result.rename(columns={"vector": embeddings_column}, inplace=True)
    logger.info(f"Full Text Search (FTS) search yielded a DataFrame with {len(result):,} rows")

    return result
