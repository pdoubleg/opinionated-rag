import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.search.query_planning import QueryFilterPlan, generate_query_plans
load_dotenv()
from functools import cached_property
from typing import List, Dict
from pydantic import BaseModel, Field

import instructor
import openai
from src.agent.tools.semantic_search import SemanticSearch, Filter
from src.search.models import QueryScreening
from src.search.query_planning import screen_query
from src.utils.gen_utils import DataFrameCache
from src.utils.settings import get_settings

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

get_settings()

DATA_PATH = 'data/splade.parquet'

df_cache = DataFrameCache(DATA_PATH)

df = df_cache.df

query = """
Regarding the pollution exclusion clause under the terms of comprehensive general liability (CGL) insurance, \
how has the California court defined the phrase 'sudden and accidental', in particular for polluting events? \
Also, has there been any consideration for intentional vs unintentional polluting events?
"""

logger.info(f"Test Query: {query}")

class SearchRequest(BaseModel):
    query: str
    initial_filters: Filter | List[Filter] | None = Field(
        None,
        description="Metadata filters to apply prior to search.",
    )
    query_flow: QueryScreening | None = None
    
    
new_query: SearchRequest = SearchRequest(
    query=query,
    initial_filters=None,
    query_flow=screen_query(user_query=query)
)

logger.info(f"Query type: {new_query.query_flow.intent}")
logger.info(f"Query topic: {new_query.query_flow.topic}")
logger.info(f"Potential sub-questions: {str(new_query.query_flow.n_subquestions)}")


query_plan: List[QueryFilterPlan] = generate_query_plans(
    input_df=df,
    query=new_query.query,
    filter_fields=[
        'court_name',
    ],
    n_subquestions=new_query.query_flow.n_subquestions,
)

logger.info(f"Sub-questions generated: {len(query_plan)}")
for index, plan in enumerate(query_plan):
    logger.info(f"Plan {index + 1}:")
    if plan.filter:
        logger.info(f"\tFilter: {plan.filter}")
    else:
        logger.info("\tNo filter applied.")
    logger.info(f"\tRephrased Query: {plan.rephrased_query}")


