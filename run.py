import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tenacity import Retrying, stop_after_attempt, wait_fixed

from src.agent.tools.utils import aget_fact_patterns_df, get_final_answer

load_dotenv()
from functools import cached_property
from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field

import instructor
import openai

# Query Planning
from src.search.models import QueryScreening
from src.search.query_planning import screen_query
from src.search.query_expansion import generate_subquestions, segment_search_query
from src.search.query_planning import (
    QueryFilterPlan,
    generate_query_plans,
)

# Search
from src.agent.tools.semantic_search import SemanticSearch, Filter
from src.agent.tools.splade_search import SparseEmbeddingsSplade
from src.parsing.search import find_closest_matches_with_bm25_df
from src.search.query_planning import (
    auto_filter_vector_search,
    auto_filter_fts_search,
)

# Re-Ranking
from src.embedding_models.models import ColbertReranker
from src.search.rank_gpt import RankGPTRerank
from src.search.rerank_openai import generate_gpt_relevance

# Filtering and Joining
from src.search.llm_filter import filter_chunks
from src.search.doc_joiner import DocJoinerDF

# Utilities
from src.utils.gen_utils import DataFrameCache
from src.utils.settings import get_settings

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

get_settings()

DATA_PATH = "data/splade.parquet"

df_cache = DataFrameCache(DATA_PATH)

df = df_cache.df



class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResults(BaseModel):
    query: str
    results: List[SearchResult]
    

def dataframe_to_search_results(
    df: pd.DataFrame,
    text_column: str = 'text',
    score_column: str = 'score',
    metadata_fields: List[str] = [], 
) -> List[SearchResult]:

    search_results = []
    for _, row in df.iterrows():
        metadata = {field: row[field] for field in metadata_fields}
        node = SearchResult(
            text=row[text_column],
            score=row[score_column],
            metadata=metadata,
            )        
        search_results.append(node)
        
    return search_results


class RerankRequest(BaseModel):
    query: str
    search_results: SearchResults


class RerankResponse(BaseModel):
    query: str
    reranked_results: List[SearchResult]


class FilterRequest(BaseModel):
    query: str
    search_results: SearchResults


class FilterResponse(BaseModel):
    query: str
    filtered_results: List[SearchResult]


class JoinRequest(BaseModel):
    query: str
    result_lists: List[List[SearchResult]]


class JoinResponse(BaseModel):
    query: str
    joined_results: List[SearchResult]


class ResearchReport(BaseModel):
    query: str = Field(..., description="The original query or sub-question.")
    research_report: str = Field(
        ...,
        description="A legal style research report comparing the query with similar past issues.",
    )
    source_documents: List[SearchResult] = Field(
        default_factory=list,
        description="A list of source documents used to generate the research report.",
    )


from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class StepConfig(BaseModel):
    step_type: str
    use_summary: bool = False

class RAGPipeline:
    def __init__(
        self, 
        steps: List[StepConfig], 
        max_subquestions: int = 3,
        max_docs_per_subquestion: int = 5,
        model_name: str = "gpt-4"
    ):
        self.steps = steps
        self.max_subquestions = max_subquestions
        self.max_docs_per_subquestion = max_docs_per_subquestion
        self.model_name = model_name
    
    async def run(self, query: str, df: pd.DataFrame, metadata_list: List[str]):
        # Query Screening
        query_screening = screen_query(query)

        # Query Expansion 
        subqueries = generate_subquestions(query, str(min(query_screening.n_subquestions, self.max_subquestions)))

        research_reports = []
        for subquery in subqueries.questions:
            # Search
            search_results = SearchResults(query=subquery, results=[])
            for step_config in self.steps:
                if step_config.step_type == "SemanticSearch":
                    results = SemanticSearch(df=df, text_column="text", embedding_column="openai_embeddings").query_similar_documents(subquery, top_k=self.max_docs_per_subquestion)
                    search_result_list = dataframe_to_search_results(df=results, metadata_fields=metadata_list)
                elif step_config.step_type == "SparseEmbeddingsSplade":
                    results = SparseEmbeddingsSplade(df=df, text_column="text", embedding_column="splade_embeddings").query_similar_documents(subquery, top_k=self.max_docs_per_subquestion)
                    search_result_list = dataframe_to_search_results(df=results, metadata_fields=metadata_list)
                elif step_config.step_type == "BM25":
                    results = find_closest_matches_with_bm25_df(df, subquery, top_k=self.max_docs_per_subquestion)
                    search_result_list = dataframe_to_search_results(df=results, metadata_fields=metadata_list)
                else:
                    pass
                
                search_results.results.extend([SearchResults(query=subquery, results=search_result_list)])

            # Fact Pattern Summary
            if any(step_config.step_type == "FactPatternSummary" for step_config in self.steps):
                results_df = pd.DataFrame([SearchResult(text=result.text, score=result.score) for result in search_results.results])
                results_df = await aget_fact_patterns_df(results_df, "text", "index")
                search_results.results = [SearchResult(text=row["text"], score=row["score"], metadata={"summary": row["summary"]}) for _, row in results_df.iterrows()]

            # Reranking and Filtering
            for step_config in self.steps:
                if step_config.step_type == "ColbertReranker":
                    rerank_request = RerankRequest(query=subquery, search_results=SearchResults(query=subquery, results=[SearchResult(text=result.metadata["summary"] if step_config.use_summary else result.text, score=result.score) for result in search_results.results]))
                    reranked = ColbertReranker().rerank(rerank_request)
                    search_results.results = [SearchResult(text=result.text, score=result.score, metadata={"summary": next((r.metadata["summary"] for r in search_results.results if r.text == result.text), None)}) for result in reranked.reranked_results]
                elif step_config.step_type == "RankGPTRerank":
                    rerank_request = RerankRequest(query=subquery, search_results=SearchResults(query=subquery, results=[SearchResult(text=result.metadata["summary"] if step_config.use_summary else result.text, score=result.score) for result in search_results.results]))
                    reranked = RankGPTRerank().rerank(rerank_request)
                    search_results.results = [SearchResult(text=result.text, score=result.score, metadata={"summary": next((r.metadata["summary"] for r in search_results.results if r.text == result.text), None)}) for result in reranked.reranked_results]
                elif step_config.step_type == "GPTRelevance":
                    rerank_request = RerankRequest(query=subquery, search_results=SearchResults(query=subquery, results=[SearchResult(text=result.metadata["summary"] if step_config.use_summary else result.text, score=result.score) for result in search_results.results]))
                    reranked = generate_gpt_relevance(rerank_request)
                    search_results.results = [SearchResult(text=result.text, score=result.score, metadata={"summary": next((r.metadata["summary"] for r in search_results.results if r.text == result.text), None)}) for result in reranked.reranked_results]
                elif step_config.step_type == "FilterChunks":
                    filter_request = FilterRequest(query=subquery, search_results=SearchResults(query=subquery, results=[SearchResult(text=result.metadata["summary"] if step_config.use_summary else result.text, score=result.score) for result in search_results.results]))
                    filtered = filter_chunks(filter_request)
                    search_results.results = [SearchResult(text=result.text, score=result.score, metadata={"summary": next((r.metadata["summary"] for r in search_results.results if r.text == result.text), None)}) for result in filtered.filtered_results]

            # Get Final Answer
            formatted_input = "\n".join([f"{result.text}" for result in search_results.results])
            research_report = get_final_answer(formatted_input, self.model_name)
            research_report.query = subquery
            research_report.source_documents = search_results.results
            research_reports.append(research_report)

        # Aggregate Sub-question Research Reports
        # aggregated_report = aggregate_research_reports(query, research_reports)

        return research_reports

# Example usage
pipeline = RAGPipeline(
    steps=[
        StepConfig(step_type="SemanticSearch"),
        StepConfig(step_type="SparseEmbeddingsSplade"),
        # StepConfig(step_type="FactPatternSummary"),
        # StepConfig(step_type="ColbertReranker", use_summary=False),
        # StepConfig(step_type="RankGPTRerank", use_summary=False),
        # StepConfig(step_type="GPTRelevance", use_summary=True),
        # StepConfig(step_type="FilterChunks", use_summary=False),
    ],
    max_subquestions=3,
    max_docs_per_subquestion=5,
    model_name="gpt-4"
)

async def main():
    df['text'] = df['context']
    metadata_list = ['name_abbreviation', 'context_citation']
    query = """
    Regarding the pollution exclusion clause under the terms of comprehensive general liability (CGL) insurance, \
    how has the California court defined the phrase 'sudden and accidental', in particular for polluting events? \
    Also, has there been any consideration for intentional vs unintentional polluting events?
    """
    logger.info(f"Test Query: {query}")

    results = await pipeline.run(query, df, metadata_list)

import asyncio
asyncio.run(main())

