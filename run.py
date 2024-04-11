import asyncio
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

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class SearchResults(BaseModel):
    query: str
    results: List[SearchResult]

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict):
        data["results"] = [SearchResult.from_dict(result) for result in data["results"]]
        return cls(**data)


class FilterRequest(BaseModel):
    query: str
    search_results: SearchResults


class RerankRequest(BaseModel):
    query: str
    search_results: SearchResults


class RerankResponse(BaseModel):
    query: str
    reranked_results: List[SearchResult]


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


class PipelineStep(BaseModel):
    step_type: str
    use_summary: bool = False
    top_k: Optional[int] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)


def dataframe_to_search_results(
    df: pd.DataFrame,
    text_column: str = "text",
    score_column: str = "score",
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


def search_results_to_dataframe(search_results: SearchResults) -> pd.DataFrame:
    data = [
        {
            "text": result.text,
            "score": result.score,
            "metadata": result.metadata,
        }
        for result in search_results.results
    ]
    return pd.DataFrame(data)


class RAGPipeline:
    def __init__(
        self,
        steps: List[PipelineStep],
        max_subquestions: int = 3,
        model_name: str = "gpt-4-turbo",
        metadata_list: List[str] | None = [],
    ):
        self.steps = steps
        self.max_subquestions = max_subquestions
        self.model_name = model_name
        self.metadata_list = metadata_list

    async def process_step(
        self, step: PipelineStep, subquery: str, search_results: SearchResults
    ):
        if step.step_type == "SemanticSearch":
            results = SemanticSearch(**step.kwargs).query_similar_documents(
                subquery, top_k=step.top_k
            )
            search_result_list = dataframe_to_search_results(
                df=results, metadata_fields=self.metadata_list
            )
        elif step.step_type == "SparseEmbeddingsSplade":
            results = SparseEmbeddingsSplade(**step.kwargs).query_similar_documents(
                subquery, top_k=step.top_k
            )
            search_result_list = dataframe_to_search_results(
                df=results, metadata_fields=self.metadata_list
            )
        elif step.step_type == "BM25":
            results = find_closest_matches_with_bm25_df(
                step.kwargs["df"], "text", subquery, top_k=step.top_k
            )
            search_result_list = dataframe_to_search_results(
                df=results, metadata_fields=self.metadata_list
            )
        elif step.step_type == "FactPatternSummary":
            results_df = pd.DataFrame(
                [result.to_dict() for result in search_results.results]
            )
            results_df = await aget_fact_patterns_df(results_df, "text")
            search_result_list = [
                SearchResult.from_dict(row) for _, row in results_df.iterrows()
            ]
        elif step.step_type == "ColbertReranker":
            rerank_request = RerankRequest(
                query=subquery,
                search_results=SearchResults(
                    query=subquery,
                    results=[
                        SearchResult(
                            text=result.metadata["summary"]
                            if step.use_summary
                            else result.text,
                            score=result.score,
                        )
                        for result in search_results.results
                    ],
                ),
            )
            results_df = search_results_to_dataframe(rerank_request.search_results)
            reranked = ColbertReranker(column='text').rerank(rerank_request.query, results_df)
            search_result_list = [
                SearchResult(
                    text=row["text"],
                    score=row["score"],
                    metadata={
                        "summary": next(
                            (
                                r.metadata["text"]
                                for r in search_results.results
                                if r.text == row["text"]
                            ),
                            None,
                        )
                    },
                )
                for _, row in reranked.iterrows()
            ]
        elif step.step_type == "RankGPTRerank":
            rerank_request = RerankRequest(
                query=subquery,
                search_results=SearchResults(
                    query=subquery,
                    results=[
                        SearchResult(
                            text=result.metadata["summary"]
                            if step.use_summary
                            else result.text,
                            score=result.score,
                        )
                        for result in search_results.results
                    ],
                ),
            )
            reranked = RankGPTRerank().rerank(rerank_request)
            search_result_list = [
                SearchResult(
                    text=result.text,
                    score=result.score,
                    metadata={
                        "summary": next(
                            (
                                r.metadata["summary"]
                                for r in search_results.results
                                if r.text == result.text
                            ),
                            None,
                        )
                    },
                )
                for result in reranked.reranked_results
            ]
        elif step.step_type == "GPTRelevance":
            rerank_request = RerankRequest(
                query=subquery,
                search_results=SearchResults(
                    query=subquery,
                    results=[
                        SearchResult(
                            text=result.metadata["summary"]
                            if step.use_summary
                            else result.text,
                            score=result.score,
                        )
                        for result in search_results.results
                    ],
                ),
            )
            reranked = generate_gpt_relevance(rerank_request)
            search_result_list = [
                SearchResult(
                    text=result.text,
                    score=result.score,
                    metadata={
                        "summary": next(
                            (
                                r.metadata["summary"]
                                for r in search_results.results
                                if r.text == result.text
                            ),
                            None,
                        )
                    },
                )
                for result in reranked.reranked_results
            ]
        elif step.step_type == "FilterChunks":
            filter_request = FilterRequest(
                query=subquery,
                search_results=SearchResults(
                    query=subquery,
                    results=[
                        SearchResult(
                            text=result.metadata["summary"]
                            if step.use_summary
                            else result.text,
                            score=result.score,
                        )
                        for result in search_results.results
                    ],
                ),
            )
            filtered = filter_chunks(filter_request)
            search_result_list = [
                SearchResult(
                    text=result.text,
                    score=result.score,
                    metadata={
                        "summary": next(
                            (
                                r.metadata["summary"]
                                for r in search_results.results
                                if r.text == result.text
                            ),
                            None,
                        )
                    },
                )
                for result in filtered.filtered_results
            ]
        else:
            search_result_list = search_results.results

        return SearchResults(query=subquery, results=search_result_list)

    async def process_subquery(self, subquery: str):
        search_results = SearchResults(query=subquery, results=[])
        for step in self.steps:
            search_results = await self.process_step(step, subquery, search_results)

        formatted_input = "\n".join(
            [f"{result.text}" for result in search_results.results]
        )
        research_report = get_final_answer(formatted_input, self.model_name)
        research_report.query = subquery
        research_report.source_documents = search_results.results
        return research_report

    async def run(self, query: str, df: pd.DataFrame):
        query_screening = screen_query(query)
        subqueries = generate_subquestions(
            query, str(min(query_screening.n_subquestions, self.max_subquestions))
        )

        tasks = [self.process_subquery(subquery) for subquery in subqueries.questions]
        research_reports = await asyncio.gather(*tasks)

        return research_reports


# Example usage
pipeline = RAGPipeline(
    steps=[
        PipelineStep(
            step_type="SemanticSearch",
            top_k=5,
            kwargs={
                "df": df,
                "text_column": "text",
                "embedding_column": "openai_embeddings",
            },
        ),
        PipelineStep(
            step_type="SparseEmbeddingsSplade",
            top_k=5,
            kwargs={
                "df": df,
                "text_column": "text",
                "embedding_column": "splade_embeddings",
            },
        ),
        PipelineStep(step_type="BM25", top_k=5, kwargs={"df": df}),
        PipelineStep(step_type="FactPatternSummary"),
        PipelineStep(step_type="ColbertReranker", use_summary=False),
        PipelineStep(step_type="RankGPTRerank", use_summary=False),
        PipelineStep(step_type="GPTRelevance", use_summary=True),
        PipelineStep(step_type="FilterChunks", use_summary=False),
    ],
    max_subquestions=2,
    model_name="gpt-4-turbo",
    metadata_list=["name_abbreviation", "context_citation"],
)


async def main():
    df["text"] = df["context"]
    query = """
    Regarding the pollution exclusion clause under the terms of comprehensive general liability (CGL) insurance, \
    how has the California court defined the phrase 'sudden and accidental', in particular for polluting events? \
    Also, has there been any consideration for intentional vs unintentional polluting events?
    """
    logger.info(f"Test Query: {query}")

    results = await pipeline.run(query, df)


asyncio.run(main())
