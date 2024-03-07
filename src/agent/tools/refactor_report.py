import os
import re
import sys
import instructor
import openai
import pandas as pd
import warnings
import numpy as np
from pathlib import Path
import logging

from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import re
from pathlib import Path

pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

from dotenv import load_dotenv
from pydantic import BaseModel, Field, root_validator, validator, ConfigDict

from typing import Optional, Type, Generic, TypeVar, List, Dict, Any
from datetime import datetime, timedelta

from src.utils.citations import create_annotated_text
from src.agent.tools.semantic_search import Filter, SemanticSearch
from src.embedding_models.models import ColbertReranker
from src.agent.tools.utils import (
    ResearchReport,
    aget_fact_patterns_df,
    extract_citation_numbers_in_brackets,
    get_claim_numbers_from_citations,
    get_llm_fact_pattern_summary,
    generate_citation_strings,
    create_formatted_input,
    get_final_answer,
    clean_string,
)


from markdown import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration


DEFAULT_MODEL_NAME = "gpt-4-turbo-preview"
TOKEN_LIMIT = 3000
DATA_PATH = "./data/reddit_legal_cluster_test_results.parquet"
TEXT_COLUMN_NAME = "body"


class ResearchPastQuestions(BaseModel):
    df: pd.DataFrame = Field(...)
    filter_criteria: Optional[Dict[str, Any] | Filter] = Field(
        None,
        description="Pre-filters for top-k vector search based on column value pairs.",
    )
    include_sources: Optional[bool] = Field(
        True, description="Whether model output should include source text."
    )
    must_have_opinion: Optional[bool] = Field(
        False, description="Whether to limit search to cases with available opinions."
    )
    linkable_citations: Optional[bool] = Field(
        True,
        description="Whether to update citations with hyperlinks public legal databases.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)

    async def run(
        self,
        user_query: str,
        rerank: bool = True,
        model_name: str = DEFAULT_MODEL_NAME,
        context_token_limit: int = TOKEN_LIMIT,
    ) -> str:
        df, opinion_df = self.prepare_dataframes()

        search_engine = SemanticSearch(df)
        query_clean = get_llm_fact_pattern_summary(user_query)
        top_n_res_df = await self.perform_search(query_clean, search_engine)

        if rerank:
            top_n_res_df = self.rerank_results(query_clean, top_n_res_df)

        formatted_input = create_formatted_input(
            top_n_res_df, user_query, context_token_limit=context_token_limit
        )
        print("Generating final response...")
        response_model = get_final_answer(formatted_input, model_name=model_name)

        result = self.format_result(
            top_n_res_df, opinion_df, user_query, response_model
        )

        return result, response_model

    def prepare_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepares and returns the main and opinion dataframes."""
        df = self.df.copy()
        opinion_df = pd.read_parquet(DATA_PATH)
        if self.must_have_opinion:
            opinion_claim_numbers = opinion_df["id"].unique().tolist()
            df = df[df["id"].isin(opinion_claim_numbers)]
        return df, opinion_df

    async def perform_search(
        self, query_clean: Any, search_engine: SemanticSearch
    ) -> pd.DataFrame:
        """Performs the semantic search and returns the top N results dataframe."""
        print("Performing vector search...")
        top_n_res_df = await search_engine.query_similar_documents(
            query_clean.summary,
            filter_criteria=self.filter_criteria or None,
            top_n=10,
            use_cosine_similarity=True,
            similarity_threshold=0.97,
        )
        return await aget_fact_patterns_df(top_n_res_df, TEXT_COLUMN_NAME, "id")

    def rerank_results(self, query_clean: Any, df: pd.DataFrame) -> pd.DataFrame:
        """Reranks the search results and returns the updated dataframe."""
        print("Reranking with ColBERTv2...")
        reranker = ColbertReranker(column="summary")
        return reranker.rerank(query=query_clean.summary, results_df=df)

    def format_result(
        self,
        df: pd.DataFrame,
        opinion_df: pd.DataFrame,
        user_query: str,
        response_model: ResearchReport,
    ) -> str:
        """Formats the final result string based on user query and response model.

        Args:
            df (pd.DataFrame): The main dataframe.
            opinion_df (pd.DataFrame): The opinion dataframe.
            user_query (str): The user's query.
            response_model (ResearchReport): The response model containing the research report.

        Returns:
            str: The formatted result string.
        """
        result = self._format_standard_result(df, user_query, response_model)
        if self.include_sources:
            result += self._format_sources(df, opinion_df, response_model)
        return result

    def _format_standard_result(
        self, df: pd.DataFrame, user_query: str, response_model: ResearchReport
    ) -> str:
        """Formats the standard part of the result string.

        Args:
            user_query (str): The user's query.
            response_model (ResearchReport): The response model containing the research report.

        Returns:
            str: The formatted standard result string.
        """
        result = "# New Referral\n"
        result += f"\n{user_query}\n"
        result += '<h1 id="research-results">Research Results</h1>'
        if self.filter_criteria is not None:
            first_key, first_value = next(iter(self.filter_criteria.items()))
            filter_string = f'<p id="search-criteria">Search Criteria: {first_key.replace("_", " ").title()} = {first_value}</p>'
            result += filter_string
        else:
            result += '<p id="search-criteria">Search Criteria: All States</p>'
        result += "\n\n_The following summary was generated by GPT..._"
        result += f"<br>\n{response_model.research_report}\n"
        result += "\n\n___\n\n"
        result += '<h2 id="sources">Sources</h2>'
        sources_intro = (
            "<p class='sources-intro'>This report is derived from a search</p>"
        )
        # Extract citations used by the LLM
        citation_numbers = extract_citation_numbers_in_brackets(
            response_model.research_report
        )
        print(f"\nCases selected by LLM: {citation_numbers}\n")

        citation_strings = generate_citation_strings(
            citation_numbers=citation_numbers,
            df=df,
            opinion_claim_numbers=df["id"].tolist(),
        )

        for citation in citation_strings:
            result += citation
        result += "\n\n___\n\n"
        result += sources_intro
        return result

    def _format_sources(
        self, df: pd.DataFrame, opinion_df: pd.DataFrame, response_model: ResearchReport
    ) -> str:
        """Formats the sources part of the result string if include_sources is True.

        Args:
            df (pd.DataFrame): The main dataframe.
            opinion_df (pd.DataFrame): The opinion dataframe.
            response_model (ResearchReport): The response model containing the research report.

        Returns:
            str: The formatted sources part of the result string.
        """
        result = ""
        cited_claim_df = get_claim_numbers_from_citations(
            citation_numbers=extract_citation_numbers_in_brackets(
                response_model.research_report
            ),
            df=df,
        )

        matched_opinions = self._get_opinion_data(
            df=cited_claim_df, opinion_df=opinion_df
        )

        if len(matched_opinions) > 0:
            result += "\n\n___\n\n"
            result += "<br><br><br><br><br>"
            result += '<h1 id="coverage-opinions">Coverage Opinions</h1>'

            for _, row in matched_opinions.iterrows():
                result += self._format_opinion_entry(row)
        return result

    def _get_opinion_data(
        self, df: pd.DataFrame, opinion_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges the cited claims with the opinion dataframe and sorts them.

        Args:
            df (pd.DataFrame): The dataframe containing cited claims.
            opinion_df (pd.DataFrame): The opinion dataframe.

        Returns:
            pd.DataFrame: The merged and sorted dataframe.
        """
        cited_opinion_df = df.merge(
            opinion_df[["id", "Citation"]],
            how="inner",
            left_on="id",
            right_on="id",
        )
        if len(cited_opinion_df) > 0:
            cited_opinion_df = cited_opinion_df.sort_values(by="Citation")
        return cited_opinion_df

    def _format_opinion_entry(self, row: pd.Series) -> str:
        """Formats a single opinion entry for the sources section.

        Args:
            row (pd.Series): A row from the dataframe containing opinion data.

        Returns:
            str: The formatted opinion entry.
        """
        citation = row["citation"]
        claim_number = row["id"]
        file_name = row["llm_title"]
        state = row["State"]
        year_month = "2020 Jan"
        author = "chat-gpt"
        opinion = row["body"]
        claim_number_for_link = clean_string(str(claim_number))

        link = row["full_link"]
        claim_number_formatted = f"**Claim Number: [{claim_number_for_link}]({link})**"

        if self.linkable_citations:
            opinion = create_annotated_text(opinion)
        opinion_anchor = f'<div class="footnotes">\n<div id="fn-{int(citation)}">\n<hr />\n<p><strong>{[int(citation)]}</strong> <em>{row["llm_title"]}</em>&#160;<a href="#fnref-{int(citation)}" class="footnoteBackLink" title="Back to the summary.">&#8617;</a></p>\n</div>\n</div>'
        workspace_state_display = f"\n\n**{file_name} | {state}**"
        author_display = f"\n\n### **{author}** ({year_month}):"
        claim_number_display = f"\n\n{claim_number_formatted}"
        imanage_document = f"\n**iManage Document [Link]({row['full_link']})**\n"
        imanage_workspace = f"\n**iManage Workspace [Link]({row['full_link']})**\n"
        return f"{opinion_anchor}{workspace_state_display}{claim_number_display}{imanage_document}{imanage_workspace}{author_display}\n{opinion}\n\n[[return to the top]](#top)\n\n___\n\n"
