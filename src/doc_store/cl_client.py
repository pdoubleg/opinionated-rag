"""Download client for Harvard's Case Access Project."""

from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Union
from datetime import date, datetime
from pydantic import BaseModel, field_validator

import requests

from .utils import normalize_case_cite
from src.schema.citations import CAPCitation
from src.schema.decisions import Decision, DecisionWithContext, CaseBody, CaseData, Court, Opinion
from src.doc_store.courtlistener import Caselist, CourtListenerCaseDataSource

from src.types import SEARCH_TYPES

load_dotenv()


class CourtListenerClient(CourtListenerCaseDataSource):
    def __init__(self):
        super().__init__()
        
    def find_cite(self, case_id: int) -> Decision:
        case_data = super().find_cite(case_id)
        search_res = SearchResult(**case_data[0])
        return search_res
        
    def fetch_case_by_id(self, case_id: int) -> Decision:
        case_data = super().fetch_case(case_id)
        decision = self.make_decisions(case_data)
        return decision[0]
    
    def fetch_case_by_cite(self, citation: str) -> Decision:
        cite_data = super().find_cite(citation)
        decision = self.fetch_case_by_id(cite_data.id)
        return decision[0]
    
    def read_forward_citations(self, case_id: int, verbose: bool = True) -> List[Decision]:
        """
        Queries the Court Listener front end 'citation search' for case ids that cited a target case, then 
            queries each id and converting the results to a Decision.
        """
        forward_citing_cases = super().fetch_forward_citations(case_id, verbose=verbose)
        decisions = self.make_decisions(forward_citing_cases)
        return decisions
    
    def fetch_cited_by(self, case_id: int) -> List[int]:
        """Fetches a list of ids for cases cited by the target citation."""
        case = super().fetch_case(case_id)
        citing_ids = list(case.citing())
        return citing_ids
    
    def read_cited_by(self, case_id: int, depth: int = 1, verbose: bool = True) -> List[Decision]:
        case = super().fetch_case(case_id)
        cited_by_cases = super().fetch_cases_cited_by(case.cases[0], depth, verbose)
        decisions = self.make_decisions(cited_by_cases)
        return decisions
    
    
    def search_query(self, query: str) -> List[SearchResult]:
        search_res = self.basic_search(query)
        return [SearchResult(**s) for s in search_res]


    def make_decisions(self, caselist: Caselist) -> List[Decision]:
        """
        Converts a list of cases into a list of Decision objects.

        Args:
            caselist: A Caselist object containing multiple cases.

        Returns:
            A list of Decision objects, each representing a case in the caselist.
        """
        decisions = []
        for case in caselist.cases:
            # Fallback mechanism for citations
            if case.cluster.citations:
                citations = [
                    CAPCitation(
                        cite=str(citation),
                        reporter=citation.reporter,
                        category=citation.type.name.title(),
                        case_ids=[case.cluster.id],
                    )
                    for citation in case.cluster.citations
                ]
            else:
                # Fallback citation when case.cluster.citations is None
                citations = [
                    CAPCitation(
                        cite=case.name_short,
                        reporter=case.court,
                        category="Fallback", 
                        case_ids=[case.cluster.id],
                    )
                ]
            decision = Decision(
                id=case.opinions[0].id,
                decision_date=case.date,
                name=case.bluebook_citation,
                name_abbreviation=case.docket.case_name_short,
                docket_num=case.docket.docket_number,
                citations=citations,
                attorneys=case.people["attorneys"],
                court=Court(
                    id=case.docket.id,
                    name=case.court,
                    url=case.docket.court,
                ),
                casebody=CaseBody(
                    data=CaseData(
                        head_matter=case.cluster.headmatter,
                        opinions=[
                            Opinion(
                                type=opinion.type.name.title(),
                                author=case.people.get("judges", ""),
                                text=opinion.html,
                                is_html=True,
                            ) for opinion in case.opinions
                        ],
                        judges=case.people.get("judges", []),
                    ),
                    status=case.cluster.precedential_status.name,
                ),
                cites_to=case.opinions[0].citing_cases if case.opinions else [],
                frontend_url=case.opinions[0].web_link if case.opinions else "",
            )
            decisions.append(decision)
        return decisions

    # @classmethod
    # def extract_parallel_citation_context(
    #     forward_decisions: List[Decision], 
    #     citations: List[str],
    #     words_before: int = 400,
    #     words_after: int = 400
    #     ) -> List[DecisionWithContext]:
    #     """
    #     Extracts the context for the first found citation in each Decision object and returns a new list of Decision objects with the context stored in an attribute.
    #         When multiple citations exist for a given opinion, i.e., are parallel, each version is checked until a match is found.

    #     Args:
    #         forward_decisions: A list of Decision objects to search through.
    #         citations: A list of citation strings to search for in each Decision.

    #     Returns:
    #         A list of DecisionWithContext objects, each representing a Decision with an added context attribute.
    #     """
    #     updated_decisions = []

    #     for decision in forward_decisions:
    #         context_found = False
    #         extracted_context = None
            
    #         for citation in citations:
    #             if context_found:
    #                 break
                
    #             try:
    #                 context = decision.extract_citation_contexts(
    #                     citation=citation,
    #                     words_before=words_before,
    #                     words_after=words_after,
    #                 )
                    
    #                 if context:
    #                     extracted_context = context
    #                     context_found = True
    #             except Exception as e:
    #                 print(f"Error extracting context for citation {citation} in decision: {e}")
            
    #         # Create a new DecisionWithContext object, copying the original decision and adding the extracted context
    #         updated_decision = DecisionWithContext(**decision.__dict__, context=extracted_context, context_citation=citation)
    #         updated_decisions.append(updated_decision)

    #     return updated_decisions


class SearchResult(BaseModel):
    id: int
    dateFiled: Optional[date | datetime | str] = None
    citation: str
    docket_id: int
    cluster_id: int
    caseName: str
    court: str
    judge: str
    status: str
    snippet: str
    
    @field_validator("dateFiled", mode="before")
    def serialize_decision_date(cls, v: Union[date, str]) -> str:
        """
        Serializes the decision_date field to a string format.

        Args:
            decision_date (Union[date, str]): The decision date to be serialized.
            _info: Additional information passed to the serializer, not used.

        Returns:
            str: The serialized decision date as a string.
        """
        if isinstance(v, date):
            return v.isoformat()
        return v
    
    @field_validator("citation", mode="before")
    def fix_citations(cls, v: Union[date, str]) -> str:
        """
        Serializes the decision_date field to a string format.

        Args:
            decision_date (Union[date, str]): The decision date to be serialized.
            _info: Additional information passed to the serializer, not used.

        Returns:
            str: The serialized decision date as a string.
        """
        if isinstance(v, list):
            return ", ".join(v)
        return v
    
    def __str__(self):
        text_parts = []
        case_id = f"Case ID: {self.id}"
        text_parts.append(case_id)
        name_short = str(self.caseName)
        text_parts.append(name_short)
        date = f"Date Filed: {self.dateFiled[:10]}"
        text_parts.append(date)
        normalized_cite = f"Citation(s): {self.citation}"
        text_parts.append(normalized_cite)
        court = f"Court: {str(self.court)}"
        text_parts.append(court)
        snippet = self.snippet
        text_parts.append(snippet)
        return "\n\n".join(text_parts)