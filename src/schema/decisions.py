"""Data models for published judicial decisions."""

from __future__ import annotations

import datetime
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Union
import eyecite

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_serializer, field_validator
from html2text import html2text
from eyecite.models import CaseCitation

from src.utils.citations import get_citation_context
from src.doc_store.utils import markdown_to_text, normalize_case_cite

from .citations import CAPCitation


class ReporterVolume(BaseModel):
    """
    Represents a collection of judicial decisions published in a single volume.

    Attributes:
        id (int): The unique identifier for this volume, corresponding to the volume number in Court Listener's Citation.volume.
        url (HttpUrl): The URL pointing to this volume's information in the Case Access Project API, derived from OpinionCluster.absolute_url and a base URL.
        full_name (str): The complete name of this volume as it appears in the publication.

    """

    id: int = Field(
        ...,
        description="The unique identifier for this volume, corresponding to the volume number in Court Listener's Citation.volume.",
    )
    url: HttpUrl = Field(
        ...,
        description="The URL pointing to this volume's information in the Case Access Project API, derived from OpinionCluster.absolute_url and a base URL.",
    )
    full_name: str
    
    @field_serializer('url')
    def serialize_reporter_url(self, url: HttpUrl, _info):
        return str(url)


class Court(BaseModel):
    """
    Represents a court entity that issues legal decisions.

    Attributes:
        id (Optional[int]): The unique identifier for this court. Defaults to None.
        name (Optional[str]): The name of this court. Defaults to None.
        url (Optional[HttpUrl]): The URL of this court in the Case Access Project API. Defaults to None.
        whitelisted (Optional[bool]): Indicates if this court's decisions are whitelisted for public access. Defaults to None.
        name_abbreviation (Optional[str]): The abbreviation of this court's name. Defaults to None.
    """

    id: Optional[int] = None
    name: Optional[str] = None
    url: Optional[HttpUrl] = None
    whitelisted: Optional[bool] = None
    name_abbreviation: Optional[str] = None

    @field_serializer('url')
    def serialize_court_url(self, url: HttpUrl, _info):
        return str(url)

class Jurisdiction(BaseModel):
    """
    Represents a jurisdiction, which is an authority that governs or oversees a legal system.

    Attributes:
        id (int): A unique identifier for the jurisdiction.
        name (str): The official name of the jurisdiction.
        url (HttpUrl): The URL pointing to this jurisdiction's information in the Case Access Project API.
        slug (str): A URL-friendly slug representing the jurisdiction's name.
        whitelisted (bool): Indicates if the jurisdiction's cases are freely accessible in the Case Access Project API.
        name_abbreviation (str): A short, abbreviated form of the jurisdiction's name.

    Note:
        The `name` field corresponds to the `Docket.jurisdiction_type` in the Court Listener database.
    """

    id: Optional[int] = None
    name: Optional[str] = None
    url: Optional[HttpUrl] = None
    slug: Optional[str] = None
    whitelisted: Optional[bool] = None
    name_abbreviation: Optional[str] = None
    
    @field_serializer('url')
    def serialize_judicial_url(self, url: HttpUrl, _info):
        return str(url)


class Opinion(BaseModel):
    """
    A document that resolves legal issues in a case and posits legal holdings.

    Usually, an opinion must have ``type="majority"`` to create holdings binding on any courts.

    Args:
        type (str): The opinion's attitude toward the court's disposition of the case.
            e.g., ``majority``, ``dissenting``, ``concurring``, ``concurring in the result``.
        author (str): Name of the judge who authored the opinion, if identified.
        text (str): The text of the opinion.
    """
    # model_config = ConfigDict(
    #     extra="allow",
    #     arbitrary_types_allowed=True,
    # )

    type: str = "majority"
    author: Optional[str] = None
    text: str
    is_html: Optional[bool] = False

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(type="{self.type}", author="{self.author}")'

    def __str__(self):
        result = f"{self.type} opinion" if self.type else "opinion"
        return result

    @field_validator("author", mode="before")
    def remove_author_name_punctuation(cls, v: str | None) -> str:
        """Normalize Opinion author names by removing punctuation."""
        result = v or ""
        result = result.replace("Judge.", "Judge").replace("Justice.", "Justice")
        return result


class CaseData(BaseModel):
    """The content of a Decision, including Opinions."""

    head_matter: Optional[str] = Field(
        None, description="Court Listener is Cluster.headmatter"
    )
    opinions: List[Opinion] = []
    parties: Optional[str | List[str]] = []
    judges: Optional[str | List[str]] = []


class CaseBody(BaseModel):
    """Data about an Opinion in the form used by the Caselaw Access Project API."""

    data: Union[str, CaseData]
    status: Optional[str] = Field(
        None, 
        description="Court Listener is OpinionCluster.precedential_status"
    )


class Decision(BaseModel):
    r"""
    A court decision to resolve a step in litigation.

    This class uses the model of a judicial decision from the Caselaw Access Project API. \
        It is designed to handle records that may contain multiple `Opinion` instances. \
        Typically, one record will contain all the `Opinion` instances from one appeal, \
        but not necessarily from the entire lawsuit. A lawsuit may contain multiple appeals \
        or other petitions, and if more than one of those generates published Opinions, the CAP API \
        will divide those Opinions into separate records for each appeal. The outcome of a decision \
        may be determined by one majority `Opinion` or by the combined effect of multiple Opinions. \
        The lead opinion is commonly, but not always, the only Opinion that creates binding legal authority. \
        Usually, every rule posited by the lead Opinion is binding, but some may not be, often because parts \
        of the Opinion fail to command a majority of the panel of judges.

    Attributes:
        decision_date (datetime.date): The date when the opinion was first published by the court (not the publication date of the reporter volume).
        name (Optional[str]): The full name of the opinion, e.g., "ORACLE AMERICA, INC., Plaintiff-Appellant, v. GOOGLE INC., Defendant-Cross-Appellant".
        name_abbreviation (Optional[str]): The shorter name of the opinion, e.g., "Oracle America, Inc. v. Google Inc.".
        docket_num (Optional[str]): The docket number of the case.
        citations (Optional[Sequence[CAPCitation]]): Ways to cite this Decision.
        parties (List[str]): The parties involved in the case.
        attorneys (List[str]): The attorneys representing the parties.
        first_page (Optional[int]): The page where the opinion begins in its official reporter.
        last_page (Optional[int]): The page where the opinion ends in its official reporter.
        court (Optional[Court]): The name of the court that published the opinion.
        casebody (Optional[CaseBody]): The Decision content including Opinions.
        jurisdiction (Optional[Jurisdiction]): The jurisdiction of the case.
        cites_to (Optional[Sequence[CAPCitation]]): CAPCitations to other Decisions.
        id (Optional[int]): The unique ID from CAP API.
        last_updated (Optional[datetime.datetime]): The date when the record was last updated in CAP API.
        frontend_url (Optional[HttpUrl]): The URL to the decision in CAP's frontend.
    """
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=False,
    )
    decision_date: Optional[str] = None
    name: Optional[str] = None
    name_abbreviation: Optional[str] = None
    docket_num: Optional[str] = None
    citations: Optional[Sequence[CAPCitation]] = None
    parties: Optional[List[str]] = []
    attorneys: Optional[Union[str, List[str]]] = None
    first_page: Optional[int] = None
    last_page: Optional[int] = None
    court: Optional[Court] = None
    casebody: Optional[CaseBody] = None
    jurisdiction: Optional[Jurisdiction] = None
    cites_to: Optional[Union[str, List[str], List[CAPCitation], Any]] = None
    id: Optional[int] = None
    last_updated: Optional[str] = None
    frontend_url: Optional[HttpUrl] = None
    forward_citation_ids: Optional[List[str | int]] = None
    
    @field_validator("decision_date", mode="before")
    def decision_date_must_include_day(
        cls, v: Union[datetime.date, str]
    ) -> Union[datetime.date, str]:
        """
        Ensures the decision_date includes a day if provided as a string in YYYY-MM format.

        Args:
            v (Union[datetime.date, str]): The decision date value to validate.

        Returns:
            Union[datetime.date, str]: The validated or modified decision date.
        """
        if isinstance(v, str) and len(v) == 7:  # YYYY-MM format
            return v + "-01"
        return v
    
    @field_validator("decision_date", mode="before")
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
    
    @field_validator("last_updated", mode="before")
    def serialize_updated_date(cls, v: Union[date, str]) -> str:
        """
        Serializes the last_updated field to a string format.

        Args:
            last_updated (Union[date, str]): The decision date to be serialized.
            _info: Additional information passed to the serializer, not used.

        Returns:
            str: The serialized decision date as a string.
        """
        if isinstance(v, date):
            return v.isoformat()
        return v
    
    @field_serializer('frontend_url')
    def serialize_frontend_url(self, frontend_url: HttpUrl, _info) -> str:
        """
        Serializes the frontend_url field to a string format.

        Args:
            frontend_url (HttpUrl): The frontend URL to be serialized.
            _info: Additional information passed to the serializer, not used.

        Returns:
            str: The serialized frontend URL as a string.
        """
        return str(frontend_url)

    def __str__(self) -> str:
        """
        String representation of the Decision instance.

        Returns:
            str: A string representation combining name, citation, and decision date.
        """
        citation = self.citations[0].cite if self.citations else ""
        name = self.name_abbreviation or self.name
        decision_date_str = self.serialize_decision_date(self.decision_date)
        return f"{name}, {citation} ({decision_date_str})"

    
    @property
    def text(self) -> Optional[str]:
        """
        Get the decision text with priority logic.

        If the casebody data is in HTML format, it returns the HTML text.
        Otherwise, it returns the text from the first opinion if available.

        Returns:
            Optional[str]: The decision text or None if not available.
        """
        if isinstance(self.casebody.data, str):
            return self.casebody.data
        elif self.casebody.data.opinions:
            return self.casebody.data.opinions[0].text
        return None
    
    @property
    def text_clean(self) -> Optional[str]:
        """
        Get the decision text with priority logic.

        If the casebody data is in HTML format, it returns the HTML text.
        Otherwise, it returns the text from the first opinion if available.

        Returns:
            Optional[str]: The decision text or None if not available.
        """
        if isinstance(self.casebody.data, str):
            return eyecite.clean_text(self.casebody.data, ["html", "inline_whitespace"])
        elif self.casebody.data.opinions:
            return eyecite.clean_text(self.casebody.data.opinions[0].text, ["html", "inline_whitespace"])
        return None

    def extract_citation_contexts(
        self,
        citation: Union[str, CAPCitation, CaseCitation],
        words_before=500,
        words_after=500,
    ) -> str:
        """
        Extracts contexts around citations found in the opinion text.

        Converts the opinion text to plain text if necessary and uses
        `get_citation_context_sents` to find citations and their surrounding text.

        Returns:
            str: A string of context around a citation found in the opinion text.
        """
        plain_text = self.text_clean
        
        normalized_cite: str = normalize_case_cite(citation)

        citation_contexts = get_citation_context(
            plain_text, normalized_cite, words_before, words_after
        )
        if isinstance(citation_contexts, list):
            citation_contexts = " ".join(citation_contexts)

        return citation_contexts

    @property
    def opinion_text(self) -> Optional[str]:
        """
        Extracts the text of the first opinion from the case body data.

        If the case body data is a string, it is assumed to be HTML and is converted to text.
        Otherwise, it directly accesses the text of the first opinion in the case body data.

        Returns:
            Optional[str]: The text of the first opinion if available, otherwise None.
        """
        if isinstance(self.casebody.data, str):
            markdown_from_html_text = html2text(self.casebody.data)
            text = markdown_to_text(markdown_from_html_text)
        if self.casebody.data.opinions[0].is_html:
            markdown_from_html_text = html2text(self.casebody.data.opinions[0].text)
            text = markdown_to_text(markdown_from_html_text) 
        else:
            text = (
                self.casebody.data.opinions[0].text
                if self.casebody.data.opinions
                else None
            )
        return text

    @property
    def opinions(self) -> List[Opinion]:
        """Get all Opinions published with this Decision."""
        return self.casebody.data.opinions if self.casebody else []

    @property
    def majority(self) -> Optional[Opinion]:
        """Get a majority opinion, if any."""
        for opinion in self.opinions:
            if opinion.type == "majority":
                return opinion
        return None


class DecisionWithContext(Decision):
    """Subclass of Decision that includes context for a citation."""
    
    def __init__(self, *args, context: Optional[str] = None, context_citation: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = context
        self.context_citation = context_citation
        
        
def extract_parallel_citation_context(
    forward_decisions: List[Decision], 
    citations: List[str],
    words_before: int = 400,
    words_after: int = 400
    ) -> List[DecisionWithContext]:
    """
    Extracts the context for the first found citation in each Decision object and returns a new list of Decision objects with the context stored in an attribute.
        When multiple citations exist for a given opinion, i.e., are parallel, each version is checked until a match is found.

    Args:
        forward_decisions: A list of Decision objects to search through.
        citations: A list of citation strings to search for in each Decision.

    Returns:
        A list of DecisionWithContext objects, each representing a Decision with an added context attribute.
    """
    updated_decisions = []

    for decision in forward_decisions:
        context_found = False
        extracted_context = None
        
        for citation in citations:
            if context_found:
                break
            
            try:
                context = decision.extract_citation_contexts(
                    citation=citation,
                    words_before=words_before,
                    words_after=words_after,
                )
                
                if context:
                    extracted_context = context
                    context_found = True
            except Exception as e:
                print(f"Error extracting context for citation {citation} in decision: {e}")
        
        # Create a new DecisionWithContext object, copying the original decision and adding the extracted context
        updated_decision = DecisionWithContext(**decision.__dict__, context=extracted_context, context_citation=citation)
        updated_decisions.append(updated_decision)

    return updated_decisions


def filter_decisions_by_forward_citations(decision: Decision, decisions_list: List[Decision]) -> List[Decision]:
    """
    Filters a list of Decision objects to include only those whose IDs match one of the IDs in the given Decision's forward_citation_ids.

    Args:
        decision (Decision): The Decision object whose forward_citation_ids will be used for filtering.
        decisions_list (List[Decision]): A list of Decision objects to be filtered.

    Returns:
        List[Decision]: A list of Decision objects filtered based on the forward_citation_ids of the given Decision.
    """
    filtered_decisions = [d for d in decisions_list if d.id in decision.forward_citation_ids]
    return filtered_decisions


