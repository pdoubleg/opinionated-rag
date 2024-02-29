# generated by datamodel-codegen:
#   filename:  opinion_cluster.json
#   timestamp: 2024-02-27T03:28:47+00:00

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, conint, constr, validator


class Type(Enum):
    FEDERAL = 1
    STATE = 2
    STATE_REGIONAL = 3
    SPECIALTY = 4
    SCOTUS_EARLY = 5
    LEXIS = 6
    WEST = 7
    NEUTRAL = 8


class Citation(BaseModel):
    volume: int = Field(
        ..., 
        description="The volume of the reporter"
    )
    reporter: str = Field(
        ..., 
        description="The abbreviation for the reporter")
    page: str = Field(
        ...,
        description="The 'page' of the citation in the reporter. Unfortunately, this is not an integer, but is a string-type because several jurisdictions do funny things with the so-called 'page'. For example, we have seen Roman numerals in Nebraska, 13301-M in Connecticut, and 144M in Montana.",
    )
    type: Type = Field(
        ..., 
        description="The type of citation that this is.")
    
    def __str__(self) -> str:
        return f"{self.volume} {self.reporter} {self.page}".format(**self.__dict__)

    @property
    def sort_score(self) -> float:
        """
        Calculates a numeric score for sorting citations according to BlueBook ordering.

        This property is intended to be used as a key for sorting methods like `sort` or `sorted`,
        providing a numeric score indicating the citation's order in a list according to BlueBook rules.

        Returns:
            float: A score representing the citation's order.

        Example:
            cs = Citation.objects.filter(cluster_id=222)
            cs = sorted(cs, key=lambda c: c.sort_score)
            This would sort the Citation items by their priority.
        """
        if self.type == Type.NEUTRAL:
            return 0
        if self.type == Type.FEDERAL:
            if self.reporter == "U.S.":
                return 1.1
            elif self.reporter == "S. Ct.":
                return 1.2
            elif "L. Ed." in self.reporter:
                return 1.3
            else:
                return 1.4
        elif self.type == Type.SCOTUS_EARLY:
            return 2
        elif self.type == Type.SPECIALTY:
            return 3
        elif self.type == Type.STATE_REGIONAL:
            return 4
        elif self.type == Type.STATE:
            return 5
        elif self.type == Type.WEST:
            return 6
        elif self.type == Type.LEXIS:
            return 7
        else:
            return 8


class ScdbDecisionDirection(Enum):
    CONSERVATIVE = "Conservative"
    LIBERAL = "Liberal"
    UNSPECIFIABLE = "Unspecifiable"

    @classmethod
    def from_int(cls, value: int):
        """
        Converts an integer to the corresponding enum member.

        Args:
            value (int): The integer value to convert.

        Returns:
            ScdbDecisionDirection: The corresponding enum member.
        """
        mapping = {
            1: cls.CONSERVATIVE,
            2: cls.LIBERAL,
            3: cls.UNSPECIFIABLE,
        }
        return mapping.get(value, None)

class Source(Enum):
    C = "C"
    R = "R"
    CR = "CR"
    L = "L"
    LC = "LC"
    LR = "LR"
    LCR = "LCR"
    M = "M"
    A = "A"
    H = "H"
    Z = "Z"
    ZA = "ZA"
    ZD = "ZD"
    ZC = "ZC"
    ZH = "ZH"
    ZLC = "ZLC"
    ZLR = "ZLR"
    ZLCR = "ZLCR"
    ZR = "ZR"
    ZCR = "ZCR"
    ZL = "ZL"
    ZM = "ZM"
    ZQ = "ZQ"
    U = "U"
    CU = "CU"
    D = "D"
    Q = "Q"
    QU = "QU"
    CRU = "CRU"
    DU = "DU"
    LU = "LU"
    LCU = "LCU"
    LRU = "LRU"
    MU = "MU"
    RU = "RU"
    ZU = "ZU"
    ZLU = "ZLU"
    ZDU = "ZDU"
    ZLRU = "ZLRU"
    ZLCRU = "ZLCRU"
    ZCU = "ZCU"
    ZMU = "ZMU"
    ZRU = "ZRU"
    ZLCU = "ZLCU"
    
class SOURCES:
    COURT_WEBSITE = "C"
    PUBLIC_RESOURCE = "R"
    COURT_M_RESOURCE = "CR"
    LAWBOX = "L"
    LAWBOX_M_COURT = "LC"
    LAWBOX_M_RESOURCE = "LR"
    LAWBOX_M_COURT_RESOURCE = "LCR"
    MANUAL_INPUT = "M"
    INTERNET_ARCHIVE = "A"
    BRAD_HEATH_ARCHIVE = "H"
    COLUMBIA_ARCHIVE = "Z"
    HARVARD_CASELAW = "U"
    COURT_M_HARVARD = "CU"
    DIRECT_COURT_INPUT = "D"
    ANON_2020 = "Q"
    ANON_2020_M_HARVARD = "QU"
    COURT_M_RESOURCE_M_HARVARD = "CRU"
    DIRECT_COURT_INPUT_M_HARVARD = "DU"
    LAWBOX_M_HARVARD = "LU"
    LAWBOX_M_COURT_M_HARVARD = "LCU"
    LAWBOX_M_RESOURCE_M_HARVARD = "LRU"
    LAWBOX_M_COURT_RESOURCE_M_HARVARD = "LCRU"
    MANUAL_INPUT_M_HARVARD = "MU"
    PUBLIC_RESOURCE_M_HARVARD = "RU"
    COLUMBIA_M_INTERNET_ARCHIVE = "ZA"
    COLUMBIA_M_DIRECT_COURT_INPUT = "ZD"
    COLUMBIA_M_COURT = "ZC"
    COLUMBIA_M_BRAD_HEATH_ARCHIVE = "ZH"
    COLUMBIA_M_LAWBOX_COURT = "ZLC"
    COLUMBIA_M_LAWBOX_RESOURCE = "ZLR"
    COLUMBIA_M_LAWBOX_COURT_RESOURCE = "ZLCR"
    COLUMBIA_M_RESOURCE = "ZR"
    COLUMBIA_M_COURT_RESOURCE = "ZCR"
    COLUMBIA_M_LAWBOX = "ZL"
    COLUMBIA_M_MANUAL = "ZM"
    COLUMBIA_M_ANON_2020 = "ZQ"
    COLUMBIA_ARCHIVE_M_HARVARD = "ZU"
    COLUMBIA_M_LAWBOX_M_HARVARD = "ZLU"
    COLUMBIA_M_DIRECT_COURT_INPUT_M_HARVARD = "ZDU"
    COLUMBIA_M_LAWBOX_M_RESOURCE_M_HARVARD = "ZLRU"
    COLUMBIA_M_LAWBOX_M_COURT_RESOURCE_M_HARVARD = "ZLCRU"
    COLUMBIA_M_COURT_M_HARVARD = "ZCU"
    COLUMBIA_M_MANUAL_INPUT_M_HARVARD = "ZMU"
    COLUMBIA_M_PUBLIC_RESOURCE_M_HARVARD = "ZRU"
    COLUMBIA_M_LAWBOX_M_COURT_M_HARVARD = "ZLCU"
    NAMES = (
        (COURT_WEBSITE, "court website"),
        (PUBLIC_RESOURCE, "public.resource.org"),
        (COURT_M_RESOURCE, "court website merged with resource.org"),
        (LAWBOX, "lawbox"),
        (LAWBOX_M_COURT, "lawbox merged with court"),
        (LAWBOX_M_RESOURCE, "lawbox merged with resource.org"),
        (LAWBOX_M_COURT_RESOURCE, "lawbox merged with court and resource.org"),
        (MANUAL_INPUT, "manual input"),
        (INTERNET_ARCHIVE, "internet archive"),
        (BRAD_HEATH_ARCHIVE, "brad heath archive"),
        (COLUMBIA_ARCHIVE, "columbia archive"),
        (COLUMBIA_M_INTERNET_ARCHIVE, "columbia merged with internet archive"),
        (
            COLUMBIA_M_DIRECT_COURT_INPUT,
            "columbia merged with direct court input",
        ),
        (COLUMBIA_M_COURT, "columbia merged with court"),
        (
            COLUMBIA_M_BRAD_HEATH_ARCHIVE,
            "columbia merged with brad heath archive",
        ),
        (COLUMBIA_M_LAWBOX_COURT, "columbia merged with lawbox and court"),
        (
            COLUMBIA_M_LAWBOX_RESOURCE,
            "columbia merged with lawbox and resource.org",
        ),
        (
            COLUMBIA_M_LAWBOX_COURT_RESOURCE,
            "columbia merged with lawbox, court, and resource.org",
        ),
        (COLUMBIA_M_RESOURCE, "columbia merged with resource.org"),
        (
            COLUMBIA_M_COURT_RESOURCE,
            "columbia merged with court and resource.org",
        ),
        (COLUMBIA_M_LAWBOX, "columbia merged with lawbox"),
        (COLUMBIA_M_MANUAL, "columbia merged with manual input"),
        (COLUMBIA_M_ANON_2020, "columbia merged with 2020 anonymous database"),
        (
            HARVARD_CASELAW,
            "Harvard, Library Innovation Lab Case Law Access Project",
        ),
        (COURT_M_HARVARD, "court website merged with Harvard"),
        (DIRECT_COURT_INPUT, "direct court input"),
        (ANON_2020, "2020 anonymous database"),
        (ANON_2020_M_HARVARD, "2020 anonymous database merged with Harvard"),
        (COURT_M_HARVARD, "court website merged with Harvard"),
        (
            COURT_M_RESOURCE_M_HARVARD,
            "court website merged with public.resource.org and Harvard",
        ),
        (
            DIRECT_COURT_INPUT_M_HARVARD,
            "direct court input merged with Harvard",
        ),
        (LAWBOX_M_HARVARD, "lawbox merged with Harvard"),
        (
            LAWBOX_M_COURT_M_HARVARD,
            "Lawbox merged with court website and Harvard",
        ),
        (
            LAWBOX_M_RESOURCE_M_HARVARD,
            "Lawbox merged with public.resource.org and with Harvard",
        ),
        (MANUAL_INPUT_M_HARVARD, "Manual input merged with Harvard"),
        (PUBLIC_RESOURCE_M_HARVARD, "public.resource.org merged with Harvard"),
        (COLUMBIA_ARCHIVE_M_HARVARD, "columbia archive merged with Harvard"),
        (
            COLUMBIA_M_LAWBOX_M_HARVARD,
            "columbia archive merged with Lawbox and Harvard",
        ),
        (
            COLUMBIA_M_DIRECT_COURT_INPUT_M_HARVARD,
            "columbia archive merged with direct court input and Harvard",
        ),
        (
            COLUMBIA_M_LAWBOX_M_RESOURCE_M_HARVARD,
            "columbia archive merged with lawbox, public.resource.org and Harvard",
        ),
        (
            COLUMBIA_M_LAWBOX_M_COURT_RESOURCE_M_HARVARD,
            "columbia archive merged with lawbox, court website, public.resource.org and Harvard",
        ),
        (
            COLUMBIA_M_COURT_M_HARVARD,
            "columbia archive merged with court website and Harvard",
        ),
        (
            COLUMBIA_M_MANUAL_INPUT_M_HARVARD,
            "columbia archive merged with manual input and Harvard",
        ),
        (
            COLUMBIA_M_PUBLIC_RESOURCE_M_HARVARD,
            "columbia archive merged with public.resource.org and Harvard",
        ),
        (
            COLUMBIA_M_LAWBOX_M_COURT_M_HARVARD,
            "columbia archive merged with lawbox, court website and Harvard",
        ),
    )


class PrecedentialStatus(Enum):
    Published = "Published"
    Unpublished = "Unpublished"
    Errata = "Errata"
    Separate = "Separate"
    In_chambers = "In-chambers"
    Relating_to = "Relating-to"
    Unknown = "Unknown"


class OpinionCluster(BaseModel):
    resource_uri: Optional[str] = None
    id: Optional[int | str] = None
    absolute_url: Optional[str] = None
    panel: Optional[List[str]] = None
    non_participating_judges: Optional[List[str]] = None
    docket_id: Optional[int | str] = None
    docket: Optional[str]
    sub_opinions: Optional[List[str]]
    citations: List[Citation]
    date_created: Optional[datetime] = Field(
        None, description="The moment when the item was created."
    )
    date_modified: Optional[datetime] = Field(
        None,
        description="The last moment when the item was modified. A value in year 1750 indicates the value is unknown",
    )
    judges: Optional[str] = Field(
        None,
        description="The judges that participated in the opinion as a simple text string. This field is used when normalized judges cannot be placed into the panel field.",
    )
    date_filed: Optional[date] = Field(
        ..., 
        description="The date the cluster of opinions was filed by the court"
    )
    slug: Optional[str] = Field(
        None, 
        description="URL that the document should map to (the slug)"
    )
    case_name_short: Optional[str] = Field(
        None,
        description="The abridged name of the case, often a single word, e.g. 'Marsh'",
    )
    case_name: Optional[str] = Field(
        None, 
        description="The shortened name of the case"
        )
    case_name_full: Optional[str] = Field(
        None, 
        description="The full name of the case"
                                          )
    scdb_id: Optional[str] = Field(
        None, 
        description="The ID of the item in the Supreme Court Database"
    )
    scdb_decision_direction: Optional[ScdbDecisionDirection] = Field(
        None,
        description='the ideological "direction" of a decision in the Supreme Court database. More details at: http://scdb.wustl.edu/documentation.php?var=decisionDirection',
    )
    scdb_votes_majority: Optional[int] = Field(
        None,
        description="the number of justices voting in the majority in a Supreme Court decision. More details at: http://scdb.wustl.edu/documentation.php?var=majVotes",
    )
    scdb_votes_minority: Optional[int] = Field(
        None,
        description="the number of justices voting in the minority in a Supreme Court decision. More details at: http://scdb.wustl.edu/documentation.php?var=minVotes",
    )
    source: Optional[int | str | Source] = Field(
        None,
        description="the source of the cluster, one of: C (court website), R (public.resource.org), CR (court website merged with resource.org), L (lawbox), LC (lawbox merged with court), LR (lawbox merged with resource.org), LCR (lawbox merged with court and resource.org), M (manual input), A (internet archive), H (brad heath archive), Z (columbia archive), ZA (columbia merged with internet archive), ZD (columbia merged with direct court input), ZC (columbia merged with court), ZH (columbia merged with brad heath archive), ZLC (columbia merged with lawbox and court), ZLR (columbia merged with lawbox and resource.org), ZLCR (columbia merged with lawbox, court, and resource.org), ZR (columbia merged with resource.org), ZCR (columbia merged with court and resource.org), ZL (columbia merged with lawbox), ZM (columbia merged with manual input), ZQ (columbia merged with 2020 anonymous database), U (Harvard, Library Innovation Lab Case Law Access Project), CU (court website merged with Harvard), D (direct court input), Q (2020 anonymous database), QU (2020 anonymous database merged with Harvard), CU (court website merged with Harvard), CRU (court website merged with public.resource.org and Harvard), DU (direct court input merged with Harvard), LU (lawbox merged with Harvard), LCU (Lawbox merged with court website and Harvard), LRU (Lawbox merged with public.resource.org and with Harvard), MU (Manual input merged with Harvard), RU (public.resource.org merged with Harvard), ZU (columbia archive merged with Harvard), ZLU (columbia archive merged with Lawbox and Harvard), ZDU (columbia archive merged with direct court input and Harvard), ZLRU (columbia archive merged with lawbox, public.resource.org and Harvard), ZLCRU (columbia archive merged with lawbox, court website, public.resource.org and Harvard), ZCU (columbia archive merged with court website and Harvard), ZMU (columbia archive merged with manual input and Harvard), ZRU (columbia archive merged with public.resource.org and Harvard), ZLCU (columbia archive merged with lawbox, court website and Harvard)",
    )
    procedural_history: Optional[str] = Field(
        None, description="The history of the case as it jumped from court to court"
    )
    attorneys: Optional[str] = Field(
        None, description="The attorneys that argued the case, as free text"
    )
    nature_of_suit: Optional[str] = Field(
        None,
        description="The nature of the suit. For the moment can be codes or laws or whatever",
    )
    posture: Optional[str] = Field(
        None, description="The procedural posture of the case."
    )
    syllabus: Optional[str] = Field(
        None,
        description="A summary of the issues presented in the case and the outcome.",
    )
    headnotes: Optional[str] = Field(
        None,
        description="Headnotes are summary descriptions of the legal issues discussed by the court in the particular case. They appear at the beginning of each case just after the summary and disposition. They are short paragraphs with a heading in bold face type. From Wikipedia - A headnote is a brief summary of a particular point of law that is added to the text of a courtdecision to aid readers in locating discussion of a legalissue in an opinion. As the term implies, headnotes appearat the beginning of the published opinion. Frequently, headnotes are value-added components appended to decisions by the publisher who compiles the decisions of a court for resale. As handed down by the court, a decision or written opinion does not contain headnotes. These are added later by an editor not connected to the court, but who instead works for a legal publishing house.",
    )
    summary: Optional[str] = Field(
        None,
        description="A summary of what happened in the case. Appears at the beginning of the case just after the title of the case and court information.",
    )
    disposition: Optional[str] = Field(
        None,
        description="Description of the procedural outcome of the case, e.g. Reversed, dismissed etc. Generally a short paragraph that appears just after the summary or synopsis",
    )
    history: Optional[str] = Field(
        None,
        description="History of the case (similar to the summary, but focused on past events related to this case). Appears at the beginning of the case just after the title of the case and court information",
    )
    other_dates: Optional[str] = Field(
        None,
        description="Other date(s) as specified in the text (case header). This may include follow-up dates.",
    )
    cross_reference: Optional[str] = Field(
        None,
        description="Cross-reference citation (often to a past or future similar case). It does NOT identify this case.",
    )
    correction: Optional[str] = Field(
        None,
        description="Publisher's correction to the case text. Example: Replace last paragraph on page 476 with this text: blah blah blah. This is basically an unstructured text that can be used to manually correct case content according to publisher's instructions. No footnotes is expected within it.",
    )
    citation_count: Optional[int] = Field(
        None, 
        description="The number of times this document is cited by other opinion"
    )
    precedential_status: Optional[PrecedentialStatus] = Field(
        None,
        description="The precedential status of document, one of: Published, Unpublished, Errata, Separate, In-chambers, Relating-to, Unknown",
    )
    date_blocked: Optional[date] = Field(
        None,
        description="The date that this opinion was blocked from indexing by search engines",
    )
    blocked: Optional[bool] = Field(
        None,
        description="Whether a document should be blocked from indexing by search engines",
    )
    filepath_json_harvard: Optional[bytes] = Field(
        None,
        description="Path to local storage of JSON collected from Harvard Case Law project containing available metadata, opinion and opinion cluster.",
    )
    arguments: Optional[str] = Field(
        None,
        description="The attorney(s) and legal arguments presented as HTML text. This is primarily seen in older opinions and can contain case law cited and arguments presented to the judges.",
    )
    headmatter: Optional[str] = Field(
        None,
        description="Headmatter is the content before an opinion in the Harvard CaseLaw import. This consists of summaries, headnotes, attorneys etc for the opinion.",
    )
    
    @validator('scdb_decision_direction', pre=True, always=True)
    def convert_int_to_enum(cls, v):
        """
        Pydantic validator to convert integers to ScdbDecisionDirection enum members.

        Args:
            v: The value to validate and convert.

        Returns:
            The corresponding ScdbDecisionDirection enum member if v is an integer; otherwise, returns v.
        """
        if isinstance(v, int):
            return ScdbDecisionDirection.from_int(v)
        return v
    
    @property
    def get_source_description(self) -> Optional[str]:
        """
        Returns the description of the current source enum value.

        Returns:
            Optional[str]: The description of the selected source, or None if no source is selected.
        """
        if self.source is None:
            return None
        # Find the description in SOURCES.NAMES using the current source value
        for code, description in SOURCES.NAMES:
            if code == self.source.value:
                return description
        return None