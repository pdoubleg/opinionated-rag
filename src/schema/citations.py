"""Citations used for cross-referencing caselaw."""

from typing import List, Optional, Union

from eyecite import get_citations
from eyecite.models import CaseCitation
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schema.opinion_cluster_model import Type


class CAPCitation(BaseModel):
    """
    A case citation generated with data from Case Access Project or Court Listener.

    Args:
        cite (str): The text making up the citation.
        reporter (Optional[str]): Identifier for the reporter cited to.
        category (Optional[str]): The type of document cited, e.g. "reporters:federal".
        case_ids (List[int]): A list of Case Access Project IDs for the cited case.
        type (Optional[str]): The kind of citation, e.g. "official".
    """
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )
    cite: Union[str, List[str]]
    reporter: Optional[str] = Field(
        None,
        description="CourtListener Citation.reporter",
    )
    category: Optional[str | Type] = Field(
        None,
        description="CAP `category` or Court Listener `type`",
        alias="type",
    )
    case_ids: List[int] = []


    @field_validator('cite', mode='before')
    def convert_cite_to_str(cls, v):
        """Converts cite field to string if it's a list.

        Args:
            v (Union[str, List[str]]): The value of the cite field.

        Returns:
            str: The cite field converted to string if it was a list.
        """
        if isinstance(v, list):
            return ', '.join(v)
        return v
    
    def __str__(self) -> str:
        return f"{self.cite}"
    
    
from typing import List, Optional, Union
from eyecite.models import CaseCitation

class GeneralCitation(CaseCitation):
    """
    A generalized citation class that can handle different types of citations.
    """
    case_ids: List[int] = []
    category: Optional[str] = None
    type: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_ids = kwargs.get('case_ids', [])
        self.category = kwargs.get('category', None)
        self.type = kwargs.get('type', None)

    # Example of overriding a method from CaseCitation
    def corrected_citation(self) -> str:
        # Custom logic here
        return super().corrected_citation()


    
def normalize_case_cite(cite: Union[str, CaseCitation, CAPCitation]) -> str:
    """Normalize a citation object or string."""
    if isinstance(cite, CAPCitation):
        return cite.cite
    if isinstance(cite, str):
        possible_cites = list(get_citations(cite))
        bad_cites = []
        for possible in possible_cites:
            if isinstance(possible, CaseCitation):
                return possible.corrected_citation()
            bad_cites.append(possible)
        error_msg = f"Could not locate a CaseCitation in the text {cite}."
        for bad_cite in bad_cites:
            error_msg += (
                f" {bad_cite} was type {bad_cite.__class__.__name__}, not CaseCitation."
            )

        raise ValueError(error_msg)
    return cite.corrected_citation()

