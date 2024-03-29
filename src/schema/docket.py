from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, constr


class Source(Enum):
    DEFAULT = 0
    RECAP = 1
    SCRAPER = 2
    RECAP_AND_SCRAPER = 3
    COLUMBIA = 4
    COLUMBIA_AND_RECAP = 5
    COLUMBIA_AND_SCRAPER = 6
    COLUMBIA_AND_RECAP_AND_SCRAPER = 7
    IDB = 8
    RECAP_AND_IDB = 9
    SCRAPER_AND_IDB = 10
    RECAP_AND_SCRAPER_AND_IDB = 11
    COLUMBIA_AND_IDB = 12
    COLUMBIA_AND_RECAP_AND_IDB = 13
    COLUMBIA_AND_SCRAPER_AND_IDB = 14
    COLUMBIA_AND_RECAP_AND_SCRAPER_AND_IDB = 15
    HARVARD = 16
    HARVARD_AND_RECAP = 17
    SCRAPER_AND_HARVARD = 18
    RECAP_AND_SCRAPER_AND_HARVARD = 19
    HARVARD_AND_COLUMBIA = 20
    COLUMBIA_AND_RECAP_AND_HARVARD = 21
    COLUMBIA_AND_SCRAPER_AND_HARVARD = 22
    COLUMBIA_AND_RECAP_AND_SCRAPER_AND_HARVARD = 23
    IDB_AND_HARVARD = 24
    RECAP_AND_IDB_AND_HARVARD = 25
    SCRAPER_AND_IDB_AND_HARVARD = 26
    RECAP_AND_SCRAPER_AND_IDB_AND_HARVARD = 27
    COLUMBIA_AND_IDB_AND_HARVARD = 28
    COLUMBIA_AND_RECAP_AND_IDB_AND_HARVARD = 29
    COLUMBIA_AND_SCRAPER_AND_IDB_AND_HARVARD = 30
    COLUMBIA_AND_RECAP_AND_SCRAPER_AND_IDB_AND_HARVARD = 31
    DIRECT_INPUT = 32
    DIRECT_INPUT_AND_HARVARD = 48
    ANON_2020 = 64
    ANON_2020_AND_SCRAPER = 66
    ANON_2020_AND_HARVARD = 80
    ANON_2020_AND_SCRAPER_AND_HARVARD = 82
    
    


class Docket(BaseModel):
    resource_uri: Optional[str] = None
    id: Optional[int] = None
    court: Optional[str] = None
    court_id: Optional[str] = None
    clusters: Optional[List[str]] = None
    audio_files: Optional[List[str]] = None
    assigned_to: Optional[str] = None
    referred_to: Optional[str] = None
    absolute_url: Optional[str] = None
    date_created: Optional[datetime] = Field(
        None, description="The moment when the item was created."
    )
    date_modified: Optional[datetime] = Field(
        None,
        description="The last moment when the item was modified. A value in year 1750 indicates the value is unknown",
    )
    source: Optional[Union[str, Source]] = Field(
        None, description="Contains the source of the Docket."
    )
    appeal_from_str: Optional[str] = Field(
        None,
        description="In appellate cases, this is the lower court or administrative body where this case was originally heard. This field is frequently blank due to it not being populated historically. This field may have values when the appeal_from field does not. That can happen if we are unable to normalize the value in this field.",
    )
    assigned_to_str: Optional[str] = Field(
        None, description="The judge that the case was assigned to, as a string."
    )
    referred_to_str: Optional[str] = Field(
        None, description="The judge that the case was referred to, as a string."
    )
    panel_str: Optional[str] = Field(
        None,
        description="The initials of the judges on the panel that heard this case. This field is similar to the 'judges' field on the cluster, but contains initials instead of full judge names, and applies to the case on the whole instead of only to a specific decision.",
    )
    case_name_short: Optional[str] = Field(
        None,
        description="The abridged name of the case, often a single word, e.g. 'Marsh'",
    )
    case_name: Optional[str] = Field(None, description="The standard name of the case")
    case_name_full: Optional[str] = Field(None, description="The full name of the case")
    slug: Optional[str] = Field(
        None, description="URL that the document should map to (the slug)"
    )
    docket_number: Optional[str] = Field(
        None,
        description="The docket numbers of a case, can be consolidated and quite long. In some instances they are too long to be indexed by postgres and we store the full docket in the correction field on the Opinion Cluster.",
    )
    docket_number_core: Optional[constr(max_length=20)] = Field(
        None,
        description="For federal district court dockets, this is the most distilled docket number available. In this field, the docket number is stripped down to only the year and serial digits, eliminating the office at the beginning, letters in the middle, and the judge at the end. Thus, a docket number like 2:07-cv-34911-MJL becomes simply 0734911. This is the format that is provided by the IDB and is useful for de-duplication types of activities which otherwise get messy. We use a char field here to preserve leading zeros.",
    )
    pacer_case_id: Optional[constr(max_length=100)] = Field(
        None, description="The cased ID provided by PACER."
    )
    tags: Optional[List[str]] = Field(
        None, description="The tags associated with the docket."
    )
    panel: Optional[List[str]] = Field(
        None,
        description="The empaneled judges for the case. Currently an unused field but planned to be used in conjunction with the panel_str field.",
    )
    
