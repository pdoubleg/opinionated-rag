from __future__ import annotations

from datetime import datetime
from enum import Enum
import re
from typing import List, Optional

from pydantic import AnyUrl, BaseModel, Field, HttpUrl, conint, constr, model_validator

COURTLISTENER_WEB_URL = "https://www.courtlistener.com"


class Source(Enum):
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
    
    
class AudioResource(BaseModel):
    resource_uri: HttpUrl
    id: int
    frontend_url: HttpUrl
    absolute_url: str
    panel: List[str] = []
    docket: HttpUrl
    date_created: datetime
    date_modified: datetime | None = None
    source: Source | None = None
    case_name_short: str
    case_name: str
    case_name_full: Optional[str] = ""
    judges: Optional[str] = ""
    sha1: str | None = None
    download_url: HttpUrl | None = None
    local_path_mp3: HttpUrl | None = None
    local_path_original_file: str | None = None
    filepath_ia: HttpUrl | None = None
    duration: int | None = None
    processing_complete: bool | None = None
    date_blocked: Optional[datetime] = None
    blocked: bool | None = None
    stt_status: conint(ge=0)
    stt_google_response: Optional[str] = ""
    
    @model_validator(mode="before")
    def generate_frontend_url(cls, values):
        absolute_url = values.get('absolute_url')
        if absolute_url:
            base_url = COURTLISTENER_WEB_URL
            values['frontend_url'] = f"{base_url}{absolute_url}"
        return values
    
    @model_validator(mode="before")
    def generate_mp3_url(cls, values):
        local_path_mp3 = values.get('local_path_mp3')
        if local_path_mp3:
            base_url = "https://www.courtlistener.com/"
            values['local_path_mp3'] = f"{base_url}{local_path_mp3}"
        return values
    
    @property
    def whisper_cost(self):
        audio_duration_in_minutes = self.duration / 60
        return round(audio_duration_in_minutes, 2) * 0.006
    
    def __str__(self) -> str:
        """
        Generates a string representation of the object, combining various attributes.

        Returns:
            str: A string representation of the object.
        """
        text_parts = []
        case_id = f"Case ID: {self.docket_num}"
        text_parts.append(case_id)
        source = self.get_source_description
        source_str = f"Source: [{source}]({self.download_url})"
        text_parts.append(source_str)
        name_short = str(self.case_name)
        text_parts.append(name_short)
        date_argued_str = self.serialize_date(self.date_argued)
        date = f"Date Argued: {date_argued_str[:10]}"
        text_parts.append(date)
        court = f"Court: {str(self.court)}"
        text_parts.append(court)
        # Clean snippet by replacing all sequences of multiple line breaks with a single line break
        cleaned_snippet = re.sub(r'\n+', '\n', self.snippet).strip()
        text_parts.append(cleaned_snippet)
        return "\n\n".join(text_parts)


class Audio(BaseModel):
    resource_uri: Optional[AnyUrl] = None
    id: Optional[str] = None
    absolute_url: Optional[str] = None
    panel: Optional[List[str]] = None
    docket: str
    date_created: Optional[datetime] = Field(
        None, description="The moment when the item was created."
    )
    date_modified: Optional[datetime] = Field(
        None,
        description="The last moment when the item was modified. A value in year 1750 indicates the value is unknown",
    )
    source: Optional[Source] = Field(
        None,
        description="the source of the audio file, one of: C (court website), R (public.resource.org), CR (court website merged with resource.org), L (lawbox), LC (lawbox merged with court), LR (lawbox merged with resource.org), LCR (lawbox merged with court and resource.org), M (manual input), A (internet archive), H (brad heath archive), Z (columbia archive), ZA (columbia merged with internet archive), ZD (columbia merged with direct court input), ZC (columbia merged with court), ZH (columbia merged with brad heath archive), ZLC (columbia merged with lawbox and court), ZLR (columbia merged with lawbox and resource.org), ZLCR (columbia merged with lawbox, court, and resource.org), ZR (columbia merged with resource.org), ZCR (columbia merged with court and resource.org), ZL (columbia merged with lawbox), ZM (columbia merged with manual input), ZQ (columbia merged with 2020 anonymous database), U (Harvard, Library Innovation Lab Case Law Access Project), CU (court website merged with Harvard), D (direct court input), Q (2020 anonymous database), QU (2020 anonymous database merged with Harvard), CU (court website merged with Harvard), CRU (court website merged with public.resource.org and Harvard), DU (direct court input merged with Harvard), LU (lawbox merged with Harvard), LCU (Lawbox merged with court website and Harvard), LRU (Lawbox merged with public.resource.org and with Harvard), MU (Manual input merged with Harvard), RU (public.resource.org merged with Harvard), ZU (columbia archive merged with Harvard), ZLU (columbia archive merged with Lawbox and Harvard), ZDU (columbia archive merged with direct court input and Harvard), ZLRU (columbia archive merged with lawbox, public.resource.org and Harvard), ZLCRU (columbia archive merged with lawbox, court website, public.resource.org and Harvard), ZCU (columbia archive merged with court website and Harvard), ZMU (columbia archive merged with manual input and Harvard), ZRU (columbia archive merged with public.resource.org and Harvard), ZLCU (columbia archive merged with lawbox, court website and Harvard)",
    )
    case_name_short: Optional[str] = Field(
        None,
        description="The abridged name of the case, often a single word, e.g. 'Marsh'",
    )
    case_name: Optional[str] = Field(None, description="The full name of the case")
    case_name_full: Optional[str] = Field(None, description="The full name of the case")
    judges: Optional[str] = Field(
        None,
        description="The judges that heard the oral arguments as a simple text string. This field is used when normalized judges cannot be placed into the panel field.",
    )
    sha1: constr(max_length=40) = Field(
        ...,
        description="unique ID for the document, as generated via SHA1 of the binary file or text data",
    )
    download_url: AnyUrl = Field(
        ...,
        description="The URL on the court website where the document was originally scraped",
    )
    local_path_mp3: Optional[bytes] = Field(
        None,
        description="The location in AWS S3 where our enhanced copy of the original audio file is stored. Note that the field name is historical, from before when we used S3. To find the location in S3, concatenate https://storage.courtlistener.com/ and the value of this field.",
    )
    duration: Optional[int] = Field(
        None, description="the length of the item, in seconds"
    )
