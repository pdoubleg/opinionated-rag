from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Union

from pydantic import AnyUrl, BaseModel, Field, constr

class OpinionType(Enum.enum):
    COMBINED = "010combined"
    UNANIMOUS = "015unamimous"
    LEAD = "020lead"
    PLURALITY = "025plurality"
    CONCURRENCE = "030concurrence"
    CONCUR_IN_PART = "035concurrenceinpart"
    DISSENT = "040dissent"
    ADDENDUM = "050addendum"
    REMITTUR = "060remittitur"
    REHEARING = "070rehearing"
    ON_THE_MERITS = "080onthemerits"
    ON_MOTION_TO_STRIKE = "090onmotiontostrike"


class OpinionsCited(BaseModel):
    resource_uri: Optional[AnyUrl] = None
    id: Optional[str] = None
    citing_opinion: Optional[str] = None
    cited_opinion: Optional[str] = None
    depth: Optional[int] = Field(
        None,
        description="The number of times the cited opinion was cited in the citing opinion",
    )


class Opinion(BaseModel):
    resource_uri: Optional[AnyUrl] = None
    id: Optional[int] = None
    absolute_url: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster: Union[int, str, None] = None
    author: Optional[str] = None
    joined_by: Optional[List[str]] = None
    date_created: Optional[datetime] = Field(
        None, description="The moment when the item was created."
    )
    date_modified: Optional[datetime] = Field(
        None,
        description="The last moment when the item was modified. A value in year 1750 indicates the value is unknown",
    )
    author_str: Optional[str] = Field(
        None,
        description="The primary author of this opinion, as a simple text string. This field is used when normalized judges cannot be placed into the author field.",
    )
    joined_by_str: Optional[str] = Field(
        None,
        description="Other judges that joined the primary author in this opinion str",
    )
    sha1: Optional[constr(max_length=40)] = Field(
        None,
        description="unique ID for the document, as generated via SHA1 of the binary file or text data",
    )
    page_count: Optional[int] = Field(
        None, description="The number of pages in the document, if known"
    )
    download_url: Optional[AnyUrl] = Field(
        None,
        description="The URL where the item was originally scraped. Note that these URLs may often be dead due to the court or the bulk provider changing their website. We keep the original link here given that it often contains valuable metadata.",
    )
    local_path: Optional[bytes] = Field(
        None,
        description="The location in AWS S3 where the original opinion file is stored. Note that the field name is historical, from before when we used S3. To find the location in S3, concatenate https://storage.courtlistener.com/ and the value of this field.",
    )
    plain_text: Optional[str] = Field(
        None,
        description="Plain text of the document after extraction using pdftotext, wpd2txt, etc.",
    )
    html: Optional[str] = Field(
        None, description="HTML of the document, if available in the original"
    )
    html_lawbox: Optional[str] = Field(None, description="HTML of Lawbox documents")
    html_columbia: Optional[str] = Field(None, description="HTML of Columbia archive")
    html_anon_2020: Optional[str] = Field(
        None, description="HTML of 2020 anonymous archive"
    )
    xml_harvard: Optional[str] = Field(
        None, description="XML of Harvard CaseLaw Access Project opinion"
    )
    html_with_citations: Optional[str] = Field(
        None,
        description="HTML of the document with citation links and other post-processed markup added",
    )
    extracted_by_ocr: Optional[bool] = Field(
        None, description="Whether OCR was used to get this document content"
    )
    opinions_cited: Union[List[str], OpinionsCited] = None
    
