from datetime import datetime, timezone
import hashlib
from tqdm.asyncio import tqdm
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Literal, Optional
import uuid
import instructor
import openai
from pydantic import BaseModel, ConfigDict, Field, validator
from tenacity import Retrying, AsyncRetrying, stop_after_attempt, wait_fixed

from src.schema.decisions import Decision, DecisionWithContext

client = instructor.patch(openai.OpenAI())


class APICommunicationError(Exception):
    
    pass




class CitationAnalysis(BaseModel):
    """Information about a legal citation"""

    citation: str = Field(
        ...,
        description="The Citation specified by the user.",
    )
    legal_question: str = Field(
        ...,
        description="A concise and well-structured legal question. For example: 'Plaintiff slipped and fell in the hotel lobby. Is the hotel liable?'",
    )
    rule: str = Field(
        ...,
        description="A concise legal ruling, decision, or authority. For example: 'If a hotel knows its floors are wet, it has a duty to take reasonable steps to avoid such injury.'",
    )
    application: str = Field(
        ...,
        description="Application, or potential application of the rule. For example: 'The hotel acted negligently'.",
    )
    citation_reference: str = Field(
        ...,
        description="A concise explanation of the way in which the citation was referenced or used in the context.",
    )


def analyze_citation(
    cited_opinion_bluebook_citation: str, excerpt: str
) -> CitationAnalysis:
    return client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_model=CitationAnalysis,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        ),
        messages=[
            {
                "role": "system",
                "content": "Your role is to extract information about a legal citation using the context, which is an excerpt from a subsequent legal proceeding that referenced the citation of interest.",
            },
            {
                "role": "user",
                "content": f"Your task focuses on citation: **{cited_opinion_bluebook_citation}**",
            },
            {   "role": "user", "content": f"Here is the context: {excerpt}"},
        ],
    )
    
    

async def analyze_citations(citation, context):
    client = instructor.patch(openai.AsyncOpenAI())
    return await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_model=CitationAnalysis,
        max_retries=AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
    ),
        messages=[
            {
                "role": "system",
                "content": "Your role is to extract information about a legal citation using the context, which is an excerpt from a subsequent legal proceeding that referenced the citation of interest.",
            },
            {"role": "user", "content": f"Your task focuses on citation: **{citation}**"},
            {"role": "user", "content": f"Here is the context: {context}"}
        ]
    )
    
    
async def process_citations_with_progress(decisions_context):
    results = []
    for forward_case in tqdm(decisions_context, total=len(decisions_context)):
        result = await analyze_citations(forward_case.context_citation, forward_case.context)
        results.append(result)
    return results
    
    
class DocMetaData(BaseModel):
    """Metadata for a legal document."""

    source: str = "context"
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )
    is_chunk: bool = False  # if it is a chunk, don't split
    id: str = ""  # unique id for the document
    window_ids: List[str] = []  # for RAG: ids of chunks around this one

    def dict_bool_int(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Convert bool fields to int, to appease downstream libraries.
        """
        original_dict = super().model_dump(*args, **kwargs)

        for key, value in original_dict.items():
            if isinstance(value, bool):
                original_dict[key] = 1 * value

        return original_dict


class LegalDataSource(ABC):
    """
    Abstract base class for legal data sources.
    """

    @abstractmethod
    def fetch_case(self, case_id: int) -> Dict[str, Any] | Decision | Any:
        """
        Fetches a case by its ID.
        """
        pass

    @abstractmethod
    def get_forward_citation_ids(self, case_id: int) -> List[int]:
        """
        Fetches 'forward citations' case IDs, i.e., cases that cite a given case.
        """
        pass

    @abstractmethod
    def get_bluebook_citation(self, case_id: int) -> str:
        """
        Generates a Bluebook citation for a given case ID.
        """
        pass
    


class LegalDocument(BaseModel):
    """
    Represents a legal document, capable of fetching and storing data related to a legal case.

    Attributes:
        case_id (str): The unique identifier for the case.
        data_source (LegalDataSource): The data source from which the case data is fetched.
        url (Optional[str]): The URL of the document, if available.
        content (Optional[str]): The content of the document.
        ...
    """

    def __init__(self, case_id: int, data_source: LegalDataSource) -> None:
        """
        Initializes a new instance of the LegalDocument class.

        Args:
            case_id (int): The unique identifier for the case.
            data_source (LegalDataSource): The data source from which the case data is fetched.
        """
        self.case_id = case_id
        self.data_source = data_source
        self._data: Optional[Dict[str, Any]] = None
        
    case_id: str
    data_source: LegalDataSource
    name: Optional[str] = None
    short_name: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None
    doc_class: Optional[str] = None
    citations: Optional[List[str]] = Field(default_factory=list)
    jurisdiction: Optional[str]
    court: Optional[str] = None
    publication_date: Optional[datetime]
    date_fetched: datetime.date = datetime.now()
    source_id: Optional[str]
    metadata: Optional[DocMetaData] = None
    
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )
    
    @staticmethod
    def hash_id(doc: str) -> str:
        doc_utf8 = str(doc).encode("utf-8")
        sha256_hash = hashlib.sha256()
        sha256_hash.update(doc_utf8)
        hash_hex = sha256_hash.hexdigest()
        hash_uuid = uuid.UUID(hash_hex[:32])
        return str(hash_uuid)

    def _unique_hash_id(self) -> str:
        return self.hash_id(str(self))

    def id(self) -> str:
        if (
            hasattr(self.metadata, "id")
            and self.metadata.id is not None
            and self.metadata.id != ""
        ):
            return self.metadata.id
        else:
            return self._unique_hash_id()

    def __str__(self) -> str:
        return f"{self.metadata.model_dump_json()} {self.content}"

    @property
    def data(self) -> Dict[str, Any]:
        if not self._data:
            self._data = self.data_source.fetch_case(self.case_id)
        return self._data
    

    def get_bluebook_citation(self) -> str:
        return self.data_source.get_bluebook_citation(self.case_id)

    def fetch_forward_citations(self) -> List[Dict[str, Any]]:
        return self.data_source.fetch_forward_citations(self.case_id)
