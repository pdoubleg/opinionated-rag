import hashlib
import textwrap
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
import datetime as dt
import time

from pydantic import BaseModel, ConfigDict, Field, field_validator
from lancedb.pydantic import LanceModel, Vector

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

Number = Union[int, float]
Embedding = List[Number]
Embeddings = List[Embedding]
EmbeddingFunction = Callable[[List[str]], Embeddings]


class Ontology(BaseModel):
    labels: List[str | Dict]
    relationships: List[str]

    def dump(self):
        if len(self.relationships) == 0:
            return self.model_dump(exclude=["relationships"])
        else:
            return self.model_dump()


class Node(BaseModel):
    label: str
    name: str

class Edge(BaseModel):
    node_1: Node
    node_2: Node
    relationship: str
    metadata: dict = {}
    order: int | None = None
 
    

@dataclass
class TurboTool:
    name: str
    config: dict
    function: Callable
    
@dataclass
class Chat:
    from_name: str
    to_name: str
    message: str
    created: int = field(default_factory=time.time)


class SEARCH_TYPES:
    OPINION = "o"
    RECAP = "r"
    DOCKETS = "d"
    ORAL_ARGUMENT = "oa"
    PEOPLE = "p"
    PARENTHETICAL = "pa"
    NAMES = (
        (OPINION, "Opinions"),
        (RECAP, "RECAP"),
        (DOCKETS, "RECAP Dockets"),
        (ORAL_ARGUMENT, "Oral Arguments"),
        (PEOPLE, "People"),
        (PARENTHETICAL, "Parenthetical"),
    )
    ALL_TYPES = [OPINION, RECAP, ORAL_ARGUMENT, PEOPLE]
    
    
def serialize_datetime(v: dt.datetime) -> str:
    """
    Serialize a datetime including timezone info.

    Uses the timezone info provided if present, otherwise uses the current runtime's timezone info.

    UTC datetimes end in "Z" while all other timezones are represented as offset from UTC, e.g. +05:00.
    """

    def _serialize_zoned_datetime(v: dt.datetime) -> str:
        if v.tzinfo is not None and v.tzinfo.tzname(None) == dt.timezone.utc.tzname(None):
            # UTC is a special case where we use "Z" at the end instead of "+00:00"
            return v.isoformat().replace("+00:00", "Z")
        else:
            # Delegate to the typical +/- offset format
            return v.isoformat()

    if v.tzinfo is not None:
        return _serialize_zoned_datetime(v)
    else:
        local_tz = dt.datetime.now().astimezone().tzinfo
        localized_dt = v.replace(tzinfo=local_tz)
        return _serialize_zoned_datetime(localized_dt)
    
    
def flatten_pydantic_instance(
    instance: BaseModel,
    prefix: str = "",
    force_str: bool = False,
) -> Dict[str, Any]:
    """
    Given a possibly nested Pydantic instance, return a flattened version of it,
    as a dict where nested traversal paths are translated to keys a__b__c.

    Args:
        instance (BaseModel): The Pydantic instance to flatten.
        prefix (str, optional): The prefix to use for the top-level fields.
        force_str (bool, optional): Whether to force all values to be strings.

    Returns:
        Dict[str, Any]: The flattened dict.

    """
    flat_data: Dict[str, Any] = {}
    for name, value in instance.model_dump().items():
        # Assuming nested pydantic model will be a dict here
        if isinstance(value, dict):
            nested_flat_data = flatten_pydantic_instance(
                instance.model_fields[name].annotation(**value),
                prefix=f"{prefix}{name}__",
                force_str=force_str,
            )
            flat_data.update(nested_flat_data)
        else:
            flat_data[f"{prefix}{name}"] = str(value) if force_str else value
    return flat_data
    

class DocMetaData(BaseModel):
    """Metadata for a document."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )


class Document(LanceModel):
    """Interface for interacting with a document."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_encoders = {dt.datetime: serialize_datetime}
    )
    id: Optional[str] = None
    content: str = Field(
        alias='text'
    )
    metadata: Optional[DocMetaData] = None
    dataframe: pd.DataFrame | None = Field(
        None,
        exclude=True,
    )
    score_: float | None = Field(
        None,
        alias='score'
    )
    embedding: Optional[Vector | List[float]] = Field(
        None,
        alias='vector',
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = self._unique_hash_id()
    
    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs_with_defaults: Any = {
            "by_alias": True, 
            "exclude_unset": True, 
            "exclude": {'embedding': True},
            **kwargs
        }
        return super().model_dump(**kwargs_with_defaults)
    
    @property
    def to_pandas(self) -> pd.DataFrame:
        """Converts the model to a pandas DataFrame."""
        flat_model = flatten_pydantic_instance(self)
        return pd.DataFrame([flat_model])

    def hash_id(self):
        """
        Creates a hash of the given content that acts as the document's ID.
        """
        text = self.content or None
        dataframe = self.dataframe.to_json() if self.dataframe is not None else None
        metadata = self.metadata.model_dump() if self.metadata is not None else None
        embedding = self.embedding if self.embedding is not None else None
        data = f"{text}{dataframe}{metadata}{embedding}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _unique_hash_id(self) -> str:
        return self.hash_id()
    
    
    @property
    def score(self):
        if self.score_ is not None:
            return str(round(self.score_, 4))
        else:
            return None
        
    @property
    def text(self):
        if self.content is not None:
            return str(self.content)
        else:
            return None

   
    def info(self) -> str:
        """Returns a detailed string representation of the Document."""
        fields = []
        if self.content is not None:
            trimmed_content = self.content[:800]
            wrapped_content = textwrap.fill(trimmed_content, width=200)
            fields.append(f"content: '{wrapped_content}'...")
        if self.dataframe is not None:
            fields.append(f"dataframe: {self.dataframe.shape}")
        if self.score is not None:
            fields.append(f"score: {self.score}")
        if self.embedding is not None:
            fields.append(f"embedding: vector of size {len(self.embedding)}")
        if self.metadata:
            fields.append(f"\nmetadata: {self.metadata.model_dump_json(indent=4)}")
        fields_str = ", ".join(fields)
        return f"{self.__class__.__name__}:\n{fields_str}"
    
    def __eq__(self, other):
        """
        Compares Documents for equality.

        Two Documents are considered equals if their dictionary representation is identical.
        """
        if type(self) != type(other):
            return False
        return self.model_dump() == other.model_dump()
    

