import hashlib
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd

from pydantic import BaseModel, ConfigDict, Field

Number = Union[int, float]
Embedding = List[Number]
Embeddings = List[Embedding]
EmbeddingFunction = Callable[[List[str]], Embeddings]


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
    


class DocMetaData(BaseModel):
    """Metadata for a document."""

    model_config = ConfigDict(extra="allow")
    source: str = "context"
    is_chunk: bool = False  # if it is a chunk, don't split
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


class Document(BaseModel):
    """Interface for interacting with a document."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    id: str = ""
    content: str
    metadata: DocMetaData
    dataframe: pd.DataFrame | None = Field(
        None,
        exclude=True,
    )
    score: float | None = None
    embedding: List[float] | None = Field(
        None,
        exclude=True,
    )

    def hash_id(self):
        """
        Creates a hash of the given content that acts as the document's ID.
        """
        text = self.content or None
        dataframe = self.dataframe.to_json() if self.dataframe is not None else None
        metadata = self.metadata.model_dump() or {}
        embedding = self.embedding if self.embedding is not None else None
        data = f"{text}{dataframe}{metadata}{embedding}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _unique_hash_id(self) -> str:
        return self.hash_id(str(self))

    def __str__(self) -> str:
        return f"{self.content[:250]}\n\n{self.metadata.model_dump_json(indent=4)}"
    
    def __repr__(self):
        fields = []
        if self.content is not None:
            fields.append(
                f"content: '{self.content[:250]}'...'"
            )
        if self.dataframe is not None:
            fields.append(f"\ndataframe: {self.dataframe.shape}")
        if self.metadata:
            fields.append(f"\nmetadata: {self.metadata.model_dump_json(indent=4)}")
        if self.score is not None:
            fields.append(f"\nscore: {self.score}")
        if self.embedding is not None:
            fields.append(f"\nembedding: vector of size {len(self.embedding)}")
        fields_str = ", ".join(fields)
        return f"{self.__class__.__name__}(id={self.id}, {fields_str})"
    
    def __eq__(self, other):
        """
        Compares Documents for equality.

        Two Documents are considered equals if their dictionary representation is identical.
        """
        if type(self) != type(other):
            return False
        return self.model_dump() == other.model_dump()
    
    def __post_init__(self):
        """
        Generate the ID based on the init parameters.
        """
        # Generate an id only if not explicitly set
        self.id = self.id or self._unique_hash_id()

