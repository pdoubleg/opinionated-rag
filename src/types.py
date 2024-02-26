import hashlib
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Union

from pydantic import BaseModel, ConfigDict

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
    

class Entity(str, Enum):
    """
    Enum for the different types of entities that can respond to the current message.
    """

    AGENT = "Agent"
    LLM = "LLM"
    USER = "User"
    SYSTEM = "System"


class DocMetaData(BaseModel):
    """Metadata for a document."""

    model_config = ConfigDict(extra="allow")
    source: str = "context"
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


class Document(BaseModel):
    """Interface for interacting with a document."""

    content: str
    metadata: DocMetaData

    @staticmethod
    def hash_id(doc: str) -> str:
        # Encode the document as UTF-8
        doc_utf8 = str(doc).encode("utf-8")
        # Create a SHA256 hash object
        sha256_hash = hashlib.sha256()
        # Update the hash object with the bytes of the document
        sha256_hash.update(doc_utf8)
        # Get the hexadecimal representation of the hash
        hash_hex = sha256_hash.hexdigest()
        # Convert the first part of the hash to a UUID
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
        self.metadata.model_dump_json()
        return f"{self.content} {self.metadata.model_dump_json()}"
