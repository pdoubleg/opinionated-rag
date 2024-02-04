import os
import pandas as pd
import lancedb
import eyecite
import hashlib
import uuid
from config import settings


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
