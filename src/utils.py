import os
import pandas as pd
import tiktoken
import hashlib
import uuid


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


def count_tokens(string: str, model: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def consolidate_strings(strings):
    consolidated = []

    while strings:
        current = strings.pop(0)
        index, max_overlap = -1, 0

        for i, string in enumerate(strings):
            overlap = min(len(current), len(string))
            while overlap > 0:
                if current[-overlap:] == string[:overlap] and overlap > max_overlap:
                    max_overlap = overlap
                    index = i
                overlap -= 1

        if index >= 0:
            current += strings.pop(index)[max_overlap:]
        consolidated.append(current)

    return consolidated
