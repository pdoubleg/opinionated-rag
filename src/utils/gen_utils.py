import asyncio
from functools import cached_property
import glob
import importlib
import inspect
import math
import os
import string
import pandas as pd
import tiktoken
import hashlib
import uuid
from typing import Any, BinaryIO, Coroutine, Dict

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


def load_system_prompts(dir_path: str = "system_prompts") -> Dict[str, str]:
    """
    Load system prompts from text files in a directory.

    Args:
        dir_path (str): The path to the directory containing the system prompt files.
            Defaults to "system_prompts".

    Returns:
        Dict[str, str]: A dictionary mapping file names (without extension) to their contents.
    """
    system_prompts = {}

    # Iterate over all .txt files in the directory
    for file_path in glob.glob(os.path.join(dir_path, "*.txt")):
        # Get the base name of the file (without .txt)
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]

        # Open the file and read its content
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            content = file.read()

        # Store the content in the dictionary
        system_prompts[file_name] = content
        # Print the name of the file
        print(f"Loaded system prompt: {file_name}")

    # Print the keys of the dictionary
    print(system_prompts.keys())

    return system_prompts


def stringify(x: Any) -> str:
    # Convert x to DataFrame if it is not one already
    if isinstance(x, pd.Series):
        df = x.to_frame()
    elif not isinstance(x, pd.DataFrame):
        return str(x)
    else:
        df = x

    # Truncate long text columns to 1000 characters
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda item: (item[:1000] + "...")
                if isinstance(item, str) and len(item) > 1000
                else item
            )

    # Limit to 10 rows
    df = df.head(10)

    # Convert to string
    return df.to_string(index=False)  # type: ignore


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


def safe_import(module: str, mitigation=None):
    """
    Import the specified module. If the module is not installed,
    raise an ImportError with a helpful message.

    Parameters
    ----------
    module : str
        The name of the module to import
    mitigation : Optional[str]
        The package(s) to install to mitigate the error.
        If not provided then the module name will be used.
    """
    try:
        return importlib.import_module(module)
    except ImportError:
        raise ImportError(f"Please install {mitigation or module}")
    
    
def maybe_is_text(s: str, thresh: float = 2.5) -> bool:
    if len(s) == 0:
        return False
    # Calculate the entropy of the string
    entropy = 0.0
    for c in string.printable:
        p = s.count(c) / len(s)
        if p > 0:
            entropy += -p * math.log2(p)

    # Check if the entropy is within a reasonable range for text
    if entropy > thresh:
        return True
    return False


def maybe_is_pdf(file: BinaryIO) -> bool:
    magic_number = file.read(4)
    file.seek(0)
    return magic_number == b"%PDF"


def maybe_is_html(file: BinaryIO) -> bool:
    magic_number = file.read(4)
    file.seek(0)
    return (
        magic_number == b"<htm"
        or magic_number == b"<!DO"
        or magic_number == b"<xsl"
        or magic_number == b"<!X"
    )
    
    
async def gather_with_concurrency(n: int, coros: list[Coroutine]) -> list[Any]:
    # https://stackoverflow.com/a/61478547/2392535
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))

def get_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def is_coroutine_callable(obj):
    if inspect.isfunction(obj):
        return inspect.iscoroutinefunction(obj)
    elif callable(obj):
        return inspect.iscoroutinefunction(obj.__call__)
    return False


class DataFrameCache:
    """
    A class to cache the DataFrame loaded from a parquet file.
    """
    def __init__(self, path_to_df: str):
        self.path_to_df = path_to_df
    
    @cached_property
    def df(self) -> pd.DataFrame:
        """
        Caches and returns the DataFrame loaded from a parquet file.

        Returns:
            pd.DataFrame: The DataFrame loaded from the parquet file.
        """
        df = pd.read_parquet(self.path_to_df)
        logger.info(f"Read in df with shape: {df.shape}")
        return df
