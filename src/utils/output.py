import asyncio
import logging
import os
import sys
from datetime import datetime
import openai
import pandas as pd
from contextlib import contextmanager
from typing import Any, Iterator, Optional
from rich import print as rprint
from rich.text import Text
import aiofiles
import urllib
import uuid
from pathlib import Path
from md2pdf.core import md2pdf

from src.utils.configuration import settings
from src.utils.constants import Colors


from datetime import datetime
import functools
import inspect
import tempfile
import markdown

@functools.lru_cache(maxsize=1000)
def download_temp_file(file_id: str, suffix: str = None):
    """
    Downloads a file from OpenAI's servers and saves it to a temporary file.

    Args:
        file_id: The ID of the file to be downloaded.
        suffix: The file extension to be used for the temporary file.

    Returns:
        The file path of the downloaded temporary file.
    """

    client = openai.OpenAI()
    response = client.files.content(file_id)

    # Create a temporary file with a context manager to ensure it's cleaned up
    # properly
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="wb", suffix=suffix)
    temp_file.write(response.content)

    return temp_file.name


def format_message(message):
    timestamp = (
        datetime.fromtimestamp(message.created_at).strftime("%I:%M:%S %p").lstrip("0")
    )
    content = []
    for item in message.content:
        if item.type == "text":
            content.append(item.text.value + "\n\n")
        elif item.type == "image_file":
            # Use the download_temp_file function to download the file and get
            # the local path
            local_file_path = download_temp_file(item.image_file.file_id, suffix=".png")
            content.append(
                f"*View attached image: [{local_file_path}]({local_file_path})*"
            )

    for file_id in message.file_ids:
        content.append(f"Attached file: {file_id}\n")

    # out_str = f"**{str(message.role.title())}**\n\n"
    # out_str = str(timestamp)
    out_str = inspect.cleandoc("\n\n".join(content))

    return out_str


async def write_to_file(filename: str, text: str) -> None:
    """
    Asynchronously write text to a file in UTF-8 encoding.

    Args:
        filename (str): The filename to write to.
        text (str): The text to write.
    """
    # Convert text to UTF-8, replacing any problematic characters
    text_utf8 = text.encode("utf-8", errors="replace").decode("utf-8")

    async with aiofiles.open(filename, "w", encoding="utf-8") as file:
        await file.write(text_utf8)


async def write_md_to_pdf(
    text: str, file_name: Optional[str] = None, output_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Converts Markdown text to a PDF file and returns the file path. Allows specifying an output directory.

    Args:
        text (str): Markdown text to convert.
        file_name (Optional[str]): Optional custom file name for the output PDF.
        output_dir (Optional[Path]): Optional directory to write the PDF file to. Defaults to './outputs'.

    Returns:
        Optional[str]: The encoded file path of the generated PDF or None if an error occurs.
    """
    if not file_name:
        file_name = datetime.now().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:5]
    if not output_dir:
        output_dir = "./outputs"
    file_path = f"{output_dir}/{file_name}"
    directory = os.path.dirname(file_path)
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    await write_to_file(f"{file_path}.md", text)

    try:
        await asyncio.to_thread(
            md2pdf,
            f"{file_path}.pdf",
            md_content=None,
            md_file_path=f"{file_path}.md",
            css_file_path="./src/style.css",
            base_url=None,
        )
        print(f"Report written to {file_path}.pdf")
    except Exception as e:
        print(f"Error in converting Markdown to PDF: {e}")
        return None
    encoded_file_path = urllib.parse.quote(f"{file_path}.pdf")
    return encoded_file_path


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


def shorten_text(text: str, chars: int = 40) -> str:
    text = " ".join(text.split())
    return text[:chars] + "..." + text[-chars:] if len(text) > 2 * chars else text


def print_long_text(
    color: str, style: str, preamble: str, text: str, chars: Optional[int] = None
) -> None:
    if chars is not None:
        text = " ".join(text.split())
        text = text[:chars] + "..." + text[-chars:] if len(text) > 2 * chars else text
    styled_text = Text(text, style=style)
    rprint(f"[{color}]{preamble} {styled_text}")


def show_if_debug(
    text: str,
    preamble: str,
    chars: Optional[int] = None,
    color: str = "red",
    style: str = "italic",
) -> None:
    if settings.debug:
        print_long_text(color, style, preamble, text, chars)


class PrintColored:
    """Context to temporarily print in a desired color"""

    def __init__(self, color: str):
        self.color = color

    def __enter__(self) -> None:
        sys.stdout.write(self.color)
        sys.stdout.flush()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        print(Colors().RESET)


@contextmanager
def silence_stdout() -> Iterator[None]:
    """
    Temporarily silence all output to stdout and from rich.print.

    This context manager redirects all output written to stdout (which includes
    outputs from the built-in print function and rich.print) to /dev/null on
    UNIX-like systems or NUL on Windows. Once the context block exits, stdout is
    restored to its original state.

    Example:
        with silence_stdout_and_rich():
            print("This won't be printed")
            rich.print("This also won't be printed")

    Note:
        This suppresses both standard print functions and the rich library outputs.
    """
    platform_null = "/dev/null" if sys.platform != "win32" else "NUL"
    original_stdout = sys.stdout
    fnull = open(platform_null, "w")
    sys.stdout = fnull
    try:
        yield
    finally:
        sys.stdout = original_stdout
        fnull.close()


class SuppressLoggerWarnings:
    def __init__(self, logger: str | None = None):
        # If no logger name is given, get the root logger
        self.logger = logging.getLogger(logger)
        self.original_level = self.logger.getEffectiveLevel()

    def __enter__(self) -> None:
        # Set the logging level to 'ERROR' to suppress warnings
        self.logger.setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        # Reset the logging level to its original value
        self.logger.setLevel(self.original_level)
        
        
from bs4 import BeautifulSoup

def generate_html_with_toc(html_text: str) -> str:
    """
    Generates HTML content with a table of contents (ToC) based on the headers found in the input HTML text.
    
    Args:
        html_text (str): The HTML content converted from Markdown.
        
    Returns:
        str: The HTML content with a ToC inserted at the beginning.
    """
    soup = BeautifulSoup(html_text, 'html.parser')

    toc_list = soup.new_tag("ul", id='table-of-contents')
    back_to_toc_template = soup.new_tag("a", href="#table-of-contents")
    back_to_toc_template.string = "Back to Table of Contents"
    back_to_toc_template['class'] = 'back-to-toc'

    for header in soup.find_all(['h1', 'h2', 'h3']):
        header_id = header.text.replace(" ", "-").lower()
        header['id'] = header_id  # Assign ID to header for linking
        
        # Create ToC entry
        li = soup.new_tag("li")
        a = soup.new_tag("a", href=f"#{header_id}")
        a.string = header.text
        li.append(a)
        toc_list.append(li)
        
        # Add back-link after the header or section
        back_link = soup.new_tag("a", href="#table-of-contents", **{'class': 'back-to-toc'})
        back_link.string = "Back to Table of Contents"
        header.insert_after(back_link)

    # Check if the body tag exists and insert the ToC, otherwise prepend to the soup object
    if soup.body:
        soup.body.insert(0, toc_list)
    else:
        soup.insert(0, toc_list)

    return str(soup)
