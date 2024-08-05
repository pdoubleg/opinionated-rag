from src.parsing.marker.schema.merged import MergedLine, MergedBlock, FullyMergedBlock
from src.parsing.marker.schema.page import Page
import re
import regex
from typing import List

from src.parsing.marker.settings import settings


def escape_markdown(text: str) -> str:
    """
    Escapes special Markdown characters in the given text.

    This function identifies and escapes characters that have special meaning in Markdown,
    ensuring they are treated as literal characters when the text is rendered as Markdown.

    Args:
        text (str): The input text to be escaped.

    Returns:
        str: The input text with special Markdown characters escaped.

    Example:
        >>> escape_markdown("This is a #hashtag")
        'This is a \\#hashtag'
    """
    # List of characters that need to be escaped in markdown
    characters_to_escape = r"[#]"
    # Escape each of these characters with a backslash
    escaped_text = re.sub(characters_to_escape, r'\\\g<0>', text)
    return escaped_text


def surround_text(s: str, char_to_insert: str) -> str:
    """
    Surrounds the non-whitespace content of a string with specified characters while preserving leading and trailing whitespace.

    Args:
        s (str): The input string to be modified.
        char_to_insert (str): The character(s) to insert around the non-whitespace content.

    Returns:
        str: The modified string with the non-whitespace content surrounded by the specified character(s).

    Example:
        >>> surround_text("  Hello, world!  ", "*")
        '  *Hello, world!*  '
    """
    leading_whitespace = re.match(r'^(\s*)', s).group(1)
    trailing_whitespace = re.search(r'(\s*)$', s).group(1)
    stripped_string = s.strip()
    modified_string = char_to_insert + stripped_string + char_to_insert
    final_string = leading_whitespace + modified_string + trailing_whitespace
    return final_string


def merge_spans(pages: List[Page]) -> List[List[MergedBlock]]:
    """
    Merge spans within blocks and lines of each page into MergedBlocks.

    This function processes a list of Page objects, merging spans within each block and line.
    It applies formatting (bold, italic) to text spans and creates MergedLine and MergedBlock
    objects to represent the processed content.

    Args:
        pages (List[Page]): A list of Page objects to process.

    Returns:
        List[List[MergedBlock]]: A list of lists, where each inner list contains MergedBlock
        objects representing the processed blocks for a single page.

    Note:
        This function assumes the existence of helper functions like `surround_text` and
        custom classes like MergedLine and MergedBlock.
    """
    merged_blocks = []
    for page in pages:
        page_blocks = []
        for blocknum, block in enumerate(page.blocks):
            block_lines = []
            for linenum, line in enumerate(block.lines):
                line_text = ""
                if len(line.spans) == 0:
                    continue
                fonts = []
                for i, span in enumerate(line.spans):
                    font = span.font.lower()
                    next_span = None
                    next_idx = 1
                    while len(line.spans) > i + next_idx:
                        next_span = line.spans[i + next_idx]
                        next_idx += 1
                        if len(next_span.text.strip()) > 2:
                            break

                    fonts.append(font)
                    span_text = span.text

                    # Don't bold or italicize very short sequences
                    # Avoid bolding first and last sequence so lines can be joined properly
                    if len(span_text) > 3 and 0 < i < len(line.spans) - 1:
                        if span.italic and (not next_span or not next_span.italic):
                            span_text = surround_text(span_text, "*")
                        elif span.bold and (not next_span or not next_span.bold):
                            span_text = surround_text(span_text, "**")
                    line_text += span_text
                block_lines.append(MergedLine(
                    text=line_text,
                    fonts=fonts,
                    bbox=line.bbox
                ))
            if len(block_lines) > 0:
                page_blocks.append(MergedBlock(
                    lines=block_lines,
                    pnum=block.pnum,
                    bbox=block.bbox,
                    block_type=block.block_type
                ))
        merged_blocks.append(page_blocks)

    return merged_blocks


def block_surround(text: str, block_type: str) -> str:
    """
    Surrounds the given text with appropriate Markdown syntax based on the block type.

    Args:
        text (str): The input text to be surrounded.
        block_type (str): The type of block (e.g., "Section-header", "Title", "Table", etc.).

    Returns:
        str: The text surrounded with appropriate Markdown syntax.
    """
    if block_type == "Section-header":
        if not text.startswith("#"):
            text = "\n## " + text.strip().title() + "\n"
    elif block_type == "Title":
        if not text.startswith("#"):
            text = "# " + text.strip().title() + "\n"
    elif block_type == "Table":
        text = "\n" + text + "\n"
    elif block_type == "List-item":
        text = escape_markdown(text)
    elif block_type == "Code":
        text = "\n```\n" + text + "\n```\n"
    elif block_type == "Text":
        text = escape_markdown(text)
    elif block_type == "Formula":
        if text.strip().startswith("$$") and text.strip().endswith("$$"):
            text = text.strip()
            text = "\n" + text + "\n"
    return text


def line_separator(line1: str, line2: str, block_type: str, is_continuation: bool = False) -> str:
    """
    Determines how to separate or join two lines of text based on their content and block type.

    Args:
        line1 (str): The first line of text.
        line2 (str): The second line of text.
        block_type (str): The type of block the lines belong to (e.g., "Text", "Title", "Formula").
        is_continuation (bool, optional): Whether the second line is a continuation of the first. Defaults to False.

    Returns:
        str: The two lines joined or separated according to the determined rules.
    """
    # Should cover latin-derived languages and russian
    lowercase_letters = r'\p{Lo}|\p{Ll}|\d'
    hyphens = r'-—¬'
    # Remove hyphen in current line if next line and current line appear to be joined
    hyphen_pattern = regex.compile(rf'.*[{lowercase_letters}][{hyphens}]\s?$', regex.DOTALL)
    if line1 and hyphen_pattern.match(line1) and regex.match(rf"^\s?[{lowercase_letters}]", line2):
        # Split on — or - from the right
        line1 = regex.split(rf"[{hyphens}]\s?$", line1)[0]
        return line1.rstrip() + line2.lstrip()

    all_letters = r'\p{L}|\d'
    sentence_continuations = r',;\(\—\"\'\*'
    sentence_ends = r'。ๆ\.?!'
    line_end_pattern = regex.compile(rf'.*[{lowercase_letters}][{sentence_continuations}]?\s?$', regex.DOTALL)
    line_start_pattern = regex.compile(rf'^\s?[{all_letters}]', regex.DOTALL)
    sentence_end_pattern = regex.compile(rf'.*[{sentence_ends}]\s?$', regex.DOTALL)

    text_blocks = ["Text", "List-item", "Footnote", "Caption", "Figure"]
    if block_type in ["Title", "Section-header"]:
        return line1.rstrip() + " " + line2.lstrip()
    elif block_type == "Formula":
        return line1 + "\n" + line2
    elif line_end_pattern.match(line1) and line_start_pattern.match(line2) and block_type in text_blocks:
        return line1.rstrip() + " " + line2.lstrip()
    elif is_continuation:
        return line1.rstrip() + " " + line2.lstrip()
    elif block_type in text_blocks and sentence_end_pattern.match(line1):
        return line1 + "\n\n" + line2
    elif block_type == "Table":
        return line1 + "\n\n" + line2
    else:
        return line1 + "\n" + line2


def block_separator(line1: str, line2: str, block_type1: str, block_type2: str) -> str:
    """
    Determines the separator between two blocks of text based on their types.

    Args:
        line1 (str): The last line of the first block.
        line2 (str): The first line of the second block.
        block_type1 (str): The type of the first block.
        block_type2 (str): The type of the second block.

    Returns:
        str: The appropriate separator between the two blocks.
    """
    sep = "\n"
    if block_type1 == "Text":
        sep = "\n\n"

    return sep + line2


def merge_lines(blocks: List[List[MergedBlock]]) -> List[FullyMergedBlock]:
    """
    Merges lines within blocks and across pages to create fully merged blocks of text.

    Args:
        blocks (List[List[MergedBlock]]): A list of pages, each containing a list of MergedBlock objects.

    Returns:
        List[FullyMergedBlock]: A list of FullyMergedBlock objects representing the merged text blocks.
    """
    text_blocks = []
    prev_type = None
    prev_line = None
    block_text = ""
    block_type = ""

    for idx, page in enumerate(blocks):
        for block in page:
            block_type = block.block_type
            if block_type != prev_type and prev_type:
                text_blocks.append(
                    FullyMergedBlock(
                        text=block_surround(block_text, prev_type),
                        block_type=prev_type
                    )
                )
                block_text = ""

            prev_type = block_type
            # Join lines in the block together properly
            for i, line in enumerate(block.lines):
                line_height = line.bbox[3] - line.bbox[1]
                prev_line_height = prev_line.bbox[3] - prev_line.bbox[1] if prev_line else 0
                prev_line_x = prev_line.bbox[0] if prev_line else 0
                prev_line = line
                is_continuation = line_height == prev_line_height and line.bbox[0] == prev_line_x
                if block_text:
                    block_text = line_separator(block_text, line.text, block_type, is_continuation)
                else:
                    block_text = line.text

        if settings.PAGINATE_OUTPUT and idx < len(blocks) - 1:
            block_text += "\n\n" + "-" * 16 + "\n\n" # Page separator horizontal rule

    # Append the final block
    text_blocks.append(
        FullyMergedBlock(
            text=block_surround(block_text, prev_type),
            block_type=block_type
        )
    )
    return text_blocks


def get_full_text(text_blocks: List[FullyMergedBlock]) -> str:
    """
    Combines a list of text blocks into a single string, applying appropriate separators between blocks.

    This function iterates through the given text blocks, applying a separator between
    each pair of blocks based on their types. The first block is added without a separator.

    Args:
        text_blocks (List[FullyMergedBlock]): A list of FullyMergedBlock objects to be combined.

    Returns:
        str: A single string containing all the text from the input blocks, with appropriate separators.

    Example:
        >>> blocks = [FullyMergedBlock(text="Hello", block_type="paragraph"),
        ...           FullyMergedBlock(text="World", block_type="heading")]
        >>> get_full_text(blocks)
        'Hello\n\nWorld'
    """
    full_text = ""
    prev_block = None
    for block in text_blocks:
        if prev_block:
            full_text += block_separator(prev_block.text, block.text, prev_block.block_type, block.block_type)
        else:
            full_text += block.text
        prev_block = block
    return full_text
