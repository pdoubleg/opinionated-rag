import re


def cleanup_text(full_text: str) -> str:
    """
    Clean up the input text by removing excessive newlines and replacing non-breaking spaces.

    Args:
        full_text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.

    Steps:
    1. Replace three or more consecutive newlines with two newlines.
    2. Replace three or more consecutive newline-space combinations with two newlines.
    3. Replace non-breaking spaces with regular spaces.

    Example:
        >>> cleanup_text("Hello\n\n\nWorld\n \n \nTest\xa0text")
        'Hello\n\nWorld\n\nTest text'
    """
    # Step 1: Replace three or more consecutive newlines with two newlines
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # Step 2: Replace three or more consecutive newline-space combinations with two newlines
    full_text = re.sub(r'(\n\s){3,}', '\n\n', full_text)

    # Step 3: Replace non-breaking spaces with regular spaces
    full_text = full_text.replace('\xa0', ' ')

    return full_text