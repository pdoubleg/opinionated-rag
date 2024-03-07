from collections import Counter
from datetime import date, datetime
import hashlib
import os
import re
from eyecite import get_citations
from eyecite.models import CaseCitation
from markdown import markdown
from bs4 import BeautifulSoup
from typing import Any, Callable, Dict, List, Optional, Union
import json
from typing import List, Type, TypeVar
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import requests
import requests_cache
from src.schema.citations import CAPCitation
from src.parsing.utils import clean_whitespace
from src.utils.citations import create_annotated_text
from .logger import get_console_logger

logger = get_console_logger("utils")

load_dotenv()

T = TypeVar('T', bound=BaseModel)


def filter_decisions_with_text(decisions: List[BaseModel]) -> List[BaseModel]:
    """
    Filters a list of Decision objects to include only those with text in either the 
    decision.casebody.data attribute or the first opinion's text attribute in decision.casebody.opinions.

    Args:
        decisions (List[Decision]): A list of Decision objects to filter.

    Returns:
        List[Decision]: A list of Decision objects that contain text.
    """
    filtered_decisions = []
    for decision in decisions:
        if decision.casebody:
            if isinstance(decision.casebody.data, str) and decision.casebody.data.strip():
                filtered_decisions.append(decision)
            elif decision.casebody.data.opinions and decision.casebody.data.opinions[0].text.strip():
                filtered_decisions.append(decision)
    return filtered_decisions


def count_duplicate_ids(decisions: List[BaseModel]) -> int:
    """
    Counts the number of duplicate `id` values in a list of Decision objects.

    Args:
        decisions (List[Decision]): A list of Decision objects.

    Returns:
        int: The count of duplicate `id` values.
    """
    id_counts = Counter([decision.id for decision in decisions])
    duplicate_count = sum(count > 1 for count in id_counts.values())
    return duplicate_count


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def convert_timestamp_to_datetime(timestamp: str) -> str:
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")


def normalize_case_cite(cite: Union[str, CaseCitation, CAPCitation]) -> str:
    """Get just the text that identifies a citation."""
    if isinstance(cite, CAPCitation):
        return cite.cite
    if isinstance(cite, str):
        possible_cites = list(get_citations(cite))
        bad_cites = []
        for possible in possible_cites:
            if isinstance(possible, CaseCitation):
                return possible.corrected_citation()
            bad_cites.append(possible)
        print(f"Could not locate a CaseCitation in the text for: type{type(cite)}.")
        return cite
        
    return cite.corrected_citation()


def save_models_to_json(models: List[BaseModel], file_path: str) -> None:
    """
    Saves a list of Pydantic models to a JSON file.

    Args:
        models: A list of Pydantic model instances.
        file_path: The path to the JSON file where the data will be saved.

    Returns:
        None
    """
    with open(file_path, 'w') as f:
        # Convert list of models to list of dictionaries
        data = [model.model_dump() for model in models]
        json.dump(data, f, indent=4)
        

def load_models_from_json(model_class: Type[T], file_path: str) -> List[T]:
    """
    Loads JSON data from a file and converts it into a list of Pydantic models.

    Args:
        model_class: The Pydantic model class to which the JSON objects will be converted.
        file_path: The path to the JSON file from which the data will be loaded.

    Returns:
        A list of Pydantic model instances.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        models = [model_class.model_validate(item) for item in data]
    return models   

def markdown_to_text(markdown_string: str) -> str:
    """Converts a markdown string to plain text.
    
    Args:
        markdown_string (str): The markdown string to convert.
    
    Returns:
        str: The converted plain text string.
    """
    # Convert markdown to HTML
    html = markdown(markdown_string)
    
    # Use BeautifulSoup to extract text from the HTML
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    
    return text

vs_check = re.compile(" [vV][sS]?[.]? ")


def truncate_name(case_name: str) -> str:
    """
    Truncates a case name to a maximum length, splitting it around a 'v.' if present.

    If the case name contains a 'v.' (indicating a versus), it splits the name into two parts,
    truncates each part to a maximum length of 40 characters, and then rejoins them with ' v. '.
    If the case name does not contain a 'v.', it truncates the entire name to a maximum of 80 characters.
    If the truncated part exceeds its maximum length, '...' is appended to indicate truncation.

    Args:
        case_name: The full name of the case to be truncated.

    Returns:
        The truncated case name.

    """
    max_part_length = 40
    parts = vs_check.split(case_name)
    if len(parts) != 2:
        return case_name[: max_part_length * 2 + 4] + (
            "..." if len(case_name) > (max_part_length * 2 + 4) else ""
        )
    part_a = parts[0][:max_part_length] + (
        "..." if len(parts[0]) > max_part_length else ""
    )
    part_b = parts[1][:max_part_length] + (
        "..." if len(parts[1]) > max_part_length else ""
    )
    return part_a + " v. " + part_b


def parse_cap_decision_date(decision_date_text: str) -> Optional[date]:
    """
    Parses a decision date string into a date object.

    Attempts to parse the date string in the format 'YYYY-MM-DD'. If the day is out of range for the month,
    it retries without the day part ('YYYY-MM'). If that still fails, it tries with just the year ('YYYY').
    If all parsing attempts fail, it returns None.

    Args:
        decision_date_text: A string representing the decision date in 'YYYY-MM-DD', 'YYYY-MM', or 'YYYY' format.

    Returns:
        A date object representing the decision date, or None if parsing fails.
    """
    try:
        try:
            return datetime.strptime(decision_date_text, "%Y-%m-%d").date()
        except ValueError as e:
            # if court used an invalid day of month (typically Feb. 29), strip day from date
            if e.args[0] == "day is out of range for month":
                decision_date_text = decision_date_text.rsplit("-", 1)[0]

            try:
                return datetime.strptime(decision_date_text, "%Y-%m").date()
            except ValueError:
                return datetime.strptime(decision_date_text, "%Y").date()
    except Exception:
        # if for some reason we can't parse the date, just store None
        return None


def looks_like_citation(s: str) -> bool:
    """
    Determines if a given string looks like a legal citation.

    This function first checks if the input string is composed entirely of digits, in which case it is likely an ID rather than a citation.
    If not, it proceeds to check the string against a regular expression pattern that matches typical legal citation formats.

    Args:
        s: A string to be checked for resemblance to a legal citation.

    Returns:
        A boolean value indicating whether the string looks like a legal citation.
    """
    s = str(s)  # Convert input to string to ensure compatibility with regex
    # Check if the input is composed entirely of digits
    if s.isdigit():
        return False  # It's likely an ID, not a citation

    # Proceed with the original regex check
    return bool(re.match(r"\d+(\s+|-).*(\s+|-)\d+$", s))


def looks_like_case_law_link(s: str) -> bool:
    """
    Determines if a given string is a link to a case law on the cite.case.law website.

    This function checks if the input string matches the pattern of a URL pointing to a resource on the cite.case.law domain.

    Args:
        s: A string to be checked for being a case law link.

    Returns:
        A boolean value indicating whether the string is a case law link.
    """
    return bool(re.match(r"^https?://cite\.case\.law(/[/0-9a-zA-Z_-]*)$", s))


def looks_like_court_listener_link(s: str) -> bool:
    """
    Determines if a given string is a link to a case or document on the Court Listener website.

    This function checks if the input string matches the pattern of a URL pointing to a resource on the Court Listener domain.

    Args:
        s (str): A string to be checked for being a Court Listener link.

    Returns:
        bool: A boolean value indicating whether the string is a Court Listener link.
    """
    return bool(re.match(r"^https?://www\.courtlistener\.com(/[/0-9a-zA-Z_-]*)?$", s))


def prep_text(df: pd.DataFrame):
    df["Excerpt"] = df["Excerpt"].apply(lambda x: clean_whitespace(x))
    df["Excerpt"] = df["Excerpt"].apply(lambda x: create_annotated_text(x))
    df["citing_case_bluebook_citation"] = df["citing_case_bluebook_citation"].apply(
        lambda x: create_annotated_text(x)
    )
    return df


def make_pretty_strings(df: pd.DataFrame):
    formatted_strings = [
        f"### {row['citing_case_bluebook_citation']}\n\n* {row['legal_question']}\n\n* {row['rule']}\n\n* {row['application']}\n\n* {row['citation_reference']}\n\n#### Excerpt:\n\n{row['Excerpt']}\n\n___\n\n"
        for index, row in df.iterrows()
    ]
    formatted_string = "\n\n".join(formatted_strings)
    return formatted_string


def iter_get(source: dict, alternatives: list) -> any:
    """
    Iterates through a list of alternative keys and returns the value from the source dictionary for the **first** key that exists and has a truthy value.

    Args:
        source (dict): The dictionary to search through.
        alternatives (list): A list of keys to look for in the source dictionary.

    Returns:
        any: The value from the source dictionary corresponding to the first key found in alternatives that has a truthy value, or None if none are found.
    """
    for a in alternatives:
        if a in source and source[a]:
            return source[a]
    return None


def safe_eager_map(func: Callable, l: List[Any]) -> List[Any]:
    """
    Applies a function to all elements of a list if the list is not empty, returning a new list with the results.

    Args:
        func (Callable): The function to apply to each element of the list.
        l (List[Any]): The list of elements to which the function will be applied.

    Returns:
        List[Any]: A list containing the results of applying the function to each element of the input list, or an empty list if the input list is empty.
    """
    if l:
        return list(map(func, l))
    return []


def disassemble_url(resource_url: str) -> dict:
    """
    Splits a resource URL into its endpoint and identifier components.

    Args:
        resource_url (str): The full URL of the resource.

    Returns:
        dict: A dictionary containing the 'endpoint' and 'identifier' extracted from the URL.
    """
    chunks = resource_url.split("/")
    return {"endpoint": chunks[-3], "identifier": chunks[-2]}


def safe_merge(d1: Dict[Any, Any], d2: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Merges two dictionaries, giving preference to non-null values in the second dictionary.

    Args:
        d1: The first dictionary.
        d2: The second dictionary.

    Returns:
        A dictionary containing the merged key-value pairs from both dictionaries.
    """
    keys = set(d1.keys()).union(set(d2.keys()))
    result = {}
    for k in keys:
        if k in d2 and d2[k]:
            result.update({k: d2[k]})
        elif k in d1 and d1[k]:
            result.update({k: d1[k]})
        else:
            result.update({k: None})
    return result


def request_builder(
    auth_header: Dict[str, str],
    baseurl: str,
    baseparams: Dict[str, str],
    apikey_in_params: bool,
) -> Callable:
    """
    Constructs a request function configured with base parameters and authentication headers.

    Args:
        auth_header: A dictionary containing authentication headers.
        baseurl: The base URL for the API requests.
        baseparams: Base parameters to be included in every request.
        apikey_in_params: A boolean indicating if the API key should be included in the parameters instead of the header.

    Returns:
        A function that takes an endpoint, headers, and parameters, then performs a GET request.
    """
    requests_cache.install_cache('api_cache', backend='sqlite', expire_after=3600)
    def request(
        endpoint: str = "",
        headers: Dict[str, str] = {},
        parameters: Dict[str, str] = baseparams,
    ) -> Dict[str, Any]:
        if endpoint.startswith("https://"):
            ep = endpoint
        else:
            ep = baseurl + endpoint
        h = safe_merge(headers, auth_header)
        if apikey_in_params:
            p = safe_merge(parameters, auth_header)
        else:
            p = parameters
        result = requests.get(ep, headers=h, params=p)
        result.raise_for_status()
        return result.json()

    return request


def session_builder(
    selfvar: Any,
    keyenv: str,
    baseurl: str,
    keyheader: str,
    key_prefix: str = "",
    baseparams: Dict[str, str] = {},
    apikey_in_params: bool = False,
) -> Callable:
    """
    Initializes the session with necessary configurations for making API requests.

    Args:
        selfvar: The instance of the class using this session builder.
        keyenv: The name of the environment variable where the API key is stored.
        baseurl: The base URL for the API requests.
        keyheader: The header name where the API key should be included.
        key_prefix: A prefix to be added before the API key in the header. Defaults to "".
        baseparams: Base parameters to be included in every request. Defaults to {}.
        apikey_in_params: Flag indicating if the API key should be included in the parameters instead of the header. Defaults to False.

    Returns:
        A function that initializes the class with API key and request builder.
    """

    def class_init(selfvar: Any, api_key: str = "ENV") -> None:
        """
        Inner function to initialize the class with API key and request builder.

        Args:
            selfvar: The instance of the class being initialized.
            api_key: The API key to be used. If "ENV", it will try to get it from the environment variable. Defaults to "ENV".
        """
        selfvar.api_key = os.getenv(
            keyenv, default=api_key if api_key != "ENV" else None
        )
        if selfvar.api_key is None:
            raise Exception(
                f"API token is missing. Please set the {keyenv} environment variable or pass the token to the session constructor."
            )

        auth_header = {keyheader: key_prefix + selfvar.api_key}
        selfvar.request = request_builder(
            auth_header, baseurl, baseparams, apikey_in_params
        )

    return class_init
