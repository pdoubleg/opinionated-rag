from datetime import datetime
import os
import re
import pandas as pd
import requests
from src.parsing.utils import clean_whitespace
from src.utils.citations import create_annotated_text


vs_check = re.compile(" [vV][sS]?[.]? ")


def truncate_name(case_name):
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


def parse_cap_decision_date(decision_date_text):
    """
    Parse a CAP decision date string into a datetime object.

    >>> assert parse_cap_decision_date('2019-10-27') == date(2019, 10, 27)
    >>> assert parse_cap_decision_date('2019-10') == date(2019, 10, 1)
    >>> assert parse_cap_decision_date('2019') == date(2019, 1, 1)
    >>> assert parse_cap_decision_date('2019-02-29') == date(2019, 2, 1)  # non-existent day of month
    >>> assert parse_cap_decision_date('not a date') is None
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


def looks_like_citation(s):
    """
    Return True if string s looks like a case citation (starts and stops with digits).

    >>> all(looks_like_citation(s) for s in [
    ...     "123 Mass. 456",
    ...     "123-mass-456",
    ...     "123 anything else here 456",
    ...     1234567890,  # Example integer input
    ... ])
    True
    >>> not any(looks_like_citation(s) for s in [
    ...     "123Mass.456",
    ...     "123 Mass.",
    ...     123,  # Example integer that should not match
    ...     "123456",  # Example of an all-digit input that should return False
    ... ])
    True
    """
    s = str(s)  # Convert input to string to ensure compatibility with regex
    # Check if the input is composed entirely of digits
    if s.isdigit():
        return False  # It's likely an ID, not a citation

    # Proceed with the original regex check
    return bool(re.match(r"\d+(\s+|-).*(\s+|-)\d+$", s))


def looks_like_case_law_link(s):
    return bool(re.match(r"^https?://cite\.case\.law(/[/0-9a-zA-Z_-]*)$", s))


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


def get_chain(source: dict, alternatives: list) -> any:
    """
    Iterates through a list of alternative keys and returns the value from the source dictionary for the first key that exists and has a truthy value.

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


def safe_merge(d1: dict, d2: dict) -> dict:
    """
    Merges two dictionaries into a new dictionary. If a key exists in both dictionaries, the value from the second dictionary is used.
    If a key's value is falsy in the second dictionary but truthy in the first, the value from the first dictionary is used.
    If a key's value is falsy in both dictionaries, None is assigned to that key in the result.

    Args:
        d1 (dict): The first dictionary.
        d2 (dict): The second dictionary, whose values take precedence over values from the first dictionary.

    Returns:
        dict: A new dictionary containing the merged key-value pairs.
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


def safe_eager_map(func, l):
    if l:
        return list(map(func, l))
    return []


def disassemble(resource_url):
    chunks = resource_url.split("/")
    return {"endpoint": chunks[-3], "identifier": chunks[-2]}


def request_builder(auth_header, baseurl, baseparams, apikey_in_params):
    def request(endpoint="", headers={}, parameters=baseparams):
        if endpoint.startswith("https://"):
            ep = endpoint
        else:
            ep = baseurl + endpoint
        h = {}
        h = safe_merge(h, headers)
        if apikey_in_params:
            p = {}
            p = safe_merge(p, parameters)
            p = safe_merge(p, auth_header)
        else:
            p = parameters
        h = safe_merge(h, auth_header)
        result = requests.get(ep, headers=h, params=p)
        result.raise_for_status()
        return result.json()

    return request


def session_builder(
    selfvar,
    keyenv,
    baseurl,
    keyheader,
    key_prefix="",
    baseparams={},
    apikey_in_params=False,
):
    def class_init(selfvar, api_key="ENV"):
        if api_key == "ENV":
            try:
                selfvar.api_key = os.environ[keyenv]
            except KeyError as e:
                raise Exception(
                    "API token is missing. Please set the {} environment variable or pass the token to the session constructor.".format(
                        keyenv
                    )
                ) from e
        else:
            selfvar.api_key = api_key
        auth_header = {keyheader: key_prefix + selfvar.api_key}
        selfvar.request = request_builder(
            auth_header, baseurl, baseparams, apikey_in_params
        )

    return class_init
