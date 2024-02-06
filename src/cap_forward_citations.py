import os
import json
import openai
from dotenv import load_dotenv
import pandas as pd
import requests
from datetime import datetime
from lxml import etree


def fetch_case(case_id: int):
    BASE_URL = "https://api.case.law/v1"
    endpoint_path = f"/cases/{case_id}"
    response = requests.get(
        f"{BASE_URL}{endpoint_path}/?full_case=true&body_format=xml",
        headers={"Authorization": f"Token {CAP_API_KEY}"},
    )
    return json.loads(response.content)


def fetch_citing_case(forward_citation: object):
    return fetch_case(forward_citation["id"])


def fetch_forward_citations(case_id: int):
    BASE_URL = "https://api.case.law/v1"
    endpoint_path = f"/cases/"
    response = requests.get(
        f"{BASE_URL}{endpoint_path}/?cites_to={case_id}",
        headers={"Authorization": f"Token {CAP_API_KEY}"},
    )
    return json.loads(response.content).get("results")


def get_bluebook_citation(case: dict):
    case_name = case["name_abbreviation"]
    reporter_info = case["citations"][0]["cite"]
    court = case["court"]["name_abbreviation"]
    decision_date = datetime.strptime(case["decision_date"], "%Y-%m-%d")
    return f"{case_name}, {reporter_info} ({court} {decision_date.year})"


def get_case_body(case: object):
    return case["casebody"]["data"]


def parse_case_body(case_body_xml: str):
    return etree.fromstring(
        bytes(case_body_xml, "utf-8"), parser=etree.XMLParser(recover=True)
    )


def dump_elems(case_body_root_elem: object):
    return [elem for elem in case_body_root_elem.iter()]


def locate_citations(citing_case_elems: list, cited_case_id: int):
    locations = []
    for location_counter, elem in enumerate(citing_case_elems):
        if elem.tag.endswith("extracted-citation"):
            case_id_info = elem.attrib.get("case-ids")
            if case_id_info is not None:
                citing_case_ids = case_id_info.split(",")
                for case_id_str in citing_case_ids:
                    if int(case_id_str) == cited_case_id:
                        locations.append(location_counter)
    return locations


def get_excerpt(
    citing_case_elems: list,
    location: int,
    length_before: int = 5,
    length_after: int = 5,
):
    excerpt_elems = citing_case_elems[
        max(0, location - length_before) : min(
            location + length_after, len(citing_case_elems)
        )
    ]
    excerpt_text = []
    for elem in excerpt_elems:
        elem_text = [text for text in elem.itertext()]
        excerpt_text.extend(elem_text)
    return " ".join(excerpt_text)


def get_text(
    citing_case_elems: list,
):
    excerpt_text = []
    for elem in citing_case_elems:
        elem_text = [text for text in elem.itertext()]
        excerpt_text.extend(elem_text)
    return " ".join(excerpt_text)


def evaluate_excerpt(excerpt: str, cited_opinion_bluebook_citation: str):
    prompt_text = f""" \
        This is an excerpt from a court opinion: {excerpt}. \
        Tell me whether this opinion evince a negative sentiment toward the \
        the following court opinion: {cited_opinion_bluebook_citation}. \
        Answer YES or NO."""
    client = openai.OpenAI()
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
    )
    return chat_completion.choices[0].message.content


def pseudo_shepardize(case_id: int):
    result = []
    case = fetch_case(case_id)
    bluebook_citation = get_bluebook_citation(case)
    forward_citations = fetch_forward_citations(case_id)
    for citation in forward_citations:
        citing_case = fetch_citing_case(citation)
        citing_case_id = citing_case["id"]
        citing_case_bluebook_citation = get_bluebook_citation(citing_case)
        print(f"Analyzing citing case: {citing_case_bluebook_citation} ...")
        citing_case_body_xml = get_case_body(citing_case)
        citing_case_root_elem = parse_case_body(citing_case_body_xml)
        citing_case_elems = dump_elems(citing_case_root_elem)
        citing_locations = locate_citations(citing_case_elems, case_id)
        for location in citing_locations:
            excerpt = get_excerpt(citing_case_elems, location)
            text = get_text(citing_case_elems)
            is_negative = evaluate_excerpt(excerpt, bluebook_citation)
            item = (
                case_id,
                bluebook_citation,
                citing_case_id,
                citing_case_bluebook_citation,
                excerpt,
                is_negative,
                text,
            )
            result.append(item)
    return result


load_dotenv()
CAP_API_KEY = os.getenv("CAP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_CASE_ID = 9117679
result = pseudo_shepardize(TARGET_CASE_ID)
column_names = [
    "Cited Case ID",
    "Cited Case Title",
    "Citing Case ID",
    "Citing Case Title",
    "Relevant Excerpt",
    "Excerpt Contains Negative Sentiment",
    "Complete Text",
]
df = pd.DataFrame(result, columns=column_names)
df.to_parquet("forward_citations.parquet", index=False)
df.to_excel("forward_citations.xlsx", index=False)
print(df.head())
