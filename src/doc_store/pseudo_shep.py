import os
from dotenv import load_dotenv
import instructor
import openai
import pandas as pd
import requests
import json
from datetime import datetime
from lxml import etree
from typing import List, Dict, Any
from tenacity import Retrying, stop_after_attempt, wait_fixed
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from markdown import markdown
from src.doc_store.base import LegalDocument, LegalDataSource, CitationAnalysis, analyze_citation

from src.utils.citations import create_annotated_text
from parsing.utils import clean_whitespace

load_dotenv()
CAP_API_KEY = os.getenv("CAP_API_KEY")
CAP_BASE_URL = "https://api.case.law/v1"


class CAPCaseDataSource(LegalDataSource):
    """
    Concrete LegalDataSource for fetching data from the CAP API.
    """

    def __init__(self):
        self.base_url = CAP_BASE_URL
        self.api_key = CAP_API_KEY

    def fetch_case(self, case_id: int) -> Dict[str, Any]:
        """
        Fetches a case by its ID from the Case Law API and returns the case data as a dictionary.
        """
        endpoint_path = f"/cases/{case_id}"
        response = requests.get(
            f"{self.base_url}{endpoint_path}/?full_case=true&body_format=xml",
            headers={"Authorization": f"Token {self.api_key}"},
        )
        return json.loads(response.content)

    def fetch_forward_citations(self, case_id: int) -> List[Dict[str, Any]]:
        """
        Fetches forward citations for a given case ID.
        """
        endpoint_path = f"/cases/"
        response = requests.get(
            f"{self.base_url}{endpoint_path}/?cites_to={case_id}",
            headers={"Authorization": f"Token {self.api_key}"},
        )
        return json.loads(response.content).get("results")

    def get_bluebook_citation(self, case_id: int) -> str:
        case_data = self.fetch_case(case_id)
        case_name = case_data["name_abbreviation"]
        reporter_info = case_data["citations"][0]["cite"]
        court = case_data["court"]["name_abbreviation"]
        decision_date = datetime.strptime(case_data["decision_date"], "%Y-%m-%d").year
        return f"{case_name}, {reporter_info} ({court} {decision_date})"

    def fetch_citing_case(self, forward_citation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetches a citing case using the forward citation information.

        Args:
            forward_citation (Dict[str, Any]): The forward citation information containing the case ID.

        Returns:
            Dict[str, Any]: The citing case data as a dictionary.
        """
        return self.fetch_case(forward_citation["id"])

    def get_case_body(self, case: Dict[str, Any]) -> str:
        """
        Extracts the case body from the case data.

        Args:
            case (Dict[str, Any]): The case data as a dictionary.

        Returns:
            str: The case body as a string.
        """
        return case["casebody"]["data"]

    @staticmethod
    def parse_case_body(case_body_xml: str) -> etree._Element:
        """
        Parses the XML case body and returns the root element.

        Args:
            case_body_xml (str): The case body as an XML string.

        Returns:
            etree._Element: The root element of the parsed XML.
        """
        return etree.fromstring(
            bytes(case_body_xml, "utf-8"), parser=etree.XMLParser(recover=True)
        )

    @staticmethod
    def dump_elems(case_body_root_elem: etree._Element) -> List[etree._Element]:
        """
        Dumps all elements from the case body's root element.

        Args:
            case_body_root_elem (etree._Element): The root element of the case body.

        Returns:
            List[etree._Element]: A list of all elements in the case body.
        """
        return [elem for elem in case_body_root_elem.iter()]

    @staticmethod
    def locate_citations(
        citing_case_elems: List[etree._Element], cited_case_id: int
    ) -> List[int]:
        """
        Locates citations within a citing case and returns their locations.

        Args:
            citing_case_elems (List[etree._Element]): A list of elements from the citing case.
            cited_case_id (int): The ID of the cited case.

        Returns:
            List[int]: A list of locations where the cited case is referenced.
        """
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

    @staticmethod
    def get_excerpt(
        citing_case_elems: List[etree._Element],
        location: int,
        length_before: int = 12,
        length_after: int = 8,
    ) -> str:
        """
        Extracts an excerpt from the citing case elements based on a specified location.

        Args:
            citing_case_elems (List[etree._Element]): A list of elements from the citing case.
            location (int): The location index of the citation.
            length_before (int, optional): The number of elements to include before the citation. Defaults to 15.
            length_after (int, optional): The number of elements to include after the citation. Defaults to 15.

        Returns:
            str: The extracted excerpt as a string.
        """
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

    @staticmethod
    def get_text(
        citing_case_elems: List[etree._Element],
    ) -> str:
        """
        Extracts the full text from the citing case elements.

        Args:
            citing_case_elems (List[etree._Element]): A list of elements from the citing case.

        Returns:
            str: The full text extracted from the citing case elements.
        """
        excerpt_text = []
        for elem in citing_case_elems:
            elem_text = [text for text in elem.itertext()]
            excerpt_text.extend(elem_text)
        return " ".join(excerpt_text)


class ShepardizeResult(BaseModel):
    """
    Represents the result of a Shepardization process for a legal case.

    Attributes:
        case_id (int): The ID of the case being Shepardized.
        bluebook_citation (str): The Bluebook citation of the case being Shepardized.
        citing_case_id (int): The ID of a case that cites the case being Shepardized.
        citing_case_bluebook_citation (str): The Bluebook citation of the citing case.
        excerpts (List[str]): A list of text excerpts from the citing case that reference the Shepardized case.
        citation_analysis (Optional[CitationAnalysis]): An analysis of how the citing case references the Shepardized case.
    """

    case_id: int
    bluebook_citation: str
    citing_case_id: int
    citing_case_bluebook_citation: str
    excerpts: List[str] = []
    citation_analysis: Optional[CitationAnalysis] = None


class ShepardizationService:
    """
    Provides services for performing pseudo-Shepardization on legal cases.

    Attributes:
        data_source (CaseDataSource): The data source from which case data is fetched.
    """

    def __init__(self, data_source: LegalDataSource):
        """
        Initializes the ShepardizationService with a specific data source.

        Args:
            data_source (CaseDataSource): The data source to use for fetching case data.
        """
        self.data_source = data_source

    def pseudo_shepardize(self, case_id: int) -> List[ShepardizeResult]:
        """
        Performs pseudo-Shepardization on a specified case.

        Args:
            case_id (int): The ID of the case to Shepardize.

        Returns:
            List[ShepardizeResult]: A list of ShepardizeResult instances representing the results of the Shepardization process.
        """
        case = self.data_source.fetch_case(case_id)
        results: List[ShepardizeResult] = []
        forward_citations = self.data_source.fetch_forward_citations()

        for citation in forward_citations:
            citing_case = LegalDocument(citation["id"], self.data_source)
            citing_case_bluebook_citation = self.data_source.get_bluebook_citation(
                citation["id"]
            )
            citing_case_body_xml = self.data_source.get_case_body(citing_case.data)
            citing_case_root_elem = CAPCaseDataSource.parse_case_body(
                citing_case_body_xml
            )
            citing_case_elems = CAPCaseDataSource.dump_elems(citing_case_root_elem)
            citing_locations = CAPCaseDataSource.locate_citations(
                citing_case_elems, case_id
            )

            excerpts = [
                CAPCaseDataSource.get_excerpt(citing_case_elems, location)
                for location in citing_locations
            ]

            citation_analysis: CitationAnalysis = analyze_citation(
                case.get_bluebook_citation(), " ".join(excerpts)
            )

            one_result = ShepardizeResult(
                case_id=case_id,
                bluebook_citation=case.get_bluebook_citation(),
                citing_case_id=citation["id"],
                citing_case_bluebook_citation=citing_case_bluebook_citation,
                excerpts=excerpts,
                citation_analysis=citation_analysis,
            )
            results.append(one_result)
        return results
