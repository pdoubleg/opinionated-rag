"""Download client for Harvard's Case Access Project."""

from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Union

import requests

from src.doc_store.base import LegalDataSource

from .utils import normalize_case_cite
from src.schema.citations import CaseCitation
from src.schema.decisions import CAPCitation, Decision

load_dotenv()


class CAPClient(LegalDataSource):
    """Downloads judicial decisions from Case Access Project API."""

    def __init__(self, api_token: Optional[str] = None):
        """
        Create download client with an API token and an API address.

        Args:
            api_token (Optional[str]): The API token for authentication. Defaults to None.
        """
        self.endpoint: str = "https://api.case.law/v1/cases/"
        self.api_token: Optional[str] = api_token or os.getenv("CAP_API_KEY")
        self.api_alert: str = (
            "To fetch full opinion text using the full_case parameter, "
            "set the CAPClient's 'api_key' attribute to "
            "your API key for the Case Access Project. See https://api.case.law/"
        )

    def fetch_case(
        self,
        query: Union[int, str, CaseCitation, CAPCitation],
        full_case: bool = False,
        body_format: Optional[str] = None,
    ) -> requests.models.Response:
        """
        Query by CAP id or citation, and download Decision from CAP API.

        Args:
            query (Union[int, str, CaseCitation, CAPCitation]): The query parameter, which can be an ID, a string, or a citation object.
            full_case (bool): Whether to fetch the full case text. Defaults to False.

        Returns:
            requests.models.Response: The response from the CAP API.
        """
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            return self.fetch_id(
                int(query), full_case=full_case, body_format=body_format
            )
        return self.fetch_cite(query, full_case=full_case, body_format=body_format)

    def read(
        self, query: Union[int, str, CaseCitation, CAPCitation], full_case: bool = False
    ) -> Decision:
        """
        Query by CAP id or citation, download, and load Decision from CAP API.

        Args:
            query (Union[int, str, CaseCitation, CAPCitation]): The query parameter, which can be an ID, a string, or a citation object.
            full_case (bool): Whether to fetch the full case text. Defaults to False.

        Returns:
            Decision: The decision object loaded from the CAP API response.
        """
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            return self.read_id(int(query), full_case=full_case)
        return self.read_cite(query, full_case=full_case)

    def get_api_headers(self, full_case: bool = False) -> Dict[str, str]:
        """
        Get API headers based on whether the full case text is requested.

        Args:
            full_case (bool): Whether to fetch the full case text. Defaults to False.

        Returns:
            Dict[str, str]: The headers for the API request.
        """
        api_dict: Dict[str, str] = {}
        if full_case:
            if not self.api_token:
                raise CaseAccessProjectAPIError(self.api_alert)
            api_dict["Authorization"] = f"Token {self.api_token}"
        return api_dict


    def fetch_cite(
        self,
        cite: Union[str, CaseCitation, CAPCitation],
        full_case: bool = False,
        body_format: Optional[str] = None,
    ) -> requests.models.Response:
        """
        Get the API list response for a queried citation from the CAP API.

        Args:
            cite (Union[str, CaseCitation, CAPCitation]): A citation linked to an opinion in the Caselaw Access Project database.
            full_case (bool): Whether to request the full text of the opinion. Defaults to False.
            body_format (Optional[str]): The format of the body text (text, html, xml). Only applicable if full_case is True. Defaults to None.

        Returns:
            requests.models.Response: The response from the CAP API.

        Raises:
            ValueError: If body_format is provided but full_case is False.
        """
        if body_format and not full_case:
            raise ValueError("body_format can only be specified if full_case is True.")

        normalized_cite: str = normalize_case_cite(cite)

        params: Dict[str, str] = {"cite": normalized_cite}

        headers: Dict[str, str] = self.get_api_headers(full_case=full_case)

        if full_case:
            params["full_case"] = "true"
            if body_format:
                params["body_format"] = body_format
        response: requests.models.Response = requests.get(
            self.endpoint, params=params, headers=headers
        )
        if response.status_code == 401:
            detail: str = response.json()["detail"]
            raise CaseAccessProjectAPIError(f"{detail} {self.api_alert}")
        return response

    def get_forward_citation_ids(self, case_id: int) -> List[str]:
            """
            Get the API 'cites_to' response for a given case id.

            Args:
                case_id (str): An identifier for an opinion in the Caselaw Access Project database.

            Returns:
                List[str]: A list of case ids that cited the input case id.
            """

            params: Dict[str, str] = {"cites_to": case_id}

            headers: Dict[str, str] = self.get_api_headers(full_case=False)

            response: requests.models.Response = requests.get(
                self.endpoint, params=params, headers=headers
            )
            forward_cites = json.loads(response.content).get("results")
            results = [forward_cite["id"] for forward_cite in forward_cites]

            if response.status_code == 401:
                detail: str = response.json()["detail"]
                raise CaseAccessProjectAPIError(f"{detail} {self.api_alert}")
            return results

    def read_decision_list_by_cite(
        self, cite: Union[str, CaseCitation, CAPCitation], full_case: bool = False, body_format: Optional[str] = None,
    ) -> List[Decision]:
        """
        Download and deserialize the "results" list for a queried citation from the CAP API.

        Args:
            cite (Union[str, CaseCitation, CAPCitation]): A citation linked to an opinion in the Caselaw Access Project database.
            full_case (bool): Whether to request the full text of the opinion. Defaults to False.

        Returns:
            List[Decision]: A list of decision objects deserialized from the CAP API response.
        """
        response: requests.models.Response = self.fetch_cite(
            cite=cite, full_case=full_case, body_format=body_format
        )
        return self.read_decisions_from_response(response)

    def read_decisions_from_response(
        self, response: requests.models.Response
    ) -> List[Decision]:
        """
        Deserialize a list of cases from the "results" list of a response from the CAP API.

        Args:
            response (requests.models.Response): A response from the CAP API.

        Returns:
            List[Decision]: A list of decision objects deserialized from the response.
        """
        results: List[Dict[str, Any]] = response.json()["results"]
        return [Decision(**result) for result in results]

    def read_decision_from_response(
        self, response: requests.models.Response
    ) -> Decision:
        """
        Deserialize a single case from the "results" list of a response from the CAP API.

        Args:
            response (requests.models.Response): A response from the CAP API.

        Returns:
            Decision: A decision object deserialized from the response.
        """
        decision: Dict[str, Any] = response.json()
        if "results" in decision:
            return Decision(**decision["results"][0])
        return Decision(**decision)

    def read_cite(
        self, cite: Union[str, CaseCitation, CAPCitation], full_case: bool = False, body_format: Optional[str] = None,
    ) -> Decision:
        """
        Download and deserialize a Decision from Caselaw Access Project API.

        Args:
            cite (Union[str, CaseCitation, CAPCitation]): A citation linked to an opinion in the Caselaw Access Project database.
            full_case (bool): Whether to request the full text of the opinion. Defaults to False.

        Returns:
            Decision: A decision object deserialized from the CAP API response.
        """
        response: requests.models.Response = self.fetch_cite(
            cite=cite, full_case=full_case, body_format=body_format
        )
        return self.read_decision_from_response(response=response)

    def fetch_id(
        self, case_id: int, full_case: bool = True, body_format: Optional[str] = "html"
    ) -> requests.models.Response:
        """
        Download a decision from Caselaw Access Project API.

        Args:
            case_id (int): An identifier for an opinion in the Caselaw Access Project database.
            full_case (bool): Whether to request the full text of the opinion. Defaults to True.
            body_format (Optional[str]): The format of the body text (text, html, xml). Defaults to html.

        Returns:
            requests.models.Response: The response from the CAP API.
        """
        if body_format and not full_case:
            raise ValueError("body_format can only be specified if full_case is True.")

        url: str = self.endpoint + f"{case_id}/"
        headers: Dict[str, str] = self.get_api_headers(full_case=full_case)
        params: Dict[str, str] = {}
        if full_case:
            params["full_case"] = "true"
        if body_format:
            params["body_format"] = body_format
        response: requests.models.Response = requests.get(
            url, params=params, headers=headers
        )
        if case_id and response.status_code == 404:
            raise CaseAccessProjectAPIError(f"API returned no cases with id {case_id}")
        return response

    def read_id(
        self, case_id: int, full_case: bool = True, body_format: Optional[str] = "html"
    ) -> Decision:
        """
        Download a decision from Caselaw Access Project API.

        Args:
            case_id (int): An identifier for an opinion in the Caselaw Access Project database.
            full_case (bool): Whether to request the full text of the opinion. Defaults to False.

        Returns:
            Decision: A decision object created from the CAP API response.
        """
        result: requests.models.Response = self.fetch_id(
            case_id=case_id,
            full_case=full_case,
            body_format=body_format,
        )

        return Decision(**result.json())
    
    
    def read_forward_citations(self, case_id: int, depth: int = 1, verbose: bool = True) -> List[Decision]:
        """
        Fetches forward citations for a given case ID, exploring up to a specified depth, with an option to print verbose output.

        Args:
            case_id (int): The ID of the case for which to fetch forward citations.
            depth (int): The depth to explore forward citations. Default is 1.
            verbose (bool): If True, prints out ids as they are being fetched. Default is True.

        Returns:
            List[Decision]: A list of Decision objects for all cases that cite the target case,
                            explored up to the specified depth.
        """
        decisions = []
        tofetch = set(self.get_forward_citation_ids(case_id))
        newtofetch = set()
        fetched = set()

        while depth > 0:
            for cid in tofetch:
                if cid not in fetched:
                    if verbose:
                        print(f"Fetching forward citation ID: {cid}")
                    case_data = self.read_id(cid, full_case=True, body_format="html")
                    decisions.append(case_data)
                    if verbose:
                        print(f"Citation: {str(case_data)}")
                    newtofetch.update(self.get_forward_citation_ids(cid))
                    fetched.add(cid)
            tofetch = newtofetch.difference(fetched)
            newtofetch = set()
            depth -= 1

        return decisions
    
    def fetch_cited_by(self, case_id: int) -> List[int]:
        """
        Fetches a list of ids for cases cited by the target case.

        Args:
            case_id (int): The ID of the target case.

        Returns:
            List[int]: A list of case IDs that are cited by the target case.
        """
        case = self.read_id(case_id)
        citations = case.cites_to
        ids = []
        for cite in citations:
            for id in cite.case_ids:
                ids.append(id)            
        return ids
    
    
    def read_cited_by(self, case_id: int, depth: int = 1, verbose: bool = True) -> List[Decision]:
        """
        Fetches Decision objects for cases cited by the target case, exploring up to a specified depth.

        Args:
            case_id (int): The ID of the target case.
            depth (int): The depth to explore citations. Default is 1.
            verbose (bool): If True, prints out ids as they are being fetched. Default is True.

        Returns:
            List[Decision]: A list of Decision objects for cases cited by the target case, explored up to the specified depth.
        """
        fetched_cases = set()
        cases_to_fetch = set(self.fetch_cited_by(case_id))
        new_cases_to_fetch = set()

        decisions = []

        while depth > 0 and cases_to_fetch:
            for cid in cases_to_fetch:
                if cid not in fetched_cases:
                    if verbose:
                        print(f"Fetching case ID: {cid}")
                    try:
                        decision = self.read_id(int(cid), full_case=True, body_format="html")
                        decisions.append(decision)
                        if verbose:
                            print(f"Fetched Decision: {str(decision)}")
                        new_cases_to_fetch.update(self.fetch_cited_by(cid))
                    except Exception as e:
                        if verbose:
                            print(f"Error fetching case ID {cid}: {e}")
                    finally:
                        fetched_cases.add(cid)
            cases_to_fetch = new_cases_to_fetch - fetched_cases
            new_cases_to_fetch = set()
            depth -= 1

        return decisions
    
    def get_bluebook_citation(
        self, 
        case_id: Optional[Union[str, int]] = None,
        ) -> str:
        decision = self.read_id(case_id, full_case=True, body_format="html")
        return str(decision)
    
    
    


class CaseAccessProjectAPIError(Exception):
    """Error for failed attempts to use the Case Access Project API."""

    pass