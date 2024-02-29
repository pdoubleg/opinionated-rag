"""Download client for Harvard's Case Access Project."""

from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Union

import requests

from .utils import normalize_case_cite
from src.schema.citations import CaseCitation
from src.schema.decisions import CAPCitation, Decision

from src.types import SEARCH_TYPES

# COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v3/"
COURTLISTENER_WEB_URL = "https://www.courtlistener.com"
# COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY")

load_dotenv()


class CourtListenerClient:
    """Downloads judicial decisions from Case Access Project API."""

    def __init__(self, api_token: Optional[str] = None):
        """
        Create download client with an API token and an API address.

        Args:
            api_token (Optional[str]): The API token for authentication. Defaults to None.
        """
        self.endpoint: str = "https://www.courtlistener.com/api/rest/v3/"
        self.api_token: Optional[str] = api_token or os.getenv("COURTLISTENER_API_KEY")
        self.api_alert: str = (
            "Set the CourtListenerClient's 'api_key' attribute to "
            "your API key for Court Listener. See https://www.courtlistener.com/help/api/"
        )

    def get_api_headers(self) -> Dict[str, str]:
        """
        Get the headers for the API request.

        Returns:
            Dict[str, str]: The headers for the API request.
        """
        return {"Authorization": f"Token {self.api_token}"}
    
    def fetch(
        self,
        query: Union[int, str, CaseCitation, CAPCitation],
        full_case: bool = False,
        body_format: Optional[str] = None,
    ) -> requests.models.Response:
        """
        Query by id or citation, and download Decision.

        Args:
            query (Union[int, str, CaseCitation, CAPCitation]): The query parameter, which can be an ID, a string, or a citation object.
            full_case (bool): Whether to fetch the full case text. Defaults to False.

        Returns:
            requests.models.Response: The response from the API.
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
        Query by CAP id or citation, download, and load Decision from API.

        Args:
            query (Union[int, str, CaseCitation, CAPCitation]): The query parameter, which can be an ID, a string, or a citation object.
            full_case (bool): Whether to fetch the full case text. Defaults to False.

        Returns:
            Decision: The decision object loaded from the API response.
        """
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            return self.read_id(int(query), full_case=full_case)
        return self.read_cite(query, full_case=full_case)
    
   

    def fetch_cite(
        self,
        cite: Union[str, CaseCitation, CAPCitation],
        full_case: bool = False,
        body_format: Optional[str] = None,
    ) -> requests.models.Response:
        """
        Get the API list response for a queried citation from the API.

        Args:
            cite (Union[str, CaseCitation, CAPCitation]): A citation linked to an opinion in the Caselaw Access Project database.
            full_case (bool): Whether to request the full text of the opinion. Defaults to False.
            body_format (Optional[str]): The format of the body text (text, html, xml). Only applicable if full_case is True. Defaults to None.

        Returns:
            requests.models.Response: The response from the API.

        Raises:
            ValueError: If body_format is provided but full_case is False.
        """

        normalized_cite: str = normalize_case_cite(cite)

        params: Dict[str, str] = {"cite": normalized_cite}

        headers: Dict[str, str] = self.get_api_headers()

        if full_case:
            params["full_case"] = "true"
            if body_format:
                params["body_format"] = body_format
        response: requests.models.Response = requests.get(
            self.endpoint, params=params, headers=headers
        )
        if response.status_code == 401:
            detail: str = response.json()["detail"]
            raise CourtListenerAPIError(f"{detail} {self.api_alert}")
        return response

    def fetch_forward_cites(self, cap_id: int) -> List[str]:
            """
            Get the API 'cites_to' response for a given case id.

            Args:
                cap_id (str): An identifier for an opinion in the Caselaw Access Project database.

            Returns:
                List[str]: A list of citation strings that cited the input case id.
            """

            params: Dict[str, str] = {"cites_to": cap_id}

            headers: Dict[str, str] = self.get_api_headers(full_case=False)

            response: requests.models.Response = requests.get(
                self.endpoint, params=params, headers=headers
            )
            forward_cites = json.loads(response.content).get("results")
            results = [forward_cite["id"] for forward_cite in forward_cites]

            if response.status_code == 401:
                detail: str = response.json()["detail"]
                raise CourtListenerAPIError(f"{detail} {self.api_alert}")
            return results

    def read_decision_list_by_cite(
        self, cite: Union[str, CaseCitation, CAPCitation], full_case: bool = False, body_format: Optional[str] = None,
    ) -> List[Decision]:
        """
        Download and deserialize the "results" list for a queried citation from the API.

        Args:
            cite (Union[str, CaseCitation, CAPCitation]): A citation linked to an opinion in the Caselaw Access Project database.
            full_case (bool): Whether to request the full text of the opinion. Defaults to False.

        Returns:
            List[Decision]: A list of decision objects deserialized from the API response.
        """
        response: requests.models.Response = self.fetch_cite(
            cite=cite, full_case=full_case, body_format=body_format
        )
        return self.read_decisions_from_response(response)

    def read_decisions_from_response(
        self, response: requests.models.Response
    ) -> List[Decision]:
        """
        Deserialize a list of cases from the "results" list of a response from the API.

        Args:
            response (requests.models.Response): A response from the API.

        Returns:
            List[Decision]: A list of decision objects deserialized from the response.
        """
        results: List[Dict[str, Any]] = response.json()["results"]
        return [Decision(**result) for result in results]

    def read_decision_from_response(
        self, response: requests.models.Response
    ) -> Decision:
        """
        Deserialize a single case from the "results" list of a response from the API.

        Args:
            response (requests.models.Response): A response from the API.

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
            Decision: A decision object deserialized from the API response.
        """
        response: requests.models.Response = self.fetch_cite(
            cite=cite, full_case=full_case, body_format=body_format
        )
        return self.read_decision_from_response(response=response)

    def fetch_id(
        self, cap_id: int, full_case: bool = False, body_format: Optional[str] = None
    ) -> requests.models.Response:
        """
        Download a decision from Caselaw Access Project API.

        Args:
            cap_id (int): An identifier for an opinion in the Caselaw Access Project database.
            full_case (bool): Whether to request the full text of the opinion. Defaults to False.
            body_format (Optional[str]): The format of the body text (text, html, xml). Defaults to None.

        Returns:
            requests.models.Response: The response from the API.
        """
        if body_format and not full_case:
            raise ValueError("body_format can only be specified if full_case is True.")

        url: str = self.endpoint + f"{cap_id}/"
        headers: Dict[str, str] = self.get_api_headers(full_case=full_case)
        params: Dict[str, str] = {}
        if full_case:
            params["full_case"] = "true"
        if body_format:
            params["body_format"] = body_format
        response: requests.models.Response = requests.get(
            url, params=params, headers=headers
        )
        if cap_id and response.status_code == 404:
            raise CourtListenerAPIError(f"API returned no cases with id {cap_id}")
        return response

    def read_id(
        self, cap_id: int, full_case: bool = False, body_format: Optional[str] = None
    ) -> Decision:
        """
        Download a decision from Caselaw Access Project API.

        Args:
            cap_id (int): An identifier for an opinion in the Caselaw Access Project database.
            full_case (bool): Whether to request the full text of the opinion. Defaults to False.

        Returns:
            Decision: A decision object created from the API response.
        """
        result: requests.models.Response = self.fetch_id(
            cap_id=cap_id,
            full_case=full_case,
            body_format=body_format,
        )

        return Decision(**result.json())


class CourtListenerAPIError(Exception):
    """Error for failed attempts to use the Case Access Project API."""

    pass
