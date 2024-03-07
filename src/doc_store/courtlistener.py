import enum
import os
from dotenv import load_dotenv
from eyecite import get_citations
from eyecite.models import CaseCitation
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, root_validator, validator
import requests
import json
from dateutil import parser
from devtools import pprint
from html2text import html2text
from pandas import DataFrame
from typing import Optional, Tuple, Union, List, Set, Dict, Any
from abc import ABC, abstractmethod
from datetime import date, datetime
from src.schema.opinion_cluster_model import OpinionCluster
from src.schema.citations import CAPCitation
from src.doc_store.base import LegalDataSource
from src.doc_store.utils import (
    iter_get,
    parse_cap_decision_date, 
    safe_eager_map, 
    safe_merge, 
    session_builder, 
    disassemble_url,
    truncate_name,
    markdown_to_text,
)

from src.schema.docket import Docket
from src.types import SEARCH_TYPES


COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v3/"
COURTLISTENER_WEB_URL = "https://www.courtlistener.com"
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY")

load_dotenv()


class Caselist(object):
    def __init__(self, list_of_cases):
        self.cases = list_of_cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, item):
        if type(item) is slice:
            return Caselist(self.cases.__getitem__(item))
        return self.cases[item]

    def add(self, request):
        self.cases = self.cases + request.cases

    def __add__(self, other):
        return Caselist(self.cases + other.cases)

    def __radd__(self, other):
        return Caselist.__add__(self, other)

    def gather(self):
        return [x.gather() for x in self.cases]

    def citing(self):
        return set([]).union(*[x.citing() for x in self.cases])

    def to_pandas(self):
        flatlist = []
        for case in self.cases:
            flatlist += case.flatten()
        return DataFrame(flatlist)
    
    
    
class OpinionType(enum.Enum):
    """Enumeration for opionion type classification."""
    COMBINED = "010combined"
    UNANIMOUS = "015unamimous"
    LEAD = "020lead"
    PLURALITY = "025plurality"
    CONCURRENCE = "030concurrence"
    CONCUR_IN_PART = "035concurrenceinpart"
    DISSENT = "040dissent"
    ADDENDUM = "050addendum"
    REMITTUR = "060remittitur"
    REHEARING = "070rehearing"
    ON_THE_MERITS = "080onthemerits"
    ON_MOTION_TO_STRIKE = "090onmotiontostrike"
    
    
class Opinion(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
    )
    type: Optional[OpinionType] = None
    resource_uri: Optional[str] = None
    id: Optional[int] = None
    absolute_url: Optional[str] = None
    html: Optional[str] = None
    text: Optional[str] = None
    citing_cases: Optional[int | str | List[str]] = None
    markdown: str = ""
    opinions_cited: Optional[Union[str, List[str]]] = None
    date_created: Optional[str] = None

    def __init__(self, api_data: dict, **data):
        super().__init__(**data)
        self.type: OpinionType = OpinionType(api_data.get("type"))
        self.html = iter_get(
            api_data,
            [
                "html_with_citations",
                "html_columbia",
                "html_lawbox",
                "xml_harvard",
                "html",
            ],
        )
        self.text = api_data.get("plain_text")
        self.citing_cases = safe_eager_map(
            lambda x: disassemble_url(x)["identifier"], api_data.get("opinions_cited", [])
        )
        if self.html:
            self.markdown = html2text(self.html)
        else:
            self.markdown = ""
            
        if self.markdown:
            self.text = markdown_to_text(self.markdown)
        else:
            self.text = ""

    def citing(self) -> Set[str]:
        """Returns a set of citing cases."""
        return set(self.citing_cases)

    def __str__(self):
        return str(self.model_dump_json(indent=2))
    
    @property
    def web_link(self):
        return str(f"{COURTLISTENER_WEB_URL}{self.absolute_url}")
    
    
class Case(object):
    def __init__(self, api_data):
        self.id = api_data.get("id", "Missing ID")
        self.name = iter_get(api_data, ["case_name", "case_name_full", "caseName"])
        self.citation_count = iter_get(api_data, ["citation_count", "citeCount"])
        self.citations = api_data.get("citation")
        self.court = iter_get(
            api_data, ["court_citation_string", "court_exact", "court", "court_id"]
        )
        if "opinions" in api_data and api_data["opinions"]:
            self.opinions = [Opinion(op, **op) for op in api_data["opinions"]]
        else:
            self.opinions = []

        self.opinion_shape = {0: None, 1: "singleton"}.get(len(self.opinions), "list")
        self.date = iter_get(api_data, ["date_filed", "dateFiled"])
        self.people = {
            "panel": api_data.get("panel"),
            "non_participating_judges": api_data.get("non_participating_judges"),
            "judges": iter_get(api_data, ["judges", "judge"]),
            "attorneys": iter_get(api_data, ["attorneys", "attorney"]),
        }
        self.courtlistener_cluster = api_data.get("cluster_id")
        self.courtlistener_docket = api_data.get("docket")
        self.docket = Docket(**api_data)
        self.cluster = OpinionCluster(**api_data)
        self.publication_date=parser.isoparse(self.date).strftime("%Y-%m-%d")
        self.publication_year=parser.isoparse(self.date).strftime("%Y")
        self.citation_network = {'cited_by': [], 'cites': []} 

    @property
    def cite_string(self) -> str:
        """Generates a string representation of citations.

        Returns:
            str: A string of citations separated by commas, or an empty string if no citations.
        """
        if isinstance(self.citations, list):
            return ", ".join(self.citations)
        elif isinstance(self.citations, str):
            return self.citations
        elif isinstance(self.citations, CAPCitation):
            return str(self.citations)
        elif isinstance(self.citations, CaseCitation):
            return self.citations.corrected_citation()
        return ""
    
    
    @property
    def name_short(self):
        return truncate_name(self.name)
    
    @property
    def bluebook_citation(self):
        name_short = self.name_short
        normalized_cite = self.cite_string
        court = self.court
        year = self.publication_year
        return f"{name_short}, {normalized_cite} ({court} {year})"

    def basicdict(self):
        return {key: val for key, val in self.__dict__.items() if key != "opinions"}

    def gather(self):
        gathered = self.basicdict()
        gathered.update({"opinions": [x.__dict__ for x in self.opinions]})
        return gathered

    def __str__(self):
        return str(self.gather())

    def flatten(self):
        if self.opinion_shape:
            oprows = []
            for op in self.opinions:
                d = self.basicdict()
                d.pop("opinion_shape", None)
                oprows.append(safe_merge(d, op.__dict__))
            return oprows
        d = self.basicdict()
        d.pop("opinion_shape", None)
        return [d]

    def citing(self):
        return set([]).union(*[x.citing() for x in self.opinions])


class CourtListenerCaseDataSource(LegalDataSource):
    def __init__(self):
        """
        Initializes the CourtListenerCaseDataSource object by setting up the session with the CourtListener API.
        """
        session_builder(
            self,
            "COURTLISTENER_API_KEY",
            "https://www.courtlistener.com/api/rest/v3/",
            "Authorization",
            "Token ",
        )(self)

    def options(self, endpoint: str = "", headers: dict = {}) -> dict:
        """
        Fetches the HTTP OPTIONS for a given API endpoint.

        Args:
            endpoint (str): The API endpoint to query.
            headers (dict): Additional headers to include in the request.

        Returns:
            dict: The JSON response from the OPTIONS request.
        """
        ep = COURTLISTENER_BASE_URL + endpoint
        h = {"Authorization": f"Token {os.getenv('COURTLISTENER_API_KEY')}"}
        h = safe_merge(h, headers)
        return requests.options(ep, headers=h).json()

    def find_cite(self, cite: str) -> list:
        """
        Searches for cases by citation.

        Args:
            cite (str): The citation to search for.

        Returns:
            list: A list of search results.
        """
        result = self.request("search/", parameters={"citation": cite})
        return result["results"]

    def find_case_id(self, case_id: str) -> list:
        """
        Searches for a case by its ID.

        Args:
            case_id (str): The ID of the case to search for.

        Returns:
            list: A list of search results.
        """
        result = self.request("search/", parameters={"id": case_id})
        return result["results"]

    def basic_search(self, query: str) -> list:
        """
        Performs a basic search query on the CourtListener database.

        Args:
            query (str): The search query string.

        Returns:
            list: A list of search results.
        """
        result = self.request("search/", parameters={"q": query})
        return result["results"]

    def get_forward_citation_ids(self, case_id: int) -> List[int]:
        """
        Calls Court Listener's front end citation lookup engine to search all records for
            a given case id. Note that the header is set to "Presidential=on" for the "status" filter.
            This means the results will not contain any non-published records.

        Args:
            case_id (int): The unique identifier for the case to find forward citations for.

        Returns:
            List[int]: A list of ids for cases that cite the target case. 
        """
        ep = f"https://www.courtlistener.com/api/rest/v3/search/?q=cites%3A({case_id})&type=o&order_by=dateFiled%20asc&stat_Precedential=on"
        h = {"Authorization": f"Token {os.getenv('COURTLISTENER_API_KEY')}"}
        cites_from_id = requests.get(ep, headers=h).json()
        forward_citations = []
        for result in cites_from_id["results"]:
            try:
                search_result_id = result["id"]
                forward_citations.append(search_result_id)
            except Exception:
                # If an error occurs, skip adding this result to the list
                pass
        return forward_citations

    def extract_case_searches(self, searchres):
        cases = []
        for res in searchres:
            bigdict = {}
            bigdict = safe_merge(bigdict, res)
            cluster = self.request("clusters/" + str(res["cluster_id"]))
            bigdict = safe_merge(bigdict, cluster)
            docket = self.request(cluster["docket"])
            bigdict = safe_merge(bigdict, docket)
            opinion_results = []
            for op in cluster["sub_opinions"]:
                opinion_results.append(self.request(op))
            bigdict = safe_merge(bigdict, {"opinions": opinion_results})
            cases.append(bigdict)
        return Caselist([Case(x) for x in cases])

    def search(self, search_header, verbose=True):
        current = self.request("search/", parameters=search_header)
        reslist = []
        while True:
            reslist = reslist + current["results"]
            if current["next"]:
                if verbose:
                    print("requesting: " + current["next"])
                current = self.request(current["next"])
            else:
                break
        return self.extract_case_searches(reslist)
    
    def fetch_cases_by_query(self, query):
        return self.search({"q": query})

    def fetch_cases_by_cite(self, cite):
        return self.search({"citation": cite})

    def fetch_cases_by_judge(self, judge):
        return self.search({"judge": judge})

    def fetch_case(self, case_id: int):
        """
        Runs id-based search for opinions.

        Args:
            id (int): The id of the opinion to search for

        Returns:
            Caselist: A data model for a list of "cases"
        """
        return self.search({"type": SEARCH_TYPES.OPINION, "q": "id:{}".format(case_id)})

    def fetch_cases_cited_by(
        self, c, depth: int = 1, verbose: bool = True
    ) -> Caselist:
        """
        Fetches cases cited by a given case, opinion, or caselist, exploring up to a specified depth.

        Args:
            c: Can be an Opinion, Case, or Caselist object.
            depth (int): The depth to explore citations. Default is 1.
            verbose (bool): If True, prints out ids as they are being fetched. Default is False.

        Returns:
            Caselist: A Caselist object containing all cases cited by the target case,
                      explored up to the specified depth.
        """
        cases = Caselist([])
        tofetch = frozenset(c.citing())
        newtofetch = set()
        fetched = set()
        while depth > 0:
            for cs in tofetch:
                if verbose:
                    print(f"Fetching case ID: {str(cs)}")
                thesecases = self.fetch_case(cs)
                if verbose:
                    try:
                        print(f"Citation: {thesecases.cases[0].bluebook_citation}")
                    except:
                        print(f"Case Name: {thesecases.cases[0].name}")
                cases.add(thesecases)
                for case in thesecases.cases:
                    c.citation_network['cites'].append(case.id)
                    case.citation_network['cited_by'].append(c.id)
                newtofetch.update(thesecases.citing())
                fetched.add(cs)
            tofetch = frozenset(newtofetch.difference(fetched))
            depth -= 1
        return cases
    
    
    def fetch_forward_citations(self, case_id: int, depth: int = 1, verbose: bool = True) -> Caselist:
        """
        Fetches forward citations for a given case ID, exploring up to a specified depth, with an option to print verbose output.

        Args:
            case_id (int): The ID of the case for which to fetch forward citations.
            depth (int): The depth to explore forward citations. Default is 1.
            verbose (bool): If True, prints out ids as they are being fetched. Default is False.

        Returns:
            Caselist: A Caselist object containing all cases that cite the target case,
                      explored up to the specified depth.
        """
        cases = Caselist([])
        tofetch = set(self.get_forward_citation_ids(case_id))
        newtofetch = set()
        fetched = set()

        while depth > 0:
            for cid in tofetch:
                if cid not in fetched:
                    if verbose:
                        print(f"Fetching forward citation ID: {cid}")
                    thesecases = self.fetch_case(cid)
                    if verbose:
                        try:
                            print(f"Citation: {thesecases.cases[0].bluebook_citation}")
                        except:
                            print(f"Case Name: {thesecases.cases[0].name}")
                    cases.add(thesecases)
                    for case in thesecases.cases:
                        case.citation_network['cited_by'].append(case_id)
                        self.fetch_case(case_id)[0].citation_network['cites'].append(case.id)
                    newtofetch.update(self.get_forward_citation_ids(cid))
                    fetched.add(cid)
            tofetch = newtofetch.difference(fetched)
            newtofetch = set()
            depth -= 1

        return cases

    def get_bluebook_citation(
        self, 
        case_id: Optional[Union[str, int]] = None,
        ) -> str:
        case_data = self.fetch_case(case_id)
        gathered_result = case_data[0].gather()
        case_name = gathered_result["docket"].case_name_short
        reporter_info = gathered_result["citations"][0]
        court = gathered_result["court"]
        decision_date = gathered_result["date"][:4]
        return f"{case_name}, {reporter_info} ({court} {decision_date})"




