import os
from dotenv import load_dotenv
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, root_validator, validator
import requests
import json
from dateutil import parser
from devtools import pprint
from html2text import html2text
from pandas import DataFrame
from typing import Optional, Union, List, Set, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime
from src.doc_store.base import LegalDataSource
from src.doc_store.utils import (
    get_chain, 
    safe_eager_map, 
    safe_merge, 
    session_builder, 
    disassemble,
)

from src.schema.docket import Docket
from src.types import SEARCH_TYPES


COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v3/"
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

    def __repr__(self):
        return "<courtlistener.com Caselist>"

    def to_pandas(self):
        flatlist = []
        for case in self.cases:
            flatlist += case.flatten()
        return DataFrame(flatlist)


class OpinionsCited(BaseModel):
    resource_uri: Optional[AnyUrl] = None
    id: Optional[str] = None
    citing_opinion: Optional[str] = None
    cited_opinion: Optional[str] = None
    depth: Optional[int] = Field(
        None,
        description="The number of times the cited opinion was cited in the citing opinion",
    )


class Opinion(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )
    resource_uri: Optional[str] = None
    id: Optional[int] = None
    absolute_url: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster: Union[int, str, None] = None
    author: Optional[str] = None
    case_name: Optional[str] = None
    html: Optional[str] = None
    text: Optional[str] = None
    citing_cases: List[str] = Field(default_factory=list)
    markdown: str = ""
    opinions_cited: Union[List[str], OpinionsCited] = None
    download_url: Optional[str] = None

    def __init__(self, api_data: dict, **data):
        super().__init__(**data)
        self.case_name = get_chain(
            api_data, ["case_name", "case_name_full", "caseName"]
        )
        self.html = get_chain(
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
            lambda x: disassemble(x)["identifier"], api_data.get("opinions_cited", [])
        )
        if self.html:
            self.markdown = html2text(self.html)
        else:
            self.markdown = ""

    def citing(self) -> Set[str]:
        """Returns a set of citing cases."""
        return set(self.citing_cases)

    def __str__(self):
        return str(self.model_dump_json(indent=4))


class Case(object):
    def __init__(self, api_data):
        self.name = get_chain(api_data, ["case_name", "case_name_full", "caseName"])
        self.citation_count = get_chain(api_data, ["citation_count", "citeCount"])
        self.citations = api_data.get("citation")
        self.court = get_chain(
            api_data, ["court_citation_string", "court_exact", "court", "court_id"]
        )
        if "opinions" in api_data and api_data["opinions"]:
            self.opinions = [Opinion(op, **op) for op in api_data["opinions"]]
        else:
            self.opinions = []

        self.opinion_shape = {0: None, 1: "singleton"}.get(len(self.opinions), "list")
        self.date = get_chain(api_data, ["date_filed", "dateFiled"])
        self.people = {
            "panel": api_data.get("panel"),
            "non_participating_judges": api_data.get("non_participating_judges"),
            "judges": get_chain(api_data, ["judges", "judge"]),
            "attorneys": get_chain(api_data, ["attorneys", "attorney"]),
        }
        self.courtlistener_cluster = api_data.get("cluster_id")
        self.courtlistener_docket = api_data.get("docket")
        self.docket = Docket(**api_data)

    @property
    def cite_string(self):
        return ", ".join(self.citations)

    def __repr__(self):
        return "<Case Name: " + self.name + ">"

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
        session_builder(
            self,
            "COURTLISTENER_API_KEY",
            "https://www.courtlistener.com/api/rest/v3/",
            "Authorization",
            "Token ",
        )(self)

    def options(self, endpoint="", headers={}):
        ep = COURTLISTENER_BASE_URL + endpoint
        h = {"Authorization": f"Token {COURTLISTENER_API_KEY}"}
        h = safe_merge(h, headers)
        return requests.options(ep, headers=h).json()

    def find_cite(self, cite):
        result = self.request("search/", parameters={"citation": cite})
        return result["results"]

    def find_case_id(self, case_id):
        result = self.request("search/", parameters={"id": case_id})
        return result["results"]

    def get_forward_cites_from_id(self, case_id: int):
        """
        Calls Court Listener's front end citation lookup engine to search all records for
            a given case id. Note that the header is set to "Presidential=on" for the "status" filter.
            This means the results will not contain any non-published records.
        """
        ep = f"https://www.courtlistener.com/api/rest/v3/search/?q=cites%3A({case_id})&type=o&order_by=dateFiled%20asc&stat_Precedential=on"
        h = {"Authorization": f"Token {os.getenv('COURTLISTENER_API_KEY')}"}
        cites_from_id = requests.get(ep, headers=h).json()
        forward_citations = []
        for i in range(len(cites_from_id["results"])):
            search_result_id = cites_from_id["results"][i]["id"]
            serch_result_bluebook = self.get_bluebook_citation(case_data=cites_from_id["results"][i])
            results = (search_result_id, serch_result_bluebook)
            forward_citations.append(results)
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
        self, c, depth=1
    ):  # can take an Opinion, Case, or Caselist object.
        cases = Caselist([])
        tofetch = frozenset(c.citing())
        newtofetch = set()
        fetched = set()
        while depth > 0:
            for c in tofetch:
                thesecases = self.fetch_case(c)
                cases.add(thesecases)
                newtofetch.update(thesecases.citing())
                fetched.add(c)
            tofetch = frozenset(newtofetch.difference(fetched))
            depth -= 1
        return cases

    def fetch_forward_citations(self, case_id: int):
        """
        Fetches forward citations for a given case ID.
        """
        cases = []
        case_list = self.get_forward_cites_from_id(case_id)

        for i in range(len(case_list)):
            case = self.fetch_case(case_list[i])
            cases.append(case)

        return cases

    def get_bluebook_citation(
        self, 
        case_id: Optional[str] = None,
        case_data: Optional[Dict[str, Any]] = None,
        ) -> str:
        if not case_data:
            case_data = self.fetch_case(case_id)
            gathered_result = case_data[0].gather()
            case_name = gathered_result["docket"].case_name_short
            reporter_info = gathered_result["citations"][0]
            court = gathered_result["court"]
            decision_date = gathered_result["date"][:4]
        else:
            case_name = case_data["caseNameShort"]
            reporter_info = case_data["citation"][0]
            court = case_data["court_citation_string"]
            decision_date = parser.isoparse(case_data["dateFiled"]).strftime("%Y")
        return f"{case_name}, {reporter_info} ({court} {decision_date})"


def main():
    # Example usage of the courtlistener class and its methods
    courtlistener_api = CourtListenerCaseDataSource()
    citation = "347 U.S. 483"  # Example citation
    # Sample case id: 107423
    cases = courtlistener_api.fetch_cases_by_cite(citation)
    print(cases[0].citing())
    cited_by = courtlistener_api.fetch_cases_cited_by(cases)
    print(cited_by)


if __name__ == "__main__":
    main()
