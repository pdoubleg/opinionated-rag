import os
import requests
import json
from devtools import pprint
from html2text import html2text
from pandas import DataFrame
from typing import Optional, Union, List, Set
from pydantic import BaseModel, ConfigDict, Field, AnyUrl

from opinion import Opinion
from docket import Docket


def request_builder(baseurl, baseparams):
    """
    Builds a request function for making API calls.

    Args:
        baseurl (str): The base URL for the API.
        baseparams (dict): The base parameters to include in every request.

    Returns:
        function: A configured request function.
    """

    def request(endpoint="", headers={}, parameters=baseparams):
        """
        Makes an API request.

        Args:
            endpoint (str): The API endpoint to request.
            headers (dict): Additional headers to include in the request.
            parameters (dict): Additional parameters to include in the request.

        Returns:
            dict: The JSON response from the API.
        """
        if endpoint.startswith("https://"):
            ep = endpoint
        else:
            ep = baseurl + endpoint
        h = safe_merge({}, headers)
        p = safe_merge({}, parameters)
        result = requests.get(ep, headers=h, params=p)
        result.raise_for_status()
        return result.json()

    return request


def session_builder(selfvar, baseurl, baseparams={}):
    """
    Builds and initializes a session for making API requests.

    Args:
        selfvar: The instance of the class that will use the session.
        baseurl (str): The base URL for the API.
        baseparams (dict, optional): The base parameters to include in every request. Defaults to {}.
    """
    selfvar.request = request_builder(baseurl, baseparams)


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


def pretty_dict(some_dictionary):
    return json.dumps(some_dictionary, sort_keys=True, indent=4)


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

    def __init__(self, api_data: dict, name: str, **data):
        super().__init__(**data)
        self.case_name = name
        self.html = get_chain(
            api_data, ["html", "html_columbia", "html_lawbox", "html_with_citations"]
        )
        self.text = api_data.get("plain_text")
        self.citing_cases = safe_eager_map(
            lambda x: disassemble(x)["identifier"], api_data.get("opinions_cited", [])
        )
        self.markdown = html2text(self.html) if self.html else ""

    def citing(self) -> Set[str]:
        """Returns a set of citing cases."""
        return set(self.citing_cases)

    def __str__(self):
        return pretty_dict(self.__dict__)


class Case(object):
    def __init__(self, api_data):
        self.name = get_chain(api_data, ["case_name", "case_name_full", "caseName"])
        self.citation_count = get_chain(api_data, ["citation_count", "citeCount"])
        self.citations = api_data.get("citation")
        self.court = get_chain(
            api_data, ["court", "court_exact", "court_id", "court_citation_string"]
        )
        if "opinions" in api_data and api_data["opinions"]:
            self.opinions = [
                Opinion(op, self.name, **op) for op in api_data["opinions"]
            ]
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

    def __repr__(self):
        return "<lawpy Case, " + self.name + ">"

    def basicdict(self):
        return {key: val for key, val in self.__dict__.items() if key != "opinions"}

    def gather(self):
        gathered = self.basicdict()
        gathered.update({"opinions": [x.__dict__ for x in self.opinions]})
        return gathered

    def __str__(self):
        return pretty_dict(self.gather())

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


class CourtListener(object):
    def __init__(self):
        session_builder(self, "https://www.courtlistener.com/api/rest/v3/", {})

    def options(self, endpoint="", headers={}):
        ep = "https://www.courtlistener.com/api/rest/v3/" + endpoint
        h = {}
        h = safe_merge(h, headers)
        h = safe_merge(h, self.auth_header)
        return requests.options(ep, headers=h).json()

    def find_cite(self, cite):
        return self.request("search/", parameters={"citation": cite})

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

    def search(self, search_header, noisy=False):
        current = self.request("search/", parameters=search_header)
        reslist = []
        while True:
            reslist = reslist + current["results"]
            if current["next"]:
                if noisy:
                    print("requesting: " + current["next"])
                current = self.request(current["next"])
            else:
                break
        return self.extract_case_searches(reslist)

    def fetch_cases_by_cite(self, cite):
        return self.search({"citation": cite})

    def fetch_cases_by_judge(self, judge):
        return self.search({"judge": judge})

    def fetch_cases_by_id(self, id):
        return self.search({"type": "o", "q": "id:{}".format(id)})

    def fetch_cases_cited_by(
        self, c, depth=1
    ):  # can take an Opinion, Case, or Caselist object.
        cases = Caselist([])
        tofetch = frozenset(c.citing())
        newtofetch = set()
        fetched = set()
        while depth > 0:
            for c in tofetch:
                thesecases = self.fetch_cases_by_id(c)
                cases.add(thesecases)
                newtofetch.update(thesecases.citing())
                fetched.add(c)
            tofetch = frozenset(newtofetch.difference(fetched))
            depth -= 1
        return cases


def main():
    # Example usage of the courtlistener class and its methods
    courtlistener_api = CourtListener()
    citation = "347 U.S. 483"  # Example citation
    cases = courtlistener_api.fetch_cases_by_id(107423)
    print(type(cases.cases[0]))
    print((cases.cases[0].opinions[0]))

    # Read the JSON back from the file and convert it into an Opinion object
    # with open("data/cases_output.json", "r") as f:
    #     opinion_data = json.load(f)

    # opinion = Opinion(json.loads(opinion_data), "test", **json.loads(opinion_data))
    # print(opinion)

    # Fetch cases by ID
    # case_id = "12345"  # Example case ID
    # cases_by_id = courtlistener_api.fetch_cases_by_id(case_id)
    # print("Cases by ID:\n", cases_by_id)

    # # Convert cases to pandas DataFrame
    # cases_df = cases_by_cite.to_pandas()
    # print("Cases DataFrame:\n", cases_df.head())


if __name__ == "__main__":
    main()


# need to add more data in case and opinion objects.  also for stuff that might return either a singleton or a list I should just have getter functions that either map over the list or just dispatch for a single, so that it's easy to get results and reports.

# need to provide a facility to dump all cases straight to JSON

# session can have a method to grab cited and such, and pass it either a case or a caselist.  or even a opinion if you really want.  methods on each will return a set.
