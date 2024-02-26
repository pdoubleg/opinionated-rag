


from datetime import datetime
import os
from dotenv import load_dotenv
from pydantic import ConfigDict
import requests
from dateutil import parser
from src.schema.docket import Docket
from src.schema.opinion import Opinion

from src.doc_store.base import DocMetaData, LegalDataSource, LegalDocument, APICommunicationError
from src.doc_store.utils import disassemble, get_chain, looks_like_citation, safe_eager_map


COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v3/"
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY")

load_dotenv()


class CourtListenerCaseDataSource(LegalDataSource):
    
    model_config = ConfigDict(
        json_schema_extra={'details': {
        "name": "CourtListener",
        "short_description": "hello",
        "long_description": "CourtListener searches millions of opinions across hundreds of jurisdictions",
        "link": COURTLISTENER_BASE_URL,
        "search_regexes": [],
        "footnote_regexes": [],
    }})

    def search(self, search_params):

        if not COURTLISTENER_API_KEY:
            raise APICommunicationError("A CourtListener API key is required")
        try: 
            params = (
                {"citation": search_params}
                if looks_like_citation(search_params)
                else {"q": search_params}
            )
            response = requests.get(
                f"{COURTLISTENER_BASE_URL}/api/rest/v3/search",
                params=params,
                headers={"Authorization": f"Token {COURTLISTENER_API_KEY}"},
            )
            response.raise_for_status()
        except (requests.exceptions.HTTPError) as e:
            msg = f"Communication with CourtListener failed: {str(e), response.status_code, response.request.url}"
            raise APICommunicationError(msg)
        
        results = []

        for r in response.json()["results"]:
            body = get_chain(
                r,
                [
                    "html_with_citations",
                    "html_columbia",
                    "html_lawbox",
                    "xml_harvard",
                    "html",
                ],
            )
            if "opinions" in r and r["opinions"]:
                opinions = [Opinion(op, **op) for op in r["opinions"]]
            else:
                opinions = []
                
            people = {
                "panel": r.get("panel"),
                "non_participating_judges": r.get("non_participating_judges"),
                "judges": get_chain(r, ["judges", "judge"]),
                "attorneys": get_chain(r, ["attorneys", "attorney"]),
            }
            
            result = LegalDocument(
                case_id=r["id"],
                source_id=r["id"],
                data_source=self,
                name=get_chain(r, ["case_name", "case_name_full", "caseName"]),
                name_short=get_chain(r, ["caseNameShort", "caseName", "case_name"]),
                url=f"{COURTLISTENER_BASE_URL}{r['absolute_url']}",               
                conetent=body,
                doc_class="Case",
                citations=safe_eager_map(
                    lambda x: disassemble(x)["identifier"], r.get("opinions_cited", [])),
                court=get_chain(r, ["court_citation_string", "court_exact", "court", "court_id"]),
                publication_date=parser.isoparse(r["dateFiled"]).strftime("%Y-%m-%d"),
                opinions=opinions,
                people=people,
                courtlistener_cluster = r.get("cluster_id"),
                courtlistener_docket = r.get("docket"),
                docket = Docket(**r),
                )
            
            results.append(result)
        return results


    def fetch_case(self, case_id: int):
        if not COURTLISTENER_API_KEY:
            raise APICommunicationError("A CourtListener API key is required")
        try:
            response = requests.get(
                f"{COURTLISTENER_BASE_URL}/api/rest/v3/clusters/{case_id}/",
                headers={"Authorization": f"Token {COURTLISTENER_API_KEY}"},
            )
            response.raise_for_status()
            cluster = response.json()
            response = requests.get(
                f"{COURTLISTENER_BASE_URL}/api/rest/v3/opinions/{case_id}/",
                headers={"Authorization": f"Token {os.getenv('COURTLISTENER_API_KEY')}"},
            )
            response.raise_for_status()

            opinion = response.json()

        except (requests.exceptions.HTTPError) as e:
            msg = f"Failed call to {response.request.url}: {e}\n{response.content}"
            raise APICommunicationError(msg)

        # body = opinion["html"]
        body = get_chain(
            opinion,
            [
                "html_with_citations",
                "html_columbia",
                "html_lawbox",
                "xml_harvard",
                "html",
            ],
        )
        case = LegalDocument(
            case_id=cluster["id"],
            source_id=opinion["sha1"],
            data_source=self,
            name=get_chain(cluster, ["case_name", "case_name_full", "caseName"]),
            name_short=get_chain(cluster, ["caseNameShort", "case_name_short", "case_name"]),
            url=f"{COURTLISTENER_BASE_URL}{cluster['absolute_url']}",               
            conetent=body,
            doc_class="Case",
            citations=safe_eager_map(
                lambda x: disassemble(x)["identifier"], opinion.get("opinions_cited", [])),
            court=get_chain(opinion, ["court_citation_string", "court_exact", "court", "court_id"]),
            publication_date=parser.isoparse(cluster["dateFiled"]).strftime("%Y-%m-%d"),
            metadata=DocMetaData(
                source=cluster["id"],
                slug=cluster["slug"],
                download_url=opinion["download_url"],
                resource_url=opinion["resource_url"],
            )
            )
        return case


    def fetch_forward_citations(self, case_id: int) -> list:
        """
        Retrieves a list of forward citation IDs for a given case ID.

        Args:
            case_id (int): The ID of the case for which to find forward citations.

        Returns:
            list: A list of IDs representing the forward citations.

        Raises:
            APICommunicationError: If there's an issue communicating with the CourtListener API.
        """
        if not os.getenv('COURTLISTENER_API_KEY'):
            raise APICommunicationError("A CourtListener API key is required")

        ep = f"https://www.courtlistener.com/api/rest/v3/search/?q=cites%3A({case_id})&type=o&order_by=dateFiled%20asc&stat_Precedential=on"
        h = {"Authorization": f"Token {os.getenv('COURTLISTENER_API_KEY')}"}

        try:
            cites_from_id = requests.get(ep, headers=h).json()
        except requests.exceptions.HTTPError as e:
            msg = f"Failed to retrieve forward citations for case ID {case_id}: {e}\n{cites_from_id.content}"
            raise APICommunicationError(msg)

        forward_citations = [result for result in cites_from_id.get("results", [])]
        return forward_citations
    
    def get_bluebook_citation(self, case_id: int) -> str:
        case_data = self.fetch_case(case_id)
        case_name = case_data.name_short
        reporter_info = case_data.citations[0]
        court = case_data.court
        decision_date = case_data.publication_date

        return f"{case_name}, {reporter_info} ({court} {decision_date})"
    
def main():
    # Example usage of the courtlistener class and its methods
    cl = CourtListenerCaseDataSource()
    citation = "120 Cal. Rptr. 2d 162"  # Example citation
    # Sample case id: 107423
    test = cl.search(2276381)
    print(test)


if __name__ == "__main__":
    main()
