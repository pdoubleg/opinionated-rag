import pandas as pd
import backoff
import requests
from datetime import datetime
import eyecite, citeurl
import time
from tqdm import tqdm


class CourtListenerExtractor:
    def __init__(self, base_url="https://www.courtlistener.com/api/rest/v3/opinions/"):
        """
        Initializes the CourtListenerExtractor with the path to the database file.

        Args:
        database_path (str): The path to the Excel file database.
        """
        self.base_url = base_url

    def fetch_records(self, start_page, end_page):
        """
        Fetches records from Court Listener based on a page range.

        Args:
        start_page (int): court listener web search start
        end_page (int): court listener web search start

        Returns:
        pd.DataFrame: A DataFrame containing the fetched records.
        """
        all_results = []
        for page in tqdm(range(start_page, end_page + 1)):
            url = f"{self.base_url}?page={page}"
            data = self.fetch_page_(url)
            all_results.extend(data["results"])
            time.sleep(1)
        df = pd.DataFrame.from_records(all_results)
        return df

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.HTTPError),
        max_tries=8,
    )
    def fetch_page_(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


# Example usage:
# extractor = CourtListenerExtractor()
# new_records = extractor.fetch_records(1, 2)
# print(new_records)
