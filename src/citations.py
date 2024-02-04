from collections import Counter
from typing import List
import difflib
import pandas as pd
import eyecite
from eyecite.models import CitationBase


def count_citations(texts: str) -> int:
    all_citations = []
    for text in texts:
        citations = eyecite.get_citations(text)
        all_citations.extend(citations)
    return Counter(all_citations)


def get_top_citations(df: pd.DataFrame, text_column: str, top_n: int) -> pd.DataFrame:
    """
    Extracts citations from a DataFrame column, counts them, and returns a DataFrame
    of the top n citations.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the legal texts.
    text_column (str): Name of the column containing legal text.
    top_n (int): Number of top citations to return.

    Returns:
    pandas.DataFrame: DataFrame containing the top n citations and their counts.
    """

    # Function to extract citations
    def extract_citations(text):
        citations = eyecite.get_citations(text, remove_ambiguous=True)
        return [str(cite.corrected_citation_full()) for cite in citations]

    # Extracting all citations from the column
    all_citations = df[text_column].apply(extract_citations).sum()
    # Counting citations
    citation_counts = Counter(all_citations)
    # Getting top n citations
    top_citations = citation_counts.most_common(top_n)
    # Creating a DataFrame for top citations
    top_citations_df = pd.DataFrame(top_citations, columns=["Citation", "Count"])
    return top_citations_df


def create_citation_lookup_table(
    df: pd.DataFrame, text_column: str, id_column: str
) -> pd.DataFrame:
    """
    Creates a citation lookup table from a dataframe by finding distinct citations in a specified text column.

    Args:
        df (pd.DataFrame): The input dataframe containing the texts and ids.
        text_column (str): The name of the column in df that contains the text to search for citations.
        id_column (str): The name of the column in df that contains the unique identifier for each text.

    Returns:
        pd.DataFrame: A dataframe with columns for id and citation.
    """
    exclude_list = ["§", "Id.", "Id.,", "id.,", "id.", "supra", "Ibid."]
    # Initialize an empty list to store the citation data
    citation_data = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Extract the text and id from the current row
        text = row[text_column]
        doc_id = row[id_column]

        # Find citations in the text
        citations = eyecite.get_citations(text, remove_ambiguous=True)

        # For each citation found, append the id and the citation to the citation_data list
        for citation in citations:
            if citation.corrected_citation() not in exclude_list:
                citation_data.append(
                    {"id": doc_id, "citation": str(citation.corrected_citation())}
                )

    # Convert the citation_data list to a DataFrame
    citation_df = pd.DataFrame(citation_data, columns=["id", "citation"])
    citation_df.drop_duplicates(inplace=True)

    return citation_df


def get_citation_context(
    text: str,
    citation: str,
    words_before: int | None = 1000,
    words_after: int | None = 1000,
) -> str:
    if words_after is None and words_before is None:
        # return entire text since we're not asked to return a bounded context
        return text, 0, 0

    found_citations = eyecite.get_citations(text)

    for cit in found_citations:
        if cit.corrected_citation() == citation:
            match = cit.matched_text()
            sequence_matcher = difflib.SequenceMatcher(None, text, match)
            match = sequence_matcher.find_longest_match(0, len(text), 0, len(match))

            if match.size == 0:
                return "", 0, 0

            segments = text.split()
            n_segs = len(segments)

            start_segment_pos = len(text[: match.a].split())

            words_before = words_before or n_segs
            words_after = words_after or n_segs
            start_pos = max(0, start_segment_pos - words_before)
            end_pos = min(
                len(segments), start_segment_pos + words_after + len(citation.split())
            )
            result = " ".join(segments[start_pos:end_pos]), start_pos, end_pos

    return result[0]


def resolve_citations(citations: List[CitationBase]) -> List[str]:
    resolutions = eyecite.resolve_citations(citations)
    resolved_list = list(resolutions.keys())
    resolved = []
    for i in range(len(resolved_list)):
        try:
            cite = resolved_list[i].citation
            citation_str = cite.corrected_citation()
            resolved.append(citation_str)
        except:
            pass
    return resolved
