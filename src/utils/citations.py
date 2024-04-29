from collections import Counter
from typing import List, Optional, Tuple, Union
import difflib
import numpy as np
import spacy
import pandas as pd
import eyecite
from eyecite.models import (
    CitationBase, 
    CaseCitation, 
    FullCaseCitation, 
    Resource,
)
from citeurl import Citator, insert_links

from src.types import Document


def count_citations(texts: str) -> int:
    all_citations = []
    for text in texts:
        citations = eyecite.get_citations(text)
        citations = [c for c in citations if isinstance(c, FullCaseCitation)]
        all_citations.extend(citations)
    return Counter(all_citations)



def get_top_citations(df, text_column, top_n):
    """
    Extracts citations from a DataFrame column, counts them, and returns a DataFrame
    of the top n citations.
    Args:
        df (pandas.DataFrame): DataFrame containing the legal texts.
        text_column (str): Name of the column containing legal text.
        top_n (int): Number of top citations to return.
    Returns:
        pandas.DataFrame: DataFrame containing the top n citations and their counts.
    """

    # Function to extract citations
    def extract_citations(text):
        citations = extract_resolved_citations(text)
        return citations

    # Extracting all citations from the column
    all_citations = df[text_column].apply(extract_citations).sum()
    # Counting citations
    citation_counts = Counter(all_citations)
    # Getting top n citations
    top_citations = citation_counts.most_common(top_n)
    # Creating a DataFrame for top citations
    top_citations_df = pd.DataFrame(top_citations, columns=["Citation", "Count"])
    links = []
    citator = Citator()
    for i in range(len(top_citations_df)):
        cite = top_citations_df.iloc[i]["Citation"]
        link = citator.list_cites(cite)
        if link:
            url = link[0].URL
            links.append(url)
        else:
            links.append("No link found")
    top_citations_df["link"] = links
    top_citations_df.drop_duplicates(inplace=True)
    return top_citations_df


def create_citation_lookup_table(
    df: pd.DataFrame, text_column: str, id_column: str
) -> pd.DataFrame:
    """
    Creates a citation lookup table from a dataframe by extracting and resolving
        citations from a text column.

    Args:
        df (pd.DataFrame): The input dataframe containing the texts and ids.
        text_column (str): The name of the column in df that contains the text to search for citations.
        id_column (str): The name of the column in df that contains the unique identifier for each text.

    Returns:
        pd.DataFrame: A dataframe with columns for id and citation.
    """
    # Initialize an empty list to store the citation data
    citation_data = []

    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        # Extract the text and id from the current row
        text = row[text_column]
        doc_id = row[id_column]

        # Find citations in the text
        citations = extract_resolved_citations(text)

        # For each citation found, append the id and the citation to the citation_data list
        for citation in citations:
            citation_data.append({"id": doc_id, "citation": str(citation)})

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
    """
    Finds a citation in text and return the context around it, with the start and end positions
    adjusted to complete sentences.

    Args:
        text (str): The text to search.
        citation (str): The citation to find.
        words_before (int | None): Maximum number of words to include before the citation. If None, defaults to 1000.
        words_after (int | None): Maximum number of words to include after the citation. If None, defaults to 1000.

    Returns:
        str: The context around the citation, adjusted to complete sentences.
    """
    if words_after is None and words_before is None:
        # Return entire text since we're not asked to return a bounded context
        return text

    found_citations = eyecite.get_citations(text)

    for cit in found_citations:
        if cit.corrected_citation() == citation:
            match = cit.matched_text()
            sequence_matcher = difflib.SequenceMatcher(None, text, match)
            match_info = sequence_matcher.find_longest_match(0, len(text), 0, len(match))

            if match_info.size == 0:
                return ""

            segments = text.split()
            n_segs = len(segments)

            start_segment_pos = len(text[:match_info.a].split())

            words_before = words_before or n_segs
            words_after = words_after or n_segs
            start_pos = max(0, start_segment_pos - words_before)
            end_pos = min(n_segs, start_segment_pos + words_after + len(citation.split()))

            context = " ".join(segments[start_pos:end_pos])

            # Use spaCy to adjust to complete sentences
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(context)
            sentences = list(doc.sents)

            # Drop the first and last sentences if they are likely incomplete
            adjusted_context = " ".join(sentence.text for sentence in sentences[1:-1])

            return adjusted_context

    # If the citation is not found, return an empty string
    return ""


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


def extract_resolved_citations(text: str) -> List[str]:
    """Extracts a list of resolved citations from text."""
    if text is np.nan:
        return []
    citations = eyecite.get_citations(text, remove_ambiguous=True)
    resolved_list = resolve_citations(citations)
    return resolved_list


def get_citeurl_list(
    input_text: Union[str, pd.DataFrame], text_col: Optional[str] = None
):
    if input_text is np.nan:
        return []
    if isinstance(input_text, pd.DataFrame):
        if text_col is None:
            raise ValueError(
                "text_col must be specified when input_text is a DataFrame."
            )
        input_text = " ".join(input_text[text_col].tolist())
    citator = Citator()
    citations = citator.list_cites(input_text)
    return citations


def get_authorities_list(
    input_text: Union[str, pd.DataFrame], text_col: Optional[str] = None
) -> List[str]:
    """
    Extracts a list of authorities from the given text or DataFrame column.

    Args:
        input_text (Union[str, pd.DataFrame]): The input text or DataFrame containing the text.
        text_col (Optional[str]): The column name in the DataFrame that contains the text. Required if input_text is a DataFrame.

    Returns:
        List[str]: A list of extracted authorities.

    Raises:
        ValueError: If input_text is a DataFrame and text_col is None.
    """
    if input_text is np.nan:
        return []
    if isinstance(input_text, pd.DataFrame):
        if text_col is None:
            raise ValueError(
                "text_col must be specified when input_text is a DataFrame."
            )
        input_text = " ".join(input_text[text_col].tolist())
    citator = Citator()
    authorities = citator.list_authorities(
        input_text,
        ignored_tokens=["subsection", "clause", "pincite", "paragraph"],
        known_authorities=[],
        sort_by_cites=True,
        id_breaks=None,
    )
    return authorities


def make_annotations(
    citations: List[CaseCitation],
) -> List[Tuple[Tuple[int, int], str, str]]:
    """
    Creates annotations from eyecite objects and citeurl mappings.
    Args:
        citations (List[CaseCitation]): Result from eyecite.get_citations.
    Returns:
        List of annotations to insert into text.
    """
    citator = Citator()
    result = []
    for cite in citations:
        if isinstance(cite, CaseCitation):
            citation = cite.corrected_citation_full()
            cite_url = citator.list_cites(citation)
            if cite_url:
                caselaw_url = cite_url[0].URL
                result.append((cite.span(), f'<a href="{caselaw_url}">', "</a>"))
    return result


def create_annotated_text(text: str) -> str:
    if text is np.nan:
        return np.nan
    citations = eyecite.get_citations(text)
    annotations = make_annotations(citations)
    annotated_text_ = eyecite.annotate_citations(text, annotations)
    annotated_text = insert_links(annotated_text_, ignore_markup=False)
    return annotated_text



def create_highlighted_citation(text: str, target_citation: str, html=True) -> str:
    """
    Creates a version of the input text with only the resolved target citation highlighted.

    Args:
        text (str): The original text.
        target_citation (str): The citation string to highlight, which will be resolved.

    Returns:
        str: The text with the resolved target citation highlighted.
    """
    # Extract all citations from the text
    citations: List[CitationBase] = eyecite.get_citations(text, remove_ambiguous=True)
    # Resolve the citations to resources
    resolutions = eyecite.resolve_citations(citations)

    # Resolve the target citation separately to get its standardized form
    target_citations = eyecite.get_citations(target_citation, remove_ambiguous=True)
    resolved_target_citations = eyecite.resolve_citations(target_citations)
    resolved_target_citation_strings = set()
    for resource, cites in resolved_target_citations.items():
        if isinstance(resource, Resource):
            for citation in cites:
                resolved_target_citation_strings.add(citation.corrected_citation())

    annotations = []

    # Iterate through the resolved citations in the text
    for resource, cites in resolutions.items():
        if isinstance(resource, Resource):
            for citation in cites:
                # Check if the resolved citation matches any resolved form of the target citation
                if citation.corrected_citation() in resolved_target_citation_strings:
                    if html:
                        annotations.append((citation.span(), "<mark>", "</mark>"))
                    else:
                        annotations.append((citation.span(), "**", "**"))

    # Annotate the text with highlights for the resolved target citation
    annotated_text = eyecite.annotate_citations(text, annotations)
    return annotated_text


def extract_citation_with_context_from_docs(
    docs: List[Document],
    target_citation: str,
    sentences_before: int,
    sentences_after: int,
) -> List[Document]:
    """
    Extracts a target citation from a list of Document objects and provides context around each citation,
    creating new Document objects with extracted text as content and remaining data as metadata.

    Args:
        docs (List[Document]): The list of Document objects to process.
        target_citation: (str): The target citation to search
        sentences_before (int): The number of sentences before the citation to include in the context.
        sentences_after (int): The number of sentences after the citation to include in the context.

    Returns:
        List[Document]: A list of new Document objects with extracted text as content and remaining data as metadata.
    """
    nlp = spacy.load("en_core_web_sm")
    new_docs = []

    for doc in docs:
        text = doc.content
        original_metadata = doc.metadata
        nlp_doc = nlp(text)
        sentences = list(nlp_doc.sents)

        citations = eyecite.get_citations(text, remove_ambiguous=True)

        for citation in citations:
            citation_str = citation.corrected_citation()
            if citation_str == target_citation:
                span = citation.span()
                start_char, end_char = span
                # Find the sentence containing the citation
                containing_sentence = next(
                    (
                        sent
                        for sent in sentences
                        if sent.start_char <= start_char and sent.end_char >= end_char
                    ),
                    None,
                )
                if containing_sentence:
                    # Expand to include a few sentences before and after, if available
                    sent_index = sentences.index(containing_sentence)
                    start_index = max(0, sent_index - sentences_before)
                    end_index = min(
                        len(sentences), sent_index + sentences_after + 1
                    )  # includes the current
                    context = " ".join(
                        [str(sent) for sent in sentences[start_index:end_index]]
                    )
                    start_char_context = (
                        sentences[start_index].start_char
                        if start_index < len(sentences)
                        else 0
                    )
                    end_char_context = (
                        sentences[end_index - 1].end_char
                        if end_index > 0
                        else len(text)
                    )
                    new_doc_metadata = {
                        **original_metadata,  # Include all original metadata
                        "parent_id": original_metadata.get("id", ""),
                        "citation": citation_str,
                        "start_char": start_char_context,
                        "end_char": end_char_context,
                    }
                    new_docs.append(
                        Document(content=context, metadata=new_doc_metadata)
                    )

    return new_docs


def get_citation_context_df(
    df: pd.DataFrame,
    text_column: str,
    id_column: str,
    sentences_before: int,
    sentences_after: int,
    target_citations: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extracts citations from a dataframe and provides context around each citation. Optionally,
    limits extraction to a specified list of citations.

    Args:
        df (pd.DataFrame): The dataframe containing the documents to process.
        text_column (str): The name of the column containing the text.
        id_column (str): The name of the column containing the document ID.
        sentences_before (int): The number of sentences before the citation to include in the context.
        sentences_after (int): The number of sentences after the citation to include in the context.
        target_citations (Optional[List[str]]): A list of citations to extract. If None, all citations are extracted.

    Returns:
        pd.DataFrame: A dataframe with columns for document ID, citation, and context around the citation.
    """
    nlp = spacy.load("en_core_web_sm")
    exclude_list = [
        "§",
        "Id.",
        "Id.,",
        "id.,",
        "id.",
        "supra",
        "Ibid.",
        "§§",
        "[§",
        "supra,",
        "(§",
    ]
    citation_data = []

    for _, row in df.iterrows():
        text = row[text_column]
        doc_id = row[id_column]
        doc = nlp(text)
        sentences = list(doc.sents)

        citations = eyecite.get_citations(text, remove_ambiguous=True)

        for citation in citations:
            citation_str = citation.corrected_citation()
            if target_citations is None or citation_str in target_citations:
                if citation_str not in exclude_list:
                    span = citation.span()
                    start_char, end_char = span
                    # Find the sentence containing the citation
                    containing_sentence = next(
                        (
                            sent
                            for sent in sentences
                            if sent.start_char <= start_char
                            and sent.end_char >= end_char
                        ),
                        None,
                    )
                    if containing_sentence:
                        # Expand to include a few sentences before and after, if available
                        sent_index = sentences.index(containing_sentence)
                        start_index = max(0, sent_index - sentences_before)
                        end_index = min(
                            len(sentences), sent_index + sentences_after
                        )  # includes the current
                        context = " ".join(
                            [str(sent) for sent in sentences[start_index:end_index]]
                        )
                        start_char_context = (
                            sentences[start_index].start_char
                            if start_index < len(sentences)
                            else 0
                        )
                        end_char_context = (
                            sentences[end_index - 1].end_char
                            if end_index > 0
                            else len(text)
                        )
                        citation_data.append(
                            {
                                "id": doc_id,
                                "citation": citation_str,
                                "context": context,
                                "start_char": start_char_context,
                                "end_char": end_char_context,
                            }
                        )

    citation_df = pd.DataFrame(
        citation_data, columns=["id", "citation", "context", "start_char", "end_char"]
    )
    citation_df.drop_duplicates(inplace=True)
    return citation_df

def add_citation_context_to_df(
    df: pd.DataFrame, 
    text_column: str, 
    citation_column: str, 
    words_before: int = 512, 
    words_after: int = 512,
) -> pd.DataFrame:
    """
    Loops over the given DataFrame and applies `get_citation_context` for the specified citation.
    Adds the resulting context string to the DataFrame as a new column.

    Args:
        df (pd.DataFrame): The DataFrame to process, expected to contain a 'body' column.
        citation (str): The citation to search for within the 'body' column.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'context' column containing the citation context.
    """
    # Ensure the 'context' column exists
    if 'context' not in df.columns:
        df['context'] = ''

    # Apply the get_citation_context function row-wise
    for index, row in df.iterrows():
        cite = row[citation_column]
        context = get_citation_context(
            text=row[text_column], 
            citation=cite, 
            words_before=words_before, 
            words_after=words_after,
        )
        df.at[index, 'context'] = context
        df.at[index, 'context_citation'] = cite

    return df

def filter_frequent_citations(citation_table: pd.DataFrame, min_occurrences: int = 10) -> pd.DataFrame:
    """
    Filters the citation table to only include citations that appear at least a specified number of times.
    
    Args:
        citation_table (pd.DataFrame): The citation table to filter.
        min_occurrences (int): The minimum number of occurrences for a citation to be included.
        
    Returns:
        pd.DataFrame: A filtered DataFrame containing only the citations that meet the minimum occurrence criterion.
    """
    # Count the occurrences of each citation
    citation_counts = citation_table['citation'].value_counts()
    
    # Filter for citations that occur at least min_occurrences times
    frequent_citations = citation_counts[citation_counts >= min_occurrences].index
    
    # Filter the original citation table to only include these citations
    filtered_citation_table = citation_table[citation_table['citation'].isin(frequent_citations)]
    
    return filtered_citation_table
