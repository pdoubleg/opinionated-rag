from collections import Counter
from typing import List, Tuple
import difflib
import numpy as np
import spacy
import pandas as pd
import eyecite
from eyecite import get_citations, annotate_citations
from eyecite.models import CitationBase, CaseCitation
from citeurl import Citator, insert_links

from src.types import Document


def count_citations(texts: str) -> int:
    all_citations = []
    for text in texts:
        citations = eyecite.get_citations(text)
        all_citations.extend(citations)
    return Counter(all_citations)


def get_citation_context_sents(
    text: str,
    citation: str,
    sentences_before: int = 2,
    sentences_after: int = 2,
) -> str:
    """
    Finds a citation in text and returns the sentence or sentences surrounding it.
    Args:
        text (str): The text to search.
        citation (str): The citation to find.
        sentences_before (int): n sentences to extract before citation sentence.
        sentences_after (int): n sentences to extract after citation sentence.

    Returns:
        sentences (str): The extracted sentences.
    """
    # Find citations using eyecite
    found_citations = eyecite.get_citations(text)
    # Tokenize text into sentences using spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = list(doc.sents)
    # Iterate over citations and sentences to find the match
    for cit in found_citations:
        if cit.matched_text() == citation:
            cit_span = cit.full_span()
            # Find the sentence index that contains the citation
            cit_sentence_idx = None
            for i, sentence in enumerate(sentences):
                if (
                    sentence.start_char <= cit_span[0]
                    and sentence.end_char >= cit_span[1]
                ):
                    cit_sentence_idx = i
                    break
            if cit_sentence_idx is not None:
                # Extract the specified number of sentences before and after the citation
                start_idx = max(cit_sentence_idx - sentences_before, 0)
                end_idx = min(cit_sentence_idx + sentences_after + 1, len(sentences))
                # Joining the extracted sentences
                context = " ".join(sentences[i].text for i in range(start_idx, end_idx))
                return context

    return None


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
        citations = eyecite.get_citations(text)
        return [str(cite.corrected_citation_full()) for cite in citations]

    # Extracting all citations from the column
    all_citations = df[text_column].apply(extract_citations).sum()
    # Filter in-line references
    exclude_list = ["Id.", "§", "id.", "supra,", "supra."]
    filtered_citations = [cite for cite in all_citations if cite not in exclude_list]
    # Counting citations
    citation_counts = Counter(filtered_citations)
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


def get_citation_context_words(
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

def extract_citations(text: str) -> List[str]:
    """Extracts a list of full name citations from text"""
    if text is np.nan:
        return []
    citations = eyecite.get_citations(text)
    return [str(cite.corrected_citation_full()) for cite in citations]

 
def get_citeurl_list(text: str):
    if text is np.nan:
        return []
    citator = Citator()
    citations = citator.list_cites(text)
    return citations

 
def get_authorities_list(text: str):
    if text is np.nan:
        return []
    citator = Citator()
    authorities = citator.list_authorities(
        text,
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
    citations = get_citations(text)
    annotations = make_annotations(citations)
    annotated_text_ = annotate_citations(text, annotations)
    annotated_text = insert_links(annotated_text_, ignore_markup=False)
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
                containing_sentence = next((sent for sent in sentences if sent.start_char <= start_char and sent.end_char >= end_char), None)
                if containing_sentence:
                    # Expand to include a few sentences before and after, if available
                    sent_index = sentences.index(containing_sentence)
                    start_index = max(0, sent_index - sentences_before)
                    end_index = min(len(sentences), sent_index + sentences_after + 1)  # includes the current
                    context = " ".join([str(sent) for sent in sentences[start_index:end_index]])
                    start_char_context = sentences[start_index].start_char if start_index < len(sentences) else 0
                    end_char_context = sentences[end_index - 1].end_char if end_index > 0 else len(text)
                    new_doc_metadata = {
                        **original_metadata,  # Include all original metadata
                        "parent_id": original_metadata.get("id", ""),
                        "citation": citation_str,
                        "start_char": start_char_context,
                        "end_char": end_char_context,
                    }
                    new_docs.append(Document(content=context, metadata=new_doc_metadata))

    return new_docs


def extract_citations_with_context_from_df(
    df: pd.DataFrame, 
    text_column: str, 
    id_column: str, 
    sentences_before: int, 
    sentences_after: int,
    ) -> pd.DataFrame:
    """
    Extracts citations from a dataframe and provides context around each citation.
    
    Args:
        df (pd.DataFrame): The dataframe containing the documents to process.
        text_column (str): The name of the column containing the text.
        id_column (str): The name of the column containing the document ID.
        sentences_before (int): The number of sentences before the citation to include in the context.
        sentences_after (int): The number of sentences after the citation to include in the context.
    
    Returns:
        pd.DataFrame: A dataframe with columns for document ID, citation, and context around the citation.
    """
    nlp = spacy.load("en_core_web_sm")
    exclude_list = ['§', 'Id.', 'Id.,', 'id.,', 'id.', 'supra', 'Ibid.', '§§', '[§', 'supra,', '(§']
    citation_data = []

    for _, row in df.iterrows():
        text = row[text_column]
        doc_id = row[id_column]
        doc = nlp(text)
        sentences = list(doc.sents)
        
        citations = eyecite.get_citations(text, remove_ambiguous=True)

        for citation in citations:
            citation_str = citation.corrected_citation()
            if citation_str not in exclude_list:
                span = citation.span()
                start_char, end_char = span
                # Find the sentence containing the citation
                containing_sentence = next((sent for sent in sentences if sent.start_char <= start_char and sent.end_char >= end_char), None)
                if containing_sentence:
                    # Expand to include a few sentences before and after, if available
                    sent_index = sentences.index(containing_sentence)
                    start_index = max(0, sent_index - sentences_before)
                    end_index = min(len(sentences), sent_index + sentences_after)  # includes the current
                    context = " ".join([str(sent) for sent in sentences[start_index:end_index]])
                    start_char_context = sentences[start_index].start_char if start_index < len(sentences) else 0
                    end_char_context = sentences[end_index - 1].end_char if end_index > 0 else len(text)
                    citation_data.append({
                        "id": doc_id,
                        "citation": citation_str,
                        "context": context,
                        "start_char": start_char_context,
                        "end_char": end_char_context,
                    })

    citation_df = pd.DataFrame(citation_data, columns=["id", "citation", "context", "start_char", "end_char"])

    return citation_df

