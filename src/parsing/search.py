"""
Utils to search for close matches in (a list of) strings.
Useful for retrieval of docs/chunks relevant to a query, in the context of
Retrieval-Augmented Generation (RAG), and SQLChat (e.g., to pull relevant parts of a
large schema).
"""

import difflib
import hashlib
import pandas as pd
from typing import List, Tuple, Optional

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from rank_bm25 import BM25Okapi
from thefuzz import fuzz, process

from src.types import Document

from .utils import download_nltk_resource

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)



def find_fuzzy_matches_in_df(
    query: str,
    df: pd.DataFrame,
    text_column: str,
    k: int = 20,
    words_before: Optional[int] = 10,
    words_after: Optional[int] = 10,
) -> pd.DataFrame:
    """
    Find approximate matches of the query in the DataFrame and return surrounding
    characters.

    Args:
        query (str): The search string.
        df (pd.DataFrame): The input DataFrame
        k (int): Number of best matches to return.
        words_before (Optional[int]): Number of words to include before each match.
            Default None => return max
        words_after (Optional[int]): Number of words to include after each match.
            Default None => return max

    Returns:
        pd.DataFrame: DataFrame of matches, including the given number of words around the match.
    """
    if df.empty:
        return pd.DataFrame()
    
    df["id"] = df[text_column].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()
    )

    best_matches = process.extract(
        query,
        df[text_column].tolist(),
        limit=k,
        scorer=fuzz.partial_ratio,
    )
    match_ids_scores = []
    for m in best_matches:
        if m[1] > 50:
            # Find the index in the DataFrame where the text matches
            matched_index = df.index[df[text_column] == m[0]].tolist()
            # Assuming the match is unique and exists, append the corresponding 'id' and score to match_ids_scores
            if matched_index:
                match_ids_scores.append((df.loc[matched_index[0], 'id'], m[1]))

    # Create a DataFrame from match_ids_scores
    matches_df = pd.DataFrame(match_ids_scores, columns=['id', 'score'])

    # Merge the original DataFrame with the matches DataFrame on 'id' to include the scores
    orig_doc_matches = pd.merge(df, matches_df, on='id', how='inner')

    if words_after is None and words_before is None:
        return orig_doc_matches

    def get_context_for_row(row):
        """
        Extracts context for each match in a row, starting with a base threshold and increasing
        it for each subsequent extraction to prioritize high-quality matches.

        Args:
            row (pd.Series): A row from the DataFrame containing the text and match score.

        Returns:
            str: A string containing the contexts of good matches separated by " ... ".
        """
        choice_text = row[text_column]
        contexts = []
        # Initial threshold for a match to be considered good
        threshold = 50
        # Amount to increase the threshold by after each good match is found
        threshold_increment = 10
        while choice_text != "":
            context, start_pos, end_pos = get_context(
                query, choice_text, words_before, words_after
            )
            if context == "" or end_pos == 0:
                break
            # Check if the context contains a good match by using a scorer, e.g., fuzz.partial_ratio
            match_quality = fuzz.partial_ratio(query, context)
            if match_quality > threshold:
                contexts.append(context)
                # Increase the threshold for the next match, making it harder to add additional contexts
                threshold += threshold_increment
            words = choice_text.split()
            end_pos = min(end_pos, len(words))
            choice_text = " ".join(words[end_pos:])
        return " ... ".join(contexts)

    # Apply context extraction
    orig_doc_matches['context'] = orig_doc_matches.apply(get_context_for_row, axis=1)
    orig_doc_matches.sort_values(by='score', ascending=False, inplace=True)
    return orig_doc_matches


def get_context(
    query: str,
    text: str,
    words_before: int | None = 100,
    words_after: int | None = 100,
) -> Tuple[str, int, int]:
    """
    Returns a portion of text surrounding the best approximate match of the query.

    This function searches for the query within the given text and returns a specified number of words before and after the best match. If no match is found, or if the match quality is below a certain threshold, an empty string and zero positions are returned.

    Args:
        query (str): The string to search for within the text.
        text (str): The body of text in which to search for the query.
        words_before (int | None, optional): The number of words before the query to include in the returned context. Defaults to 100.
        words_after (int | None, optional): The number of words after the query to include in the returned context. Defaults to 100.

    Returns:
        Tuple[str, int, int]: A tuple containing the context string (words before, the match, and words after the best match), the start position, and the end position of the match within the text. If no match is found, returns an empty string and zeros for the positions.

    Example:
        >>> get_context("apple", "The quick brown fox jumps over the lazy dog.", 3, 2)
        ('fox jumps over the lazy dog', 4, 9)
    """
    if words_after is None and words_before is None:
        # return entire text since we're not asked to return a bounded context
        return text, 0, 0

    # make sure there is a good enough match to the query
    if fuzz.partial_ratio(query, text) < 50:
        return "", 0, 0

    sequence_matcher = difflib.SequenceMatcher(None, text, query)
    match = sequence_matcher.find_longest_match(0, len(text), 0, len(query))

    if match.size == 0:
        return "", 0, 0

    segments = text.split()
    n_segs = len(segments)

    start_segment_pos = len(text[: match.a].split())

    words_before = words_before or n_segs
    words_after = words_after or n_segs
    start_pos = max(0, start_segment_pos - words_before)
    end_pos = min(len(segments), start_segment_pos + words_after + len(query.split()))

    return " ".join(segments[start_pos:end_pos]), start_pos, end_pos


def find_fuzzy_matches_in_docs(
    query: str,
    docs: List[Document],
    docs_clean: List[Document],
    k: int,
    words_before: int | None = None,
    words_after: int | None = None,
) -> List[Document]:
    """
    Find approximate matches of the query in the docs and return surrounding
    characters.

    Args:
        query (str): The search string.
        docs (List[Document]): List of Document objects to search through.
        k (int): Number of best matches to return.
        words_before (int|None): Number of words to include before each match.
            Default None => return max
        words_after (int|None): Number of words to include after each match.
            Default None => return max

    Returns:
        List[Document]: List of Documents containing the matches,
            including the given number of words around the match.
    """
    if len(docs) == 0:
        return []
    best_matches = process.extract(
        query,
        [d.content for d in docs_clean],
        limit=k,
        scorer=fuzz.partial_ratio,
    )

    real_matches = [m for m, score in best_matches if score > 50]
    # find the original docs that corresponding to the matches
    orig_doc_matches = []
    for i, m in enumerate(real_matches):
        for j, doc_clean in enumerate(docs_clean):
            if m in doc_clean.content:
                orig_doc_matches.append(docs[j])
                break
    if words_after is None and words_before is None:
        return orig_doc_matches

    contextual_matches = []
    for match in orig_doc_matches:
        choice_text = match.content
        contexts = []
        while choice_text != "":
            context, start_pos, end_pos = get_context(
                query, choice_text, words_before, words_after
            )
            if context == "" or end_pos == 0:
                break
            contexts.append(context)
            words = choice_text.split()
            end_pos = min(end_pos, len(words))
            choice_text = " ".join(words[end_pos:])
        if len(contexts) > 0:
            contextual_matches.append(
                Document(
                    content=" ... ".join(contexts),
                    metadata=match.metadata,
                )
            )

    return contextual_matches


def preprocess_text(text: str) -> str:
    """
    Preprocesses the given text by:
    1. Lowercasing all words.
    2. Tokenizing (splitting the text into words).
    3. Removing punctuation.
    4. Removing stopwords.
    5. Lemmatizing words.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text.
    """
    # Ensure the NLTK resources are available
    for resource in ["punkt", "wordnet", "stopwords"]:
        download_nltk_resource(resource)

    # Lowercase the text
    text = text.lower()

    # Tokenize the text and remove punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # Join the words back into a string
    text = " ".join(tokens)

    return text


def find_closest_matches_with_bm25_df(
    df: pd.DataFrame,
    text_column: str,
    query: str,
    k: int = 5,
) -> pd.DataFrame:
    """
    Finds the k closest approximate matches using the BM25 algorithm in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the Documents to search through.
        text_column (str): The name of the column in the DataFrame containing the text.
        query (str): The search query.
        k (int, optional): Number of matches to retrieve. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame of the top k matches with an additional 'score' column.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Pre-process query and search space text for bm25
    df['clean_text'] = df[text_column].apply(preprocess_text)
    texts = df['clean_text'].tolist()
    query = preprocess_text(query)

    text_words = [text.split() for text in texts]
    bm25 = BM25Okapi(text_words)
    
    query_words = query.split()
    doc_scores = bm25.get_scores(query_words)

    # Get indices of top k scores
    top_indices = sorted(range(len(doc_scores)), key=lambda i: -doc_scores[i])[:k]

    # Select the top k documents based on the scores and add a 'score' column
    top_docs = df.iloc[top_indices].copy()
    top_docs['score'] = [doc_scores[i] for i in top_indices]

    return top_docs


def find_closest_matches_with_bm25(
    docs: List[Document],
    docs_clean: List[Document],
    query: str,
    k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Finds the k closest approximate matches using the BM25 algorithm.

    Args:
        docs (List[Document]): List of Documents to search through.
        docs_clean (List[Document]): List of cleaned Documents
        query (str): The search query.
        k (int, optional): Number of matches to retrieve. Defaults to 5.

    Returns:
        List[Tuple[Document,float]]: List of (Document, score) tuples.
    """
    if len(docs) == 0:
        return []
    texts = [doc.content for doc in docs_clean]
    query = preprocess_text(query)

    text_words = [text.split() for text in texts]

    bm25 = BM25Okapi(text_words)
    query_words = query.split()
    doc_scores = bm25.get_scores(query_words)

    # Get indices of top k scores
    top_indices = sorted(range(len(doc_scores)), key=lambda i: -doc_scores[i])[:k]

    # return the original docs, based on the scores from cleaned docs
    return [(docs[i], doc_scores[i]) for i in top_indices]


def eliminate_near_duplicates(passages: List[str], threshold: float = 0.8) -> List[str]:
    """
    Eliminate near duplicate text passages from a given list using MinHash and LSH.
    Args:
        passages (List[str]): A list of text passages.
        threshold (float, optional): Jaccard similarity threshold to consider two
                                     passages as near-duplicates. Default is 0.8.

    Returns:
        List[str]: A list of passages after eliminating near duplicates.

    Example:
        passages = ["Hello world", "Hello, world!", "Hi there", "Hello world!"]
        print(eliminate_near_duplicates(passages))
        # ['Hello world', 'Hi there']
    """

    from datasketch import MinHash, MinHashLSH

    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # Create MinHash objects for each passage and insert to LSH
    minhashes = {}
    for idx, passage in enumerate(passages):
        m = MinHash(num_perm=128)
        for word in passage.split():
            m.update(word.encode("utf-8"))
        lsh.insert(idx, m)
        minhashes[idx] = m

    unique_idxs = set()
    for idx in minhashes.keys():
        # Query for similar passages (including itself)
        result = lsh.query(minhashes[idx])

        # If only the passage itself is returned, it's unique
        if len(result) == 1 and idx in result:
            unique_idxs.add(idx)

    return [passages[idx] for idx in unique_idxs]


def eliminate_near_duplicates_df(
    df: pd.DataFrame, column: str, threshold: float = 0.9
) -> pd.DataFrame:
    """
    Eliminate near duplicate text passages from a given DataFrame column using MinHash and LSH.

    Args:
        df (pd.DataFrame): DataFrame containing the text passages.
        column (str): The column of the DataFrame containing the text passages.
        threshold (float, optional): Jaccard similarity threshold to consider two
                                     passages as near-duplicates. Default is 0.9.

    Returns:
        pd.DataFrame: DataFrame after eliminating near duplicates in the specified column.
    """

    from datasketch import MinHash, MinHashLSH

    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # Create MinHash objects for each passage and insert to LSH
    minhashes = {}
    for idx, passage in df[column].items():
        m = MinHash(num_perm=128)
        for word in str(passage).split():
            m.update(word.encode("utf-8"))
        lsh.insert(idx, m)
        minhashes[idx] = m

    unique_idxs = set()
    for idx in minhashes.keys():
        # Query for similar passages (including itself)
        result = lsh.query(minhashes[idx])

        # If only the passage itself is returned, it's unique
        if len(result) == 1 and idx in result:
            unique_idxs.add(idx)

    return df.loc[list(unique_idxs)]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def eliminate_near_duplicates_grouped(
    df: pd.DataFrame, text_column: str, group_columns: list, threshold: float = 0.9
) -> pd.DataFrame:
    """
    Groups the DataFrame by specified columns and eliminates near-duplicate rows within each group based on text similarity.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        text_column (str): The column containing text for duplication check.
        group_columns (list): Columns to group by before deduplication.
        threshold (float): Similarity threshold for considering texts as duplicates.

    Returns:
        pd.DataFrame: DataFrame with near-duplicates removed within each group.
    """

    # Function to eliminate near duplicates within a group
    def dedupe_group(group: pd.DataFrame) -> pd.DataFrame:
        if group.shape[0] > 1:  # Proceed only if there are at least 2 rows to compare
            tfidf = TfidfVectorizer().fit_transform(group[text_column])
            similarities = cosine_similarity(tfidf)
            # Mark rows for dropping
            drop_indices = []
            for i in range(similarities.shape[0]):
                for j in range(i + 1, similarities.shape[0]):
                    if similarities[i, j] > threshold:
                        drop_indices.append(j)
            group = group.drop(group.index[drop_indices]).reset_index(drop=True)
        return group

    # Apply deduplication to each group
    deduped_df = (
        df.groupby(group_columns, group_keys=False)
        .apply(dedupe_group)
        .reset_index(drop=True)
    )

    return deduped_df


def deduplicate(seq: list[str]) -> list[str]:
    """
    Source: https://stackoverflow.com/a/480227/1493011
    """

    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]
