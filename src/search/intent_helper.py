import string
from enum import Enum
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pydantic import BaseModel
from transformers import AutoTokenizer

from src.search.intent import IntentModel, get_intent_model_tokenizer
from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


class SearchType(str, Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class QueryFlow(str, Enum):
    SEARCH = "search"
    QUESTION_ANSWER = "question-answer"


class HelperResponse(BaseModel):
    values: dict[str, str]
    details: list[str] | None = None


def remove_stop_words_and_punctuation(text: str) -> list[str]:
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    text_trimmed = [
        word
        for word in word_tokens
        if (word.casefold() not in stop_words and word not in string.punctuation)
    ]
    return text_trimmed or word_tokens


def count_unk_tokens(text: str, tokenizer: "AutoTokenizer") -> int:
    """Unclear if the wordpiece tokenizer used is actually tokenizing anything as the [UNK] token
    It splits up even foreign characters and unicode emojis without using UNK"""
    tokenized_text = tokenizer.tokenize(text)
    num_unk_tokens = len(
        [token for token in tokenized_text if token == tokenizer.unk_token]
    )
    logger.debug(f"Total of {num_unk_tokens} UNKNOWN tokens found")
    return num_unk_tokens


def query_intent(query: str) -> tuple[SearchType, QueryFlow]:
    intent_model = IntentModel()
    class_probs = intent_model.predict(query)
    keyword = class_probs[0]
    semantic = class_probs[1]
    qa = class_probs[2]

    # Heavily bias towards QA, from user perspective, answering a statement is not as bad as not answering a question
    if qa > 20:
        # If one class is very certain, choose it still
        if keyword > 70:
            predicted_search = SearchType.KEYWORD
            predicted_flow = QueryFlow.SEARCH
        elif semantic > 70:
            predicted_search = SearchType.SEMANTIC
            predicted_flow = QueryFlow.SEARCH
        # If it's a QA question, it must be a "Semantic" style statement/question
        else:
            predicted_search = SearchType.SEMANTIC
            predicted_flow = QueryFlow.QUESTION_ANSWER
    # If definitely not a QA question, choose between keyword or semantic search
    elif keyword > semantic:
        predicted_search = SearchType.KEYWORD
        predicted_flow = QueryFlow.SEARCH
    else:
        predicted_search = SearchType.SEMANTIC
        predicted_flow = QueryFlow.SEARCH

    logger.debug(f"Predicted Search: {predicted_search}")
    logger.debug(f"Predicted Flow: {predicted_flow}")
    return predicted_search, predicted_flow


def recommend_search_flow(
    query: str,
    keyword: bool = False,
    max_percent_stopwords: float = 0.30,  # ~Every third word max, ie "effects of caffeine" still viable keyword search
) -> HelperResponse:
    heuristic_search_type: SearchType | None = None
    message: str | None = None

    # Heuristics based decisions
    words = query.split()
    non_stopwords = remove_stop_words_and_punctuation(query)
    non_stopword_percent = len(non_stopwords) / len(words)

    # UNK tokens -> suggest Keyword (still may be valid QA)
    if count_unk_tokens(query, get_intent_model_tokenizer()) > 0:
        if not keyword:
            heuristic_search_type = SearchType.KEYWORD
            message = "Unknown tokens in query."

    # Too many stop words, most likely a Semantic query (still may be valid QA)
    if non_stopword_percent < 1 - max_percent_stopwords:
        if keyword:
            heuristic_search_type = SearchType.SEMANTIC
            message = "Stopwords in query"

    # Model based decisions
    model_search_type, flow = query_intent(query)
    if not message:
        if model_search_type == SearchType.SEMANTIC and keyword:
            message = "Intent model classified Semantic Search"
        if model_search_type == SearchType.KEYWORD and not keyword:
            message = "Intent model classified Keyword Search."

    return HelperResponse(
        values={
            "flow": flow,
            "search_type": model_search_type
            if heuristic_search_type is None
            else heuristic_search_type,
        },
        details=[message] if message else [],
    )
