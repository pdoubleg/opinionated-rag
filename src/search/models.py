from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from typing import Any, List, Literal, Optional
import pandas as pd

from pydantic import BaseModel
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import chromadb

from src.types import Document

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

DISABLE_LLM_CHUNK_FILTER = False
NUM_RERANKED_RESULTS = 10
NUM_RETURNED_HITS = 20
SKIP_RERANK = False

MAX_METRICS_CONTENT = (
    200  # Just need enough characters to identify where in the doc the chunk is
)


Embedding = list[float]

from typing import Optional, List
from pydantic import BaseModel, Field


class QueryScreening(BaseModel):
    """Initial evaluation screening for a user new query."""
    
    topic: str = Field(
        ...,
        description="The high-level type of question being asked using as few words as possible.",
    )
    intent: Literal['keyword', 'question-answer'] = Field(
        ...,
        description="The predicted search type that would best address the user query. Use 'keyword' when it's clear the user only required dictionary look-up-type search.",
    )
    n_subquestions: int = Field(
        default=0,
        description="The number of distinct sub-questions that the user query naturally breaks down to, or `0` if the original user query is already atomic.",
    )
    
   


class CitationInformation(BaseModel):
    """Generalized information about a legal citation."""

    citation: str = Field(
        ...,
        description="The Citation specified by the user.",
    )
    summary: str = Field(
        ...,
        description="A fact-focused summary of the citation's scope of authority as evidenced by its past use.",
    )
    questions: List[str] = Field(
        default_factory=list,
        description="A list of named-entity-free generalized questions, or potential questions the cited case can help answer or address. Do NOT reference specific entities, but rather the broader idea, theory or concept.",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="A correctly resolved list of important keywords from the Context. Group similar topics together."
    )
    recency: Optional[str] = Field(
        None,
        description="Optional summary of changes in the interpretation or use of the cited case with a focus on the most recent (listed toward the end of the Context) texts.",
    )

    def __str__(self) -> str:
        """Pretty prints the citation information, wrapping the summary, keywords, and recency text, and listing questions as bullet points."""
        from textwrap import fill
        summary_wrapped = fill(self.summary, width=100)
        questions_formatted = '\n'.join([f"- {q}" for q in self.questions])
        keywords_wrapped = fill(', '.join(self.keywords), width=100)
        recency_wrapped = fill(self.recency, width=100) if self.recency else "No recent observations."
        return (
            f"Citation: {self.citation}\n"
            f"Summary:\n{summary_wrapped}\n"
            f"Questions:\n{questions_formatted}\n"
            f"Keywords:\n{keywords_wrapped}\n"
            f"Recency:\n{recency_wrapped}"
        )
        
        
        
def init_vector_store_index(nodes: List[TextNode]):
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("chroma_db")

    embeddings = OpenAIEmbedding(
        model='text-embedding-ada-002',
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )
    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embeddings,
        storage_context=storage_context,
    )
    return index



def dataframe_to_text_nodes(
    df: pd.DataFrame,
    id_column: str,
    text_column: str,
    metadata_fields: List[str], 
    embedding_column: str | None = None,
    has_score: bool = False
) -> List[TextNode | NodeWithScore]:
    """
    Creates a list of TextNode objects from a DataFrame based on specified fields for text, metadata, excluded metadata keys, and optionally embedding columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be converted into TextNode objects.
        text_column (str): The column name in the DataFrame to use as the text for each TextNode.
        metadata_fields (List[str]): A list of column names in the DataFrame to include in the metadata of each TextNode.
        excluded_metadata_keys (List[str] | None): A list of keys to exclude from the embedded metadata in the text representation of each TextNode. Defaults to None.
        embedding_column (List[str]): Column name whose values should be added as an embedding attribute to each TextNode. If None, the embedding attribute is skipped. Defaults to None.

    Returns:
        List[TextNode]: A list of TextNode objects created from the DataFrame.
    """
    nodes = []
    for _, row in df.iterrows():
        metadata = {field: row[field] for field in metadata_fields}
        if embedding_column:
            embedding = list(row[embedding_column])
            node = TextNode(
                id_=row[id_column],
                text=row[text_column],
                metadata=metadata,
                embedding=embedding,
                metadata_template="{key} = {value}",
                text_template="Metadata:\n{metadata_str}\n----------------------------------------\nContent:\n{content}",
                )
            if has_score:
                node_with_score = NodeWithScore(
                    node=node,
                    score=row['score'],
                )
                nodes.append(node_with_score)
            else:          
                nodes.append(node)
        else:
            node = TextNode(
                id_=row[id_column],
                text=row[text_column],
                metadata=metadata,
                metadata_template="{key} = {value}",
                text_template="Metadata:\n{metadata_str}\n----------------------------------------\nContent:\n{content}",
                )
            if has_score:
                node_with_score = NodeWithScore(
                node=node,
                score=row['score'],
                )
                nodes.append(node_with_score)
            else:          
                nodes.append(node)
        
    return nodes


def text_nodes_to_dataframe(
    text_nodes: List[TextNode | NodeWithScore],
    text_column: str = 'text',
    metadata_fields: Optional[List[str]] = None,
    embedding_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Converts a list of TextNode or NodeWithScore objects back into a DataFrame. The DataFrame will contain columns for text,
    optionally for embedding, and for each metadata field specified. If the object is a NodeWithScore, a 'score' column will also be included.

    Args:
        text_nodes (List[TextNode | NodeWithScore]): The list of TextNode or NodeWithScore objects to be converted into a DataFrame.
        text_column (str): The column name to use for the text from each TextNode. Defaults to 'text'.
        metadata_fields (Optional[List[str]]): A list of metadata field names to include in the DataFrame.
            If None, all metadata fields found in the TextNodes will be included. Defaults to None.
        embedding_column (Optional[str]): The column name to use for the embedding from each TextNode.
            If None, the embedding attribute is skipped. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame created from the list of TextNode or NodeWithScore objects.
    """
    data = []
    for node in text_nodes:
        if isinstance(node, NodeWithScore):
            text_node = node.node
            score = node.score
        else:
            text_node = node
            score = None  # Use None for rows where the node is not a NodeWithScore

        row = {text_column: text_node.text, 'score': score}
        if metadata_fields is None:
            row.update(text_node.metadata)
        else:
            for field in metadata_fields:
                row[field] = text_node.metadata.get(field)
        if embedding_column and hasattr(text_node, 'embedding'):
            row[embedding_column] = text_node.embedding
        data.append(row)
    
    return pd.DataFrame(data)


def documents2Dataframe(documents: List[TextNode]) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.text,
            **chunk.metadata,
            "chunk_id": chunk.id,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df





@dataclass
class ChunkEmbedding:
    full_embedding: Embedding
    mini_chunk_embeddings: list[Embedding]


@dataclass
class BaseChunk:
    chunk_id: int
    blurb: str  # The first sentence(s) of the first Section of the chunk
    content: str
    source_links: dict[
        int, str
    ] | None  # Holds the link and the offsets into the raw Chunk text
    section_continuation: bool  # True if this Chunk's start is not at the start of a Section


@dataclass
class DocAwareChunk(BaseChunk):
    # During indexing flow, we have access to a complete "Document"
    # During inference we only have access to the document id and do not reconstruct the Document
    source_document: Document

    def to_short_descriptor(self) -> str:
        """Used when logging the identity of a chunk"""
        return (
            f"Chunk ID: '{self.chunk_id}'; {self.source_document.to_short_descriptor()}"
        )


@dataclass
class IndexChunk(DocAwareChunk):
    embeddings: ChunkEmbedding
    title_embedding: Embedding | None


@dataclass
class DocMetadataAwareIndexChunk(IndexChunk):
    """An `IndexChunk` that contains all necessary metadata to be indexed. This includes
    the following:

    document_sets: all document sets the source document for this chunk is a part
                   of. This is used for filtering / personas.
    boost: influences the ranking of this chunk at query time. Positive -> ranked higher,
           negative -> ranked lower.
    """
    document_sets: set[str]
    boost: int

    @classmethod
    def from_index_chunk(
        cls,
        index_chunk: IndexChunk,
        document_sets: set[str],
        boost: int,
    ) -> "DocMetadataAwareIndexChunk":
        return cls(
            **{
                field.name: getattr(index_chunk, field.name)
                for field in fields(index_chunk)
            },
            document_sets=document_sets,
            boost=boost,
        )


@dataclass
class InferenceChunk(BaseChunk):
    document_id: str
    semantic_identifier: str
    boost: int
    recency_bias: float
    score: float | None
    hidden: bool
    metadata: dict[str, str | list[str]]
    # Matched sections in the chunk. Uses Vespa syntax e.g. <hi>TEXT</hi>
    # to specify that a set of words should be highlighted. For example:
    # ["<hi>the</hi> <hi>answer</hi> is 42", "he couldn't find an <hi>answer</hi>"]
    match_highlights: list[str]
    # when the doc was last updated
    updated_at: datetime | None
    primary_owners: list[str] | None = None
    secondary_owners: list[str] | None = None

    @property
    def unique_id(self) -> str:
        return f"{self.document_id}__{self.chunk_id}"

    def __repr__(self) -> str:
        blurb_words = self.blurb.split()
        short_blurb = ""
        for word in blurb_words:
            if not short_blurb:
                short_blurb = word
                continue
            if len(short_blurb) > 25:
                break
            short_blurb += " " + word
        return f"Inference Chunk: {self.document_id} - {short_blurb}..."


class DocumentSource(str, Enum):
    # Special case, document passed in via Danswer APIs without specifying a source type
    INGESTION_API = "ingestion_api"
    SLACK = "slack"
    WEB = "web"
    GOOGLE_DRIVE = "google_drive"
    GMAIL = "gmail"


class OptionalSearchSetting(str, Enum):
    ALWAYS = "always"
    NEVER = "never"
    # Determine whether to run search based on history and latest query
    AUTO = "auto"


class RecencyBiasSetting(str, Enum):
    FAVOR_RECENT = "favor_recent"  # 2x decay rate
    BASE_DECAY = "base_decay"
    NO_DECAY = "no_decay"
    # Determine based on query if to use base_decay or favor_recent
    AUTO = "auto"


class SearchType(str, Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class QueryFlow(str, Enum):
    SEARCH = "search"
    QUESTION_ANSWER = "question-answer"


class Tag(BaseModel):
    tag_key: str
    tag_value: str


class BaseFilters(BaseModel):
    source_type: list[DocumentSource] | None = None
    document_set: list[str] | None = None
    time_cutoff: datetime | None = None
    tags: list[Tag] | None = None


class IndexFilters(BaseFilters):
    access_control_list: list[str] | None


class ChunkMetric(BaseModel):
    document_id: str
    chunk_content_start: str
    first_link: str | None
    score: float


class SearchQuery(BaseModel):
    query: str
    filters: IndexFilters
    recency_bias_multiplier: float
    num_hits: int = NUM_RETURNED_HITS
    offset: int = 0
    search_type: SearchType = SearchType.HYBRID
    skip_rerank: bool = not SKIP_RERANK
    # Only used if not skip_rerank
    num_rerank: int | None = NUM_RERANKED_RESULTS
    skip_llm_chunk_filter: bool = DISABLE_LLM_CHUNK_FILTER
    # Only used if not skip_llm_chunk_filter
    max_llm_filter_chunks: int = NUM_RERANKED_RESULTS

    class Config:
        frozen = True


class RetrievalDetails(BaseModel):
    # Use LLM to determine whether to do a retrieval or only rely on existing history
    # If the Persona is configured to not run search (0 chunks), this is bypassed
    # If no Prompt is configured, the only search results are shown, this is bypassed
    run_search: OptionalSearchSetting = OptionalSearchSetting.ALWAYS
    # Is this a real-time/streaming call or a question where Danswer can take more time?
    # Used to determine reranking flow
    real_time: bool = True
    # The following have defaults in the Persona settings which can be overriden via
    # the query, if None, then use Persona settings
    filters: BaseFilters | None = None
    enable_auto_detect_filters: bool | None = None
    # if None, no offset / limit
    offset: int | None = None
    limit: int | None = None


class SearchDoc(BaseModel):
    document_id: str
    chunk_ind: int
    semantic_identifier: str
    link: str | None
    blurb: str
    source_type: DocumentSource
    boost: int
    # Whether the document is hidden when doing a standard search
    # since a standard search will never find a hidden doc, this can only ever
    # be `True` when doing an admin search
    hidden: bool
    metadata: dict[str, str | list[str]]
    score: float | None
    # Matched sections in the doc. Uses Vespa syntax e.g. <hi>TEXT</hi>
    # to specify that a set of words should be highlighted. For example:
    # ["<hi>the</hi> <hi>answer</hi> is 42", "the answer is <hi>42</hi>""]
    match_highlights: list[str]
    # when the doc was last updated
    updated_at: datetime | None
    primary_owners: list[str] | None
    secondary_owners: list[str] | None

    def dict(self, *args: list, **kwargs: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        initial_dict = super().dict(*args, **kwargs)  # type: ignore
        initial_dict["updated_at"] = (
            self.updated_at.isoformat() if self.updated_at else None
        )
        return initial_dict


class SavedSearchDoc(SearchDoc):
    db_doc_id: int
    score: float = 0.0

    @classmethod
    def from_search_doc(
        cls, search_doc: SearchDoc, db_doc_id: int = 0
    ) -> "SavedSearchDoc":
        """IMPORTANT: careful using this and not providing a db_doc_id"""
        search_doc_data = search_doc.dict()
        search_doc_data["score"] = search_doc_data.get("score", 0.0)
        return cls(**search_doc_data, db_doc_id=db_doc_id)


class RetrievalDocs(BaseModel):
    top_documents: list[SavedSearchDoc]


class SearchResponse(RetrievalDocs):
    llm_indices: list[int]


class RetrievalMetricsContainer(BaseModel):
    search_type: SearchType
    metrics: list[ChunkMetric]  # This contains the scores for retrieval as well


class RerankMetricsContainer(BaseModel):
    """The score held by this is the un-boosted, averaged score of the ensemble cross-encoders"""

    metrics: list[ChunkMetric]
    raw_similarity_scores: list[float]