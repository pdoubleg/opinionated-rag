"""
This module includes an implementation of Instructional Permutation Generation with Sliding Window Strategy as described in the RankGPT paper. 
For more details, refer to the paper available at: https://arxiv.org/abs/2304.09542

The code is largely based on the llama-index version with some modifications, namely a method for operating over pandas DataFrame.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import uuid
from llama_index.core.llms import LLM, ChatMessage, ChatResponse
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import RANKGPT_RERANK_PROMPT
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.utils import print_text
import numpy as np
import pandas as pd
from pydantic import Field

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

PromptDictType = Dict[str, BasePromptTemplate]

DEFAULT_SYSTEM_MESSAGE = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."

RANKGPT_RERANK_PROMPT_TMPL = (
    "Search Query: {query}. \nRank the {num} passages above "
    "based on their relevance to the search query. The passages "
    "should be listed in descending order using identifiers. "
    "The most relevant passages should be listed first. "
    "The output format should be [] > [], e.g., [1] > [2]. "
    "Only response the ranking results, "
    "do not say any word or explain."
)
RANKGPT_RERANK_PROMPT = PromptTemplate(
    RANKGPT_RERANK_PROMPT_TMPL, prompt_type=PromptType.RANKGPT_RERANK
)


def get_default_llm() -> LLM:
    """
    Returns the default LLM (Language Model) to use for RankGPT.

    Returns:
        LLM: The default LLM instance (OpenAI's gpt-3.5-turbo-16k).
    """
    from llama_index.llms.openai import OpenAI

    return OpenAI(model="gpt-3.5-turbo-16k")


def dataframe_to_text_nodes(
    df: pd.DataFrame,
    id_column: str,
    text_column: str,
    metadata_fields: List[str],
    embedding_column: str | None = None,
    has_score: bool = False,
) -> List[TextNode | NodeWithScore]:
    """
    Creates a list of TextNode or NodeWithScore objects from a DataFrame based on specified fields for text, metadata, and optionally embedding columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be converted into TextNode or NodeWithScore objects.
        id_column (str): The column name in the DataFrame to use as the ID for each TextNode.
        text_column (str): The column name in the DataFrame to use as the text for each TextNode.
        metadata_fields (List[str]): A list of column names in the DataFrame to include in the metadata of each TextNode.
        embedding_column (str | None): Column name whose values should be added as an embedding attribute to each TextNode. If None, the embedding attribute is skipped. Defaults to None.
        has_score (bool): Whether the DataFrame contains a 'score' column. If True, NodeWithScore objects are created instead of TextNode objects. Defaults to False.

    Returns:
        List[TextNode | NodeWithScore]: A list of TextNode or NodeWithScore objects created from the DataFrame.
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
                    score=row["score"],
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
                    score=row["score"],
                )
                nodes.append(node_with_score)
            else:
                nodes.append(node)

    return nodes


def text_nodes_to_dataframe(
    text_nodes: List[TextNode | NodeWithScore],
    text_column: str = "text",
    metadata_fields: Optional[List[str]] = None,
    embedding_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Converts a list of TextNode or NodeWithScore objects back into a DataFrame.

    The DataFrame will contain columns for text, optionally for embedding, and for each metadata field specified.
    If the object is a NodeWithScore, a 'score' column will also be included.

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

        row = {text_column: text_node.text, "score": score}
        if metadata_fields is None:
            row.update(text_node.metadata)
        else:
            for field in metadata_fields:
                row[field] = text_node.metadata.get(field)
        if embedding_column and hasattr(text_node, "embedding"):
            row[embedding_column] = text_node.embedding
        data.append(row)

    return pd.DataFrame(data)


class RankGPTRerank(BaseNodePostprocessor):
    """RankGPT-based reranker."""

    top_n: int = Field(default=5, description="Top N nodes to return from reranking.")
    llm: LLM = Field(
        default_factory=get_default_llm,
        description="LLM to use for RankGPT",
    )
    verbose: bool = Field(
        default=False, description="Whether to print intermediate steps."
    )
    rankgpt_rerank_prompt: BasePromptTemplate = Field(
        description="RankGPT rerank prompt."
    )

    def __init__(
        self,
        top_n: int = 5,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        rankgpt_rerank_prompt: Optional[BasePromptTemplate] = None,
    ):
        rankgpt_rerank_prompt = rankgpt_rerank_prompt or RANKGPT_RERANK_PROMPT
        super().__init__(
            verbose=verbose,
            llm=llm,
            top_n=top_n,
            rankgpt_rerank_prompt=rankgpt_rerank_prompt,
        )

    @classmethod
    def class_name(cls) -> str:
        return "RankGPTRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Union[QueryBundle, str],
    ) -> List[NodeWithScore]:
        """
        Post-processes nodes based on the query bundle or query string provided.

        Args:
            nodes (List[NodeWithScore]): A list of nodes with scores to be post-processed.
            query_bundle (Union[QueryBundle, str]): The query bundle or query string used for post-processing. Must be an instance of QueryBundle or a string.

        Raises:
            ValueError: If query_bundle is not an instance of QueryBundle or a string.

        Returns:
            List[NodeWithScore]: A list of post-processed nodes with scores.
        """
        if not isinstance(query_bundle, (QueryBundle, str)):
            raise ValueError("query_bundle must be a QueryBundle object or a string.")

        query_str = (
            query_bundle.query_str
            if isinstance(query_bundle, QueryBundle)
            else query_bundle
        )

        items = {
            "query": query_str,
            "hits": [{"content": node.get_content()} for node in nodes],
        }

        messages = self.create_permutation_instruction(item=items)
        permutation = self.run_llm(messages=messages)
        if permutation.message is not None and permutation.message.content is not None:
            rerank_ranks = self._receive_permutation(
                items, str(permutation.message.content)
            )
            if self.verbose:
                print_text(f"After Reranking, new rank list for nodes: {rerank_ranks}")

            initial_results: List[NodeWithScore] = []

            for idx in rerank_ranks:
                initial_results.append(
                    NodeWithScore(node=nodes[idx].node, score=nodes[idx].score)
                )
            return initial_results[: self.top_n]
        else:
            return nodes[: self.top_n]

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"rankgpt_rerank_prompt": self.rankgpt_rerank_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "rankgpt_rerank_prompt" in prompts:
            self.rankgpt_rerank_prompt = prompts["rankgpt_rerank_prompt"]

    def _get_prefix_prompt(self, query: str, num: int) -> List[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content=DEFAULT_SYSTEM_MESSAGE,
            ),
            ChatMessage(
                role="user",
                content=f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
            ),
            ChatMessage(role="assistant", content="Okay, please provide the passages."),
        ]

    def _get_post_prompt(self, query: str, num: int) -> str:
        return self.rankgpt_rerank_prompt.format(query=query, num=num)

    def create_permutation_instruction(self, item: Dict[str, Any]) -> List[ChatMessage]:
        query = item["query"]
        num = len(item["hits"])

        messages = self._get_prefix_prompt(query, num)
        rank = 0
        for hit in item["hits"]:
            rank += 1
            content = hit["content"]
            content = content.replace("Title: Content: ", "")
            content = content.strip()
            content = " ".join(content.split()[:300])
            messages.append(ChatMessage(role="user", content=f"[{rank}] {content}"))
            messages.append(
                ChatMessage(role="assistant", content=f"Received passage [{rank}].")
            )
        messages.append(
            ChatMessage(role="user", content=self._get_post_prompt(query, num))
        )
        return messages

    def run_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        return self.llm.chat(messages)

    def _clean_response(self, response: str) -> str:
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        return new_response.strip()

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def _receive_permutation(self, item: Dict[str, Any], permutation: str) -> List[int]:
        rank_end = len(item["hits"])

        response = self._clean_response(permutation)
        response_list = [int(x) - 1 for x in response.split()]
        response_list = self._remove_duplicate(response_list)
        response_list = [ss for ss in response_list if ss in range(rank_end)]
        return response_list + [tt for tt in range(rank_end) if tt not in response_list]
    
    def rerank_dataframe(
        self,
        df: pd.DataFrame,
        query: Union[QueryBundle, str],
        text_column: str,
    ) -> pd.DataFrame:
        """
        Reranks a DataFrame using RankGPT based on the provided query bundle or query string.

        Args:
            df (pd.DataFrame): The DataFrame to be reranked.
            query (Union[QueryBundle, str]): The query bundle or query string used for reranking.
            text_column (str): The column name in the DataFrame to use as the text for each TextNode.

        Returns:
            pd.DataFrame: The reranked DataFrame.
        """
        # Add a temporary 'id' column with UUIDs
        df["temp_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        df.rename(columns={text_column: 'text'}, inplace=True)
        
        metadata_fields = df.columns.tolist()
        metadata_fields.remove('text')
        
        # Check if 'score' column exists, if not create it with random floats
        has_score = 'score' in df.columns
        if not has_score:
            df['score'] = np.random.rand(len(df))
        
        # Convert DataFrame to TextNode or NodeWithScore objects
        nodes = dataframe_to_text_nodes(
            df,
            "temp_id",
            "text",
            metadata_fields,
            has_score=True,
        )

        # Rerank nodes using RankGPT
        reranked_nodes = self._postprocess_nodes(nodes, query)

        # Convert reranked nodes back to DataFrame
        reranked_df = text_nodes_to_dataframe(
            reranked_nodes,
            "text",
            metadata_fields,
        )

        # Remove the temporary 'id' column and 'score' column if it was created
        reranked_df.drop(columns=["temp_id"], inplace=True)
        if not has_score:
            reranked_df.drop(columns=["score"], inplace=True)
        
        reranked_df.rename(columns={'text': text_column}, inplace=True)
        
        return reranked_df
