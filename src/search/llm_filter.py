import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any, Callable, List
from langchain_openai import ChatOpenAI
from langchain.schema.messages import AIMessage
from langchain.schema.messages import BaseMessage
from langchain.schema.messages import HumanMessage
from langchain.schema.messages import SystemMessage
from langchain.schema.document import Document
from langchain_community.document_loaders import DataFrameLoader
from pydantic import BaseModel

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


class UsefulLLMFilter:
    USEFUL_PAT = "Yes useful"
    NONUSEFUL_PAT = "Not useful"

    CHUNK_FILTER_PROMPT = f"""
    Determine if the reference section is USEFUL for answering the user query.
    It is good enough for the section to be related or similar to the query; \
    it should be relevant information that is USEFUL for comparing to the query.
    If the section contains ANY useful information, that is good enough, \
    it does not need to fully answer the user query, but it \
    should at least address a component to be USEFUL.

    Reference Section:
    ```
    {{chunk_text}}
    ```

    User Query:
    ```
    {{user_query}}
    ```

    Respond with EXACTLY AND ONLY: "{USEFUL_PAT}" or "{NONUSEFUL_PAT}"
    """.strip()

    @staticmethod
    def dataframe2documents(df: pd.DataFrame, text_column: str) -> List[Document]:
        """
        Converts a DataFrame to a list of Document objects.

        Args:
            df (pd.DataFrame): The DataFrame to convert.
            text_column (str): The column name containing the text content.

        Returns:
            List[Document]: A list of Document objects.
        """
        loader = DataFrameLoader(
            df,
            page_content_column=text_column,
        )
        docs = loader.load()
        return docs

    @staticmethod
    def documents2dataframe(
        documents: List[Document], text_column: str
    ) -> pd.DataFrame:
        """
        Converts a list of Document objects to a DataFrame.

        Args:
            documents (List[Document]): The list of Document objects to convert.
            text_column (str): The column name to use for the text content.

        Returns:
            pd.DataFrame: A DataFrame containing the document data.
        """
        rows = []
        for chunk in documents:
            row = {
                text_column: chunk.page_content,
                **chunk.metadata,
            }
            rows = rows + [row]

        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def dict_based_prompt_to_langchain_prompt(
        messages: list[dict[str, str]]
    ) -> list[BaseMessage]:
        """
        Converts a list of dictionary-based messages to a list of BaseMessage objects.

        Args:
            messages (list[dict[str, str]]): The list of dictionary-based messages.

        Returns:
            list[BaseMessage]: A list of BaseMessage objects.

        Raises:
            ValueError: If a message is missing the 'role' or 'content' key, or if an unknown role is encountered.
        """
        prompt: list[BaseMessage] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not role:
                raise ValueError(f"Message missing `role`: {message}")
            if not content:
                raise ValueError(f"Message missing `content`: {message}")
            elif role == "user":
                prompt.append(HumanMessage(content=content))
            elif role == "system":
                prompt.append(SystemMessage(content=content))
            elif role == "assistant":
                prompt.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown role: {role}")
        return prompt
    
    @staticmethod
    def run_functions_tuples_in_parallel(
        functions_with_args: list[tuple[Callable, tuple]],
        allow_failures: bool = False,
        max_workers: int | None = None,
    ) -> list[Any]:
        """
        Executes multiple functions in parallel and returns a list of the results in the same order as the input.

        Args:
            functions_with_args: List of tuples, each containing a function callable and a tuple of its arguments.
            allow_failures: If True, continues execution even if a function raises an exception. The result for that
                            function will be None. If False (default), raises the exception and stops execution.
            max_workers: Maximum number of worker threads to use. If None (default), uses the number of input functions.

        Returns:
            list: A list of results in the same order as the input functions. If a function raised an exception and
                allow_failures is True, its result will be None.

        Raises:
            Exception: If allow_failures is False and any of the functions raised an exception.
        """
        workers = (
            min(max_workers, len(functions_with_args))
            if max_workers is not None
            else len(functions_with_args)
        )

        if workers <= 0:
            return []

        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(func, *args): i
                for i, (func, args) in enumerate(functions_with_args)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results.append((index, future.result()))
                except Exception as e:
                    logger.exception(f"Function at index {index} failed due to {e}")
                    results.append((index, None))

                    if not allow_failures:
                        raise

        results.sort(key=lambda x: x[0])
        return [result for index, result in results]

    @staticmethod
    def llm_eval_chunk(query: str, chunk_content: str) -> bool:
        """
        Evaluates the usefulness of a chunk using an LLM.

        Args:
            query (str): The user query.
            chunk_content (str): The content of the chunk to evaluate.

        Returns:
            bool: True if the chunk is useful, False otherwise.
        """

        def _get_usefulness_messages() -> list[dict[str, str]]:
            messages = [
                {
                    "role": "user",
                    "content": UsefulLLMFilter.CHUNK_FILTER_PROMPT.format(
                        chunk_text=chunk_content, user_query=query
                    ),
                },
            ]
            return messages

        def _extract_usefulness(model_output: str) -> bool:
            """
            Default 'useful' if the LLM doesn't match the pattern exactly. 
                We don't want to throw away data just because the LLM fails
            """
            if (
                model_output.content.strip().strip('"').lower()
                == UsefulLLMFilter.NONUSEFUL_PAT.lower()
            ):
                return False
            return True

        llm = ChatOpenAI(model="gpt-3.5-turbo")

        messages = _get_usefulness_messages()
        filled_llm_prompt = UsefulLLMFilter.dict_based_prompt_to_langchain_prompt(messages)
        model_output = llm.invoke(filled_llm_prompt)

        return _extract_usefulness(model_output)

    @staticmethod
    def llm_batch_eval_chunks(
        query: str, chunk_contents: list[str], use_threads: bool = True
    ) -> list[bool]:
        """
        Evaluates the usefulness of multiple chunks using an LLM in batch.

        Args:
            query (str): The user query.
            chunk_contents (list[str]): A list of chunk contents to evaluate.
            use_threads (bool, optional): Whether to use threads for parallel processing. Defaults to True.

        Returns:
            list[bool]: A list of boolean values indicating the usefulness of each chunk.
        """
        if use_threads:
            functions_with_args: list[tuple[Callable, tuple]] = [
                (UsefulLLMFilter.llm_eval_chunk, (query, chunk_content))
                for chunk_content in chunk_contents
            ]

            logger.info(
                "Running LLM usefulness eval in parallel"
            )
            parallel_results = UsefulLLMFilter.run_functions_tuples_in_parallel(
                functions_with_args, allow_failures=True
            )

            # In case of failure/timeout, don't throw out the chunk
            return [True if item is None else item for item in parallel_results]

        else:
            return [
                UsefulLLMFilter.llm_eval_chunk(query, chunk_content)
                for chunk_content in chunk_contents
            ]

    @staticmethod
    def filter_chunks(
        query: str,
        chunks_to_filter: list[BaseModel],
        max_llm_filter_chunks: int = 20,
    ) -> list[BaseModel]:
        """
        Filters chunks based on whether the LLM thought they were relevant to the query.

        Args:
            query (str): The query to filter chunks against.
            chunks_to_filter (list[BaseModel]): A list of BaseModel objects to filter.
            max_llm_filter_chunks (int, optional): The maximum number of chunks to consider. Defaults to 20.

        Returns:
            list[BaseModel]: A list of BaseModel objects that were marked as relevant.
        """
        chunks_to_filter = chunks_to_filter[:max_llm_filter_chunks]
        llm_chunk_selection = UsefulLLMFilter.llm_batch_eval_chunks(
            query=query,
            chunk_contents=[chunk.page_content for chunk in chunks_to_filter],
        )
        return [
            chunk
            for ind, chunk in enumerate(chunks_to_filter)
            if llm_chunk_selection[ind]
        ]

    @staticmethod
    def filter_df_with_llm(
        query: str,
        df: pd.DataFrame,
        text_column: str,
        max_llm_chunks: int = 20,
    ) -> pd.DataFrame:
        """
        Filters a DataFrame based on whether the LLM thought the chunks were relevant to the query.

        Args:
            query (str): The query to filter chunks against.
            df (pd.DataFrame): The DataFrame to filter.
            text_column (str): The column name containing the text content.
            max_llm_chunks (int, optional): The maximum number of chunks to pass to the llm. Defaults to 20.

        Returns:
            pd.DataFrame: A filtered DataFrame containing only the relevant chunks.
        """
        logger.info(f"Input DataFrame has {len(df)} rows")
        df.rename(columns={text_column: 'text'}, inplace=True)
        documents = UsefulLLMFilter.dataframe2documents(df, text_column='text')
        filtered_documents = UsefulLLMFilter.filter_chunks(query, documents, max_llm_chunks)
        filtered_df = UsefulLLMFilter.documents2dataframe(filtered_documents, text_column='text')
        filtered_df.rename(columns={'text': text_column}, inplace=True)
        logger.info(f"LLM filtered out {len(df)-len(filtered_df)} rows - output df shape is {filtered_df.shape}")
        return filtered_df
    
