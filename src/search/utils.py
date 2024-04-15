import logging
from contextlib import contextmanager
from typing import (
    Any,
    List,
    Optional,
    Type,
)

import numpy as np
import pandas as pd
from pydantic import BaseModel, create_model

from src.search.models import DocMetaData, Document


def numpy_to_python_type(numpy_type: Type[Any]) -> Type[Any]:
    """Converts a numpy data type to its Python equivalent."""
    type_mapping = {
        np.float64: float,
        np.float32: float,
        np.int64: int,
        np.int32: int,
        np.bool_: bool,
        # Add other numpy types as necessary
    }
    return type_mapping.get(numpy_type, numpy_type)


def dataframe_to_pydantic_model(df: pd.DataFrame) -> Type[BaseModel]:
    """Make a Pydantic model from a dataframe."""
    fields = {col: (type(df[col].iloc[0]), ...) for col in df.columns}
    return create_model("DataFrameModel", __base__=BaseModel, **fields)  # type: ignore


def dataframe_to_pydantic_objects(df: pd.DataFrame) -> List[BaseModel]:
    """Make a list of Pydantic objects from a dataframe."""
    Model = dataframe_to_pydantic_model(df)
    return [Model(**row.to_dict()) for index, row in df.iterrows()]


def first_non_null(series: pd.Series) -> Any | None:
    """Find the first non-null item in a pandas Series."""
    for item in series:
        if item is not None:
            return item
    return None


def dataframe_to_document_model(
    df: pd.DataFrame,
    content: str = "content",
    metadata: List[str] = [],
    exclude: List[str] = [],
) -> Type[BaseModel]:
    """
    Make a subclass of Document from a dataframe.

    Args:
        df (pd.DataFrame): The dataframe.
        content (str): The name of the column containing the content,
            which will map to the Document.content field.
        metadata (List[str]): A list of column names containing metadata;
            these will be included in the Document.metadata field.
        exclude (List[str]): A list of column names to exclude from the model.
            (e.g. "vector" when lance is used to add an embedding vector to the df)

    Returns:
        Type[BaseModel]: A pydantic model subclassing Document.
    """

    # Remove excluded columns
    df = df.drop(columns=exclude, inplace=False)
    # Check if metadata_cols is empty

    if metadata:
        # Define fields for the dynamic subclass of DocMetaData
        metadata_fields = {
            col: (
                Optional[numpy_to_python_type(type(first_non_null(df[col])))],
                None,
            )
            for col in metadata
        }
        DynamicMetaData = create_model(  # type: ignore
            "DynamicMetaData", __base__=DocMetaData, **metadata_fields
        )
    else:
        # Use the base DocMetaData class directly
        DynamicMetaData = DocMetaData

    # Define additional top-level fields for DynamicDocument
    additional_fields = {
        col: (
            Optional[numpy_to_python_type(type(first_non_null(df[col])))],
            None,
        )
        for col in df.columns
        if col not in metadata and col != content
    }

    # Create a dynamic subclass of Document
    DynamicDocumentFields = {
        **{"metadata": (DynamicMetaData, ...)},
        **additional_fields,
    }
    DynamicDocument = create_model(  # type: ignore
        "DynamicDocument", __base__=Document, **DynamicDocumentFields
    )

    def from_df_row(
        cls: type[BaseModel],
        row: pd.Series,
        content: str = "content",
        metadata: List[str] = [],
    ) -> BaseModel | None:
        content_val = row[content] if (content and content in row) else ""
        metadata_values = (
            {col: row[col] for col in metadata if col in row} if metadata else {}
        )
        additional_values = {
            col: row[col] for col in additional_fields if col in row and col != content
        }
        metadata = DynamicMetaData(**metadata_values)
        return cls(content=content_val, metadata=metadata, **additional_values)

    # Bind the method to the class
    DynamicDocument.from_df_row = classmethod(from_df_row)

    return DynamicDocument  # type: ignore


def dataframe_to_documents(
    df: pd.DataFrame,
    content: str = "content",
    metadata: List[str] = [],
    doc_cls: Type[BaseModel] | None = None,
) -> List[Document]:
    """
    Make a list of Document objects from a dataframe.
    Args:
        df (pd.DataFrame): The dataframe.
        content (str): The name of the column containing the content,
            which will map to the Document.content field.
        metadata (List[str]): A list of column names containing metadata;
            these will be included in the Document.metadata field.
        doc_cls (Type[BaseModel], optional): A Pydantic model subclassing
            Document. Defaults to None.
    Returns:
        List[Document]: The list of Document objects.
    """
    Model = doc_cls or dataframe_to_document_model(df, content, metadata)
    docs = [
        Model.from_df_row(row, content, metadata)  # type: ignore
        for _, row in df.iterrows()
    ]
    return [m for m in docs if m is not None]

