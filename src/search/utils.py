import logging
from contextlib import contextmanager
import re
from typing import (
    Any,
    List,
    Set,
    Tuple,
    Optional,
    Type,
)
import spacy
import textacy
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
from thefuzz import fuzz  
from thefuzz import process  
from spacy.lang.en import English
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


def get_nlp() -> English:
    # Load spacy model
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe("merge_noun_chunks")
    nlp.add_pipe("merge_entities")

    return nlp


def extract_entities(spacy_doc: spacy.tokens.doc.Doc, entity_types: List[str] = ["PERSON"], lowercase: bool = False) -> Set[str]:
    """
    Extract entities of specified types from a text, matching regardless of capitalization.
    Cleans entity text by removing special characters and numbers.

    Args:
        spacy_doc (spacy.tokens.doc.Doc): The spaCy document to extract entities from.
        entity_types (List[str], optional): The types of entities to extract. Defaults to ["PERSON"].
        lowercase (bool, optional): Whether to lowercase the extracted entities in the output. Defaults to False.

    Returns:
        Set[str]: A set of unique extracted entities.
    """
    unique_entities = set()
    for ent in spacy_doc.ents:
        if ent.label_.upper() in [et.upper() for et in entity_types]:
            ent_clean = "".join([c for c in ent.text if c.isalpha() or c.isspace()]).strip()
            ent_clean = re.sub('[!,*)@#%(&$_?.^]', '', ent_clean).replace("\n", " ").strip()
            if ent_clean:
                if lowercase:
                    unique_entities.add(ent_clean.lower())
                else:
                    unique_entities.add(ent_clean)
    return unique_entities


def merge_similar_entities(entities: Set[str], threshold=80) -> List[str]:
    """
    Merge similar entities using fuzzy matching.

    Args:
        entities (Set[str]): A set of entities to merge.
        threshold (int, optional): The similarity threshold for merging. Defaults to 80.

    Returns:
        List[str]: A list of merged entities.
    """
    merged_entities = []
    for entity in entities:
        found = False
        for idx, merged_entity in enumerate(merged_entities):
            if fuzz.token_set_ratio(entity, merged_entity) >= threshold:
                final_entity = process.extractOne(entity, [entity, merged_entity])[0]
                print(f"Merging '{entity}' and '{merged_entity}' into '{final_entity}'")
                merged_entities[idx] = final_entity
                found = True
                break
        if not found:
            # print(f"Adding new entity: '{entity}'")
            merged_entities.append(entity)
    return merged_entities


def is_entity_present(entity_set: Set[str], token: spacy.tokens.token.Token, threshold = 80) -> Tuple[bool, str]:
    """Check if a token is present in a set of entities. If yes, return the entity.
    For matching entities and tokens, use fuzzy matching with a threshold.
    """
    for entity in entity_set:  
        if fuzz.partial_ratio(entity, token.text) >= threshold:  
            return True, entity  
    return False, ''

def extract_triplets_for_entities(
        spacy_doc: spacy.tokens.doc.Doc,
        entities: Set[str],
        nlp: English,
        threshold=80):
    """Extracts subject-verb-object triplets from a text only if the subject or object
    is present in the set of entities. Uses fuzzy matching with a threshold.
    """
    triplets = list(textacy.extract.subject_verb_object_triples(spacy_doc))
    stopwords = nlp.Defaults.stop_words

    triplets_with_ents = []  
    for triplet in triplets:
        for sub in triplet.subject:
            if sub.text in stopwords:
                continue
            ent_present_sub, ent_sub = is_entity_present(entities, sub, threshold)
            for obj in triplet.object:
                if obj.text in stopwords:
                    continue
                ent_present_ob, ent_ob = is_entity_present(entities, obj, threshold)
                if ent_present_sub or ent_present_ob:
                    triplets_with_ents.append(
                        (ent_sub if ent_present_sub else sub.text.replace("\n", " ").strip(), 
                         " ".join([tok.text for tok in triplet.verb]).strip(),
                         ent_ob if ent_present_ob else obj.text.replace("\n", " ").strip()))
    return triplets_with_ents 