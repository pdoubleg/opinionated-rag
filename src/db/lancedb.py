import json
import logging
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple, Type

import lancedb
import pandas as pd
from dotenv import load_dotenv
from lancedb.pydantic import LanceModel, Vector
from lancedb.query import LanceVectorQueryBuilder
from pydantic import BaseModel, ValidationError, create_model

from src.embedding_models.base import (
    EmbeddingModel,
    EmbeddingModelsConfig,
)
from src.embedding_models.models import OpenAIEmbeddingsConfig
from src.types import Document, EmbeddingFunction
from src.utils.configuration import settings
from src.utils.pydantic_utils import (
    clean_schema,
    dataframe_to_document_model,
    dataframe_to_documents,
    extract_fields,
    flatten_pydantic_instance,
    flatten_pydantic_model,
    nested_dict_from_flat,
)
from src.db.base import VectorStore, VectorStoreConfig

logger = logging.getLogger(__name__)


class LanceDBConfig(VectorStoreConfig):
    collection_name: str | None = "temp"
    storage_path: str = ".lancedb/data"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    distance: str = "cosine"
    document_class: Type[Document] = Document
    flatten: bool = False  # flatten Document class into LanceSchema ?
    filter_fields: List[str] = []  # fields usable in filter
    filter: str | None = None  # filter condition for lexical/semantic search


class LanceDB(VectorStore):
    def __init__(self, config: LanceDBConfig = LanceDBConfig()):
        super().__init__(config)
        self.config: LanceDBConfig = config
        emb_model = EmbeddingModel.create(config.embedding)
        self.embedding_fn: EmbeddingFunction = emb_model.embedding_fn()
        self.embedding_dim = emb_model.embedding_dims
        self.host = None
        self.port = None
        self.is_from_dataframe = False  # were docs ingested from a dataframe?
        self.df_metadata_columns: List[str] = []  # metadata columns from dataframe
        self._setup_schemas(config.document_class)

        load_dotenv()

        try:
            self.client = lancedb.connect(
                uri=config.storage_path,
            )
        except Exception as e:
            new_storage_path = config.storage_path + ".new"
            logger.warning(
                f"""
                Error connecting to local LanceDB at {config.storage_path}:
                {e}
                Switching to {new_storage_path}
                """
            )
            self.client = lancedb.connect(
                uri=new_storage_path,
            )

        # Note: Only create collection if a non-null collection name is provided.
        # This is useful to delay creation of vecdb until we have a suitable
        # collection name (e.g. we could get it from the url or folder path).
        if config.collection_name is not None:
            self.create_collection(
                config.collection_name, replace=config.replace_collection
            )

    def _setup_schemas(self, doc_cls: Type[Document] | None) -> None:
        doc_cls = doc_cls or self.config.document_class
        self.unflattened_schema = self._create_lance_schema(doc_cls)
        self.schema = (
            self._create_flat_lance_schema(doc_cls)
            if self.config.flatten
            else self.unflattened_schema
        )

    def clear_empty_collections(self) -> int:
        coll_names = self.list_collections()
        n_deletes = 0
        for name in coll_names:
            nr = self.client.open_table(name).head(1).shape[0]
            if nr == 0:
                n_deletes += 1
                self.client.drop_table(name)
        return n_deletes

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        """Clear all collections with the given prefix."""
        if not really:
            logger.warning("Not deleting all collections, set really=True to confirm")
            return 0
        coll_names = [
            c for c in self.list_collections(empty=True) if c.startswith(prefix)
        ]
        if len(coll_names) == 0:
            logger.warning(f"No collections found with prefix {prefix}")
            return 0
        n_empty_deletes = 0
        n_non_empty_deletes = 0
        for name in coll_names:
            nr = self.client.open_table(name).head(1).shape[0]
            n_empty_deletes += nr == 0
            n_non_empty_deletes += nr > 0
            self.client.drop_table(name)
        logger.warning(
            f"""
            Deleted {n_empty_deletes} empty collections and 
            {n_non_empty_deletes} non-empty collections.
            """
        )
        return n_empty_deletes + n_non_empty_deletes

    def list_collections(self, empty: bool = False) -> List[str]:
        """
        Returns:
            List of collection names that have at least one vector.

        Args:
            empty (bool, optional): Whether to include empty collections.
        """
        colls = self.client.table_names()
        if len(colls) == 0:
            return []
        if empty:  # include empty tbls
            return colls  # type: ignore
        counts = [self.client.open_table(coll).head(1).shape[0] for coll in colls]
        return [coll for coll, count in zip(colls, counts) if count > 0]

    def _create_lance_schema(self, doc_cls: Type[Document]) -> Type[BaseModel]:
        """
        Create a subclass of LanceModel with fields:
         - id (str)
         - Vector field that has dims equal to
            the embedding dimension of the embedding model, and a data field of type
            DocClass.
         - other fields from doc_cls

        Args:
            doc_cls (Type[Document]): A Pydantic model which should be a subclass of
                Document, to be used as the type for the data field.

        Returns:
            Type[BaseModel]: A new Pydantic model subclassing from LanceModel.

        Raises:
            ValueError: If `n` is not a non-negative integer or if `DocClass` is not a
                subclass of Document.
        """
        if not issubclass(doc_cls, Document):
            raise ValueError("DocClass must be a subclass of Document")

        n = self.embedding_dim

        # Prepare fields for the new model
        fields = {"id": (str, ...), "vector": (Vector(n), ...)}

        # Add both statically and dynamically defined fields from doc_cls
        for field_name, field in doc_cls.model_fields.items():
            fields[field_name] = (field.annotation, field.default)

        # Create the new model with dynamic fields
        NewModel = create_model(
            "NewModel", __base__=LanceModel, **fields
        )  # type: ignore
        return NewModel  # type: ignore

    def _create_flat_lance_schema(self, doc_cls: Type[Document]) -> Type[BaseModel]:
        """
        Flat version of the lance_schema, as nested Pydantic schemas are not yet
        supported by LanceDB.
        """
        lance_model = self._create_lance_schema(doc_cls)
        FlatModel = flatten_pydantic_model(lance_model, base_model=LanceModel)
        return FlatModel

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Create a collection with the given name, optionally replacing an existing
            collection if `replace` is True.
        Args:
            collection_name (str): Name of the collection to create.
            replace (bool): Whether to replace an existing collection
                with the same name. Defaults to False.
        """
        self.config.collection_name = collection_name
        collections = self.list_collections()
        if collection_name in collections:
            coll = self.client.open_table(collection_name)
            if coll.head().shape[0] > 0:
                logger.warning(f"Non-empty Collection {collection_name} already exists")
                if not replace:
                    logger.warning("Not replacing collection")
                    return
                else:
                    logger.warning("Recreating fresh collection")
        self.client.create_table(
            collection_name, schema=self.schema, mode="overwrite", on_bad_vectors="drop"
        )
        tbl = self.client.open_table(self.config.collection_name)
        # We assume "content" is available as top-level field
        if "content" in tbl.schema.names:
            tbl.create_fts_index("content", replace=True)

        if settings.debug:
            level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            logger.setLevel(level)

    def add_documents(self, documents: Sequence[Document]) -> None:
        super().maybe_add_ids(documents)
        colls = self.list_collections(empty=True)
        if len(documents) == 0:
            return
        embedding_vecs = self.embedding_fn([doc.content for doc in documents])
        coll_name = self.config.collection_name
        if coll_name is None:
            raise ValueError("No collection name set, cannot ingest docs")
        if (
            coll_name not in colls
            or self.client.open_table(coll_name).head(1).shape[0] == 0
        ):
            # collection either doesn't exist or is empty, so replace it,
            # possibly with a new schema
            doc_cls = type(documents[0])
            self.config.document_class = doc_cls
            self._setup_schemas(doc_cls)
            self.create_collection(coll_name, replace=True)

        ids = [str(d.id()) for d in documents]
        # don't insert all at once, batch in chunks of b,
        # else we get an API error
        b = self.config.batch_size

        def make_batches() -> Generator[List[BaseModel], None, None]:
            for i in range(0, len(ids), b):
                batch = [
                    self.unflattened_schema(
                        id=ids[i],
                        vector=embedding_vecs[i],
                        **doc.model_dump(),
                    )
                    for i, doc in enumerate(documents[i : i + b])
                ]
                if self.config.flatten:
                    batch = [
                        flatten_pydantic_instance(instance)  # type: ignore
                        for instance in batch
                    ]
                yield batch

        tbl = self.client.open_table(self.config.collection_name)
        try:
            tbl.add(make_batches())
            if "content" in tbl.schema.names:
                tbl.create_fts_index("content", replace=True)
                
        except Exception as e:
            logger.error(
                f"""
                Error adding documents to LanceDB: {e}
                POSSIBLE REMEDY: Delete the LancdDB storage directory
                {self.config.storage_path} and try again.
                """
            )

    def add_dataframe(
        self,
        df: pd.DataFrame,
        content: str = "content",
        metadata: List[str] = [],
    ) -> None:
        """
        Add a dataframe to the collection.
        Args:
            df (pd.DataFrame): A dataframe
            content (str): The name of the column in the dataframe that contains the
                text content to be embedded using the embedding model.
            metadata (List[str]): A list of column names in the dataframe that contain
                metadata to be stored in the database. Defaults to [].
        """
        self.is_from_dataframe = True
        actual_metadata = metadata.copy()
        self.df_metadata_columns = actual_metadata  # could be updated below
        # get content column
        content_values = df[content].values.tolist()
        if "vector" not in df.columns:
            embedding_vecs = self.embedding_fn(content_values)
            df["vector"] = embedding_vecs
        
        if content != "content":
            # rename content column to "content", leave existing column intact
            df = df.rename(columns={content: "content"}, inplace=False)

        if "id" not in df.columns:
            docs = dataframe_to_documents(df, content="content", metadata=metadata)
            ids = [str(d.id()) for d in docs]
            df["id"] = ids

        if "id" not in actual_metadata:
            actual_metadata += ["id"]

        colls = self.list_collections(empty=True)
        coll_name = self.config.collection_name
        if (
            coll_name not in colls
            or self.client.open_table(coll_name).head(1).shape[0] == 0
        ):
            # collection either doesn't exist or is empty, so replace it
            # and set new schema from df
            self.client.create_table(
                self.config.collection_name,
                data=df,
                mode="overwrite",
                on_bad_vectors="drop",
            )
            doc_cls = dataframe_to_document_model(
                df,
                content=content,
                metadata=actual_metadata,
                exclude=["vector"],
            )
            self.config.document_class = doc_cls  # type: ignore
            self._setup_schemas(doc_cls)  # type: ignore
            tbl = self.client.open_table(self.config.collection_name)
            # We assume "content" is available as top-level field
            if "content" in tbl.schema.names:
                tbl.create_fts_index("content", replace=True)
        else:
            # collection exists and is not empty, so append to it
            tbl = self.client.open_table(self.config.collection_name)
            tbl.add(df)
            if "content" in tbl.schema.names:
                tbl.create_fts_index("content", replace=True)

    def delete_collection(self, collection_name: str) -> None:
        self.client.drop_table(collection_name)

    def _lance_result_to_docs(self, result: LanceVectorQueryBuilder) -> List[Document]:
        if self.is_from_dataframe:
            df = result.to_pandas()
            return dataframe_to_documents(
                df,
                content="content",
                metadata=self.df_metadata_columns,
                doc_cls=self.config.document_class,
            )
        else:
            records = result.to_arrow().to_pylist()
            return self._records_to_docs(records)

    def _records_to_docs(self, records: List[Dict[str, Any]]) -> List[Document]:
        if self.config.flatten:
            docs = [
                self.unflattened_schema(**nested_dict_from_flat(rec)) for rec in records
            ]
        else:
            try:
                docs = [self.schema(**rec) for rec in records]
            except ValidationError as e:
                raise ValueError(
                    f"""
                Error validating LanceDB result: {e}
                HINT: This could happen when you're re-using an 
                existing LanceDB store with a different schema.
                Try deleting your local lancedb storage at `{self.config.storage_path}`
                re-ingesting your documents and/or replacing the collections.
                """
                )

        doc_cls = self.config.document_class
        doc_cls_field_names = doc_cls.model_fields.keys()
        return [
            doc_cls(
                **{
                    field_name: getattr(doc, field_name)
                    for field_name in doc_cls_field_names
                }
            )
            for doc in docs
        ]

    def get_all_documents(self, where: str = "") -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        tbl = self.client.open_table(self.config.collection_name)
        pre_result = tbl.search(None).where(where or None)
        return self._lance_result_to_docs(pre_result)

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        if self.config.collection_name is None:
            raise ValueError("No collection name set, cannot retrieve docs")
        _ids = [str(id) for id in ids]
        tbl = self.client.open_table(self.config.collection_name)
        docs = [
            self._lance_result_to_docs(tbl.search().where(f"id == '{_id}'"))
            for _id in _ids
        ]
        return docs

    def similar_texts_with_scores(
        self,
        text: str,
        k: int = 1,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_fn([text])[0]
        tbl = self.client.open_table(self.config.collection_name)
        result = (
            tbl.search(embedding).metric(self.config.distance).where(where).limit(k)
        )
        docs = self._lance_result_to_docs(result)
        # note _distance is 1 - cosine
        if self.is_from_dataframe:
            scores = [
                1 - rec["_distance"] for rec in result.to_pandas().to_dict("records")
            ]
        else:
            scores = [1 - rec["_distance"] for rec in result.to_arrow().to_pylist()]
        if len(docs) == 0:
            logger.warning(f"No matches found for {text}")
            return []
        if settings.debug:
            logger.info(f"Found {len(docs)} matches, max score: {max(scores)}")
        doc_score_pairs = list(zip(docs, scores))
        self.show_if_debug(doc_score_pairs)
        return doc_score_pairs

    def get_fts_chunks(
        self,
        query: str,
        k: int = 5,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Uses LanceDB FTS (Full Text Search).
        """
        # Clean up query: replace all newlines with spaces in query,
        # force special search keywords to lower case, remove quotes,
        # so it's not interpreted as code syntax
        query_clean = (
            query.replace("\n", " ")
            .replace("AND", "and")
            .replace("OR", "or")
            .replace("NOT", "not")
            .replace("'", "")
            .replace('"', "")
        )

        tbl = self.client.open_table(self.config.collection_name)
        tbl.create_fts_index(field_names="content", replace=True)
        result = (
            tbl.search(query_clean)
            .where(where)
            .limit(k)
            .with_row_id(True)
        )
        docs = self._lance_result_to_docs(result)
        scores = [r["score"] for r in result.to_list()]
        return list(zip(docs, scores))


    def _get_clean_vecdb_schema(self) -> str:
        """Get a cleaned schema of the vector-db, to pass to the LLM
        as part of instructions on how to generate a SQL filter."""
        if len(self.config.filter_fields) == 0:
            filterable_fields = (
                self.client.open_table(self.config.collection_name)
                .search()
                .limit(1)
                .to_pandas(flatten=True)
                .columns.tolist()
            )
            # drop id, vector, metadata.id, metadata.window_ids, metadata.is_chunk
            for fields in [
                "id",
                "vector",
                "metadata.id",
                "metadata.window_ids",
                "metadata.is_chunk",
            ]:
                if fields in filterable_fields:
                    filterable_fields.remove(fields)
            logger.warning(
                f"""
            No filter_fields set in config, so using these fields as filterable fields:
            {filterable_fields}
            """
            )
            self.config.filter_fields = filterable_fields

        if self.is_from_dataframe:
            return self.is_from_dataframe
        schema_dict = clean_schema(
            self.schema,
            excludes=["id", "vector"],
        )
        # intersect config.filter_fields with schema_dict.keys() in case
        # there are extraneous fields in config.filter_fields
        filter_fields_set = set(
            self.config.filter_fields or schema_dict.keys()
        ).intersection(schema_dict.keys())

        # remove 'content' from filter_fields_set, even if it's not in filter_fields_set
        filter_fields_set.discard("content")

        # possible values of filterable fields
        filter_field_values = self.get_field_values(list(filter_fields_set))

        # add field values to schema_dict as another field `values` for each field
        for field, values in filter_field_values.items():
            if field in schema_dict:
                schema_dict[field]["values"] = values
        # if self.config.filter_fields is set, restrict to these:
        if len(self.config.filter_fields) > 0:
            schema_dict = {
                k: v for k, v in schema_dict.items() if k in self.config.filter_fields
            }
        schema = json.dumps(schema_dict, indent=2)

        schema += f"""
        NOTE when creating a filter for a query, 
        ONLY the following fields are allowed:
        {",".join(self.config.filter_fields)} 
        """
        return schema


    def get_field_values(self, fields: list[str]) -> Dict[str, str]:
        """Get string-listing of possible values of each filterable field,
        e.g.
        {
            "genre": "crime, drama, mystery, ... (10 more)",
            "certificate": "R, PG-13, PG, R",
        }
        """
        field_values: Dict[str, Set[str]] = {}
        # make empty set for each field
        for f in fields:
            field_values[f] = set()
        # get all documents and accumulate possible values of each field until 10
        docs = self.get_all_documents()  # only works for vecdbs that support this
        for d in docs:
            # extract fields from d
            doc_field_vals = extract_fields(d, fields)
            for field, val in doc_field_vals.items():
                field_values[field].add(val)
        # For each field make a string showing list of possible values,
        # truncate to 20 values, and if there are more, indicate how many
        # more there are, e.g. Genre: crime, drama, mystery, ... (20 more)
        field_values_list = {}
        for f in fields:
            vals = list(field_values[f])
            n = len(vals)
            remaining = n - 20
            vals = vals[:20]
            if n > 20:
                vals.append(f"(...{remaining} more)")
            # make a string of the values, ensure they are strings
            field_values_list[f] = ", ".join(str(v) for v in vals)
        return field_values_list
    