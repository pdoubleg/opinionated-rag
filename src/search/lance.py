import datetime
import re
from typing import List, Optional, Union
import numpy as np
import json
import pyarrow as pa
from pydantic import Field, BaseModel, create_model, model_validator, ConfigDict
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import (
    EmbeddingFunctionRegistry,
    TextEmbeddingFunction,
)

from lancedb.embeddings import (
    EmbeddingFunctionRegistry,
    TextEmbeddingFunction,
)


class AbstractModel(BaseModel):
    name: str

    @classmethod
    @property
    def __entity_name__(cls):
        s = cls.model_json_schema(by_alias=False)
        return s["title"]

    @classmethod
    @property
    def __entity_namespace__(cls):
        """
        Takes the namespace from config or module - returns default if nothing else e.g. dynamic modules
        """
        # convention
        namespace = cls.__module__.split(".")[-1]
        # for now we use the default namespace for anything in entities
        return (
            namespace if namespace not in ["model", "__main__", "entity"] else "default"
        )

    @classmethod
    @property
    def __fullname__(cls):
        return f"{cls.__entity_namespace__}/{cls.__entity_name__}"

    @classmethod
    @property
    def __key_field__(cls):
        s = cls.model_json_schema(by_alias=False)
        key_props = [k for k, v in s["properties"].items() if v.get("is_key")]
        if len(key_props):
            return key_props[0]
        # convention
        return "name"

    @classmethod
    @property
    def __about__(cls):
        if hasattr(cls, "config"):
            c = cls.config
            return getattr(c, "about", "")

    @classmethod
    def create_model(cls, name, namespace=None, **fields):
        """
        For dynamic creation of models for the type systems
        create something that inherits from the class and add any extra fields

        when we create models we should audit them somewhere in the system because we can later come back and make them
        """
        namespace = namespace or cls.__entity_namespace__
        return create_model(name, **fields, __module__=namespace, __base__=cls)

    @classmethod
    def create_model_from_pyarrow(
        cls, name, py_arrow_schema, namespace=None, key_field=None, **kwargs
    ):
        def _Field(name, fi):
            d = dict(fi)
            t = d.pop("type")
            # field = Field (t, None, is_key=True) if name == key_field else Field
            return (t, None)

        namespace = namespace or cls.__entity_namespace__
        fields = map_field_types_from_pa_schema(py_arrow_schema)
        fields = {k: _Field(k, v) for k, v in fields.items()}
        return create_model(name, **fields, __module__=namespace, __base__=cls)

    @classmethod
    def create_model_from_data(cls, name, data, namespace=None, **kwargs):
        schema = pa.Table.from_pandas(data).schema
        data = data.to_dict("records")
        Model = cls.create_model_from_pyarrow(
            name=name,
            namespace=namespace,
            py_arrow_schema=schema,
        )

        return Model
   

"""
Schema tools / pyarrow to mpydantic etc. crude    
"""


def map_pyarrow_type_info(field_type, name=None):
    """
    Load not only the type but extra type metadata from pyarrow that we use in Pydantic type
    """
    t = map_pyarrow_type(field_type, name=name)
    d = {"type": t}
    if pa.types.is_fixed_size_list(field_type):
        d["fixed_size_length"] = field_type.list_size
    return d


def map_pyarrow_type(field_type, name=None):
    """
    basic mapping between pyarrow types and typing info for some basic types we use in stores
    """

    if pa.types.is_fixed_size_list(field_type):
        return List[map_pyarrow_type(field_type.value_type)]
    if pa.types.is_map(field_type):
        return dict
    if pa.types.is_list(field_type):
        return list
    else:
        if field_type in [pa.string(), pa.utf8(), pa.large_string()]:
            return str
        if field_type == pa.string():
            return str
        if field_type in [pa.int16(), pa.int32(), pa.int8(), pa.int64()]:
            return int
        if field_type in [pa.float16(), pa.float32(), pa.float64()]:
            return float
        if field_type in [pa.date32(), pa.date64()]:
            return datetime.datetime
        if field_type in [pa.bool_()]:
            return bool
        if field_type in [pa.binary()]:
            return bytes
        if pa.types.is_timestamp(field_type):
            return datetime.datetime

        if "timestamp" in str(field_type):
            return Optional[str]
        return Optional[str]

        raise NotImplementedError(f"We dont handle {field_type} ({name})")


def map_field_types_from_pa_schema(schema):
    """
    given a pyarrow schema return type info so we can create the Pydantic Model from the pyarrow table
    """

    return {f.name: map_pyarrow_type_info(f.type, f.name) for f in schema}



"""
Embeddings
"""

EMBEDDING_REGISTRY = EmbeddingFunctionRegistry.get_instance()
INSTRUCT_EMBEDDING_VECTOR_LENGTH = 768
OPEN_AI_EMBEDDING_VECTOR_LENGTH = 1536


@EMBEDDING_REGISTRY.register("openai")
class OpenAIEmbeddings(TextEmbeddingFunction):
    name: str = "text-embedding-ada-002"

    def ndims(self):
        return OPEN_AI_EMBEDDING_VECTOR_LENGTH

    def generate_embeddings(
        self, texts: Union[List[str], np.ndarray]
    ) -> List[np.array]:
        openai = self.safe_import("openai")
        rs = openai.embeddings.create(input=texts, model=self.name).data
        emb = [v.embedding for v in rs]
        return emb

    def __call__(
        self, texts: Union[List[str], np.ndarray]
    ) -> List[np.array]:
        return self.generate_embeddings(texts)

    def compute_source_embeddings_with_retry(
        self, texts: Union[List[str], np.ndarray], *args, **kwargs
    ) -> List:
        return super().compute_source_embeddings(texts, *args, **kwargs)


@EMBEDDING_REGISTRY.register("instruct")
class InstructEmbeddings(TextEmbeddingFunction):
    name: str = "hkunlp/instructor-base"
    batch_size: int = 32
    device: str = "cpu"
    show_progress_bar: bool = True
    normalize_embeddings: bool = True
    quantize: bool = False

    source_instruction: str = "represent the document for retrieval"
    query_instruction: str = (
        "represent the document for retrieving the most similar documents"
    )

    def ndims(self):
        return INSTRUCT_EMBEDDING_VECTOR_LENGTH

    def compute_query_embeddings(self, query: str, *args, **kwargs) -> List[np.array]:
        return self.generate_embeddings([[self.query_instruction, query]])

    def compute_source_embeddings(
        self, texts: Union[List[str], np.ndarray], *args, **kwargs
    ) -> List[np.array]:
        texts = self.sanitize_input(texts)
        texts_formatted = []
        for text in texts:
            texts_formatted.append([self.source_instruction, text])
        return self.generate_embeddings(texts_formatted)

    def generate_embeddings(self, texts: List) -> List:
        model = self.get_model()
        res = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        ).tolist()
        return res

    def __call__(
        self, texts: Union[List[str], np.ndarray]
    ) -> List[np.array]:
        return self.generate_embeddings(texts)

    def compute_source_embeddings_with_retry(
        self, texts: Union[List[str], np.ndarray], *args, **kwargs
    ) -> List:
        return super().compute_source_embeddings(texts, *args, **kwargs)

    # @weak_lru(maxsize=1)
    def get_model(self):
        instructor_embedding = self.safe_import(
            "InstructorEmbedding", "InstructorEmbedding"
        )
        torch = self.safe_import("torch", "torch")

        model = instructor_embedding.INSTRUCTOR(self.name)
        if self.quantize:
            if (
                "qnnpack" in torch.backends.quantized.supported_engines
            ):  # fix for https://github.com/pytorch/pytorch/issues/29327
                torch.backends.quantized.engine = "qnnpack"
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        return model


class Embeddings:
    """
    an opinionated wrapper around the embedding function loaders - may deprecate
    """

    def __init__(self):
        self._registry = {}

    def __getitem__(self, key):
        if key in self._registry:
            return self._registry[key]
        else:
            if key == "openai":
                open_ai_embedding = EMBEDDING_REGISTRY.get("openai").create()
                self._registry[key] = open_ai_embedding
                return open_ai_embedding

            if key == "instruct":
                instruct_embedding = EMBEDDING_REGISTRY.get("instruct").create()
                self._registry[key] = instruct_embedding
                return instruct_embedding

    @property
    def openai(self):
        return self["openai"]

    @property
    def instruct(self):
        return self["instruct"]


EmbeddingFunctions = Embeddings()

class AbstractContentModel(LanceModel, AbstractModel):
    """
    MyModel = AbstractContentModel(name='test', content='test', vector=nd.zeros(EmbeddingFunctions.openai.ndims()))
    """

    vector: Vector(
        EmbeddingFunctions.openai.ndims()
    ) = EmbeddingFunctions.openai.VectorField()
    content: str = EmbeddingFunctions.openai.SourceField()
    updated_by: Optional[str] = None
    updated_at: Optional[str]
    # the depth in a text tree. 0 is original content, increasing depth splits text and negative depth summarizes
    aperture: int = 0
    refs: List[str] = []
    document: str = ""
    cluster_id: str = ""

    # this is a convenience for testing for now - todo inspect the embedding and embed the content if its blank
    @model_validator(mode="before")
    def _init_vector(cls, values):
        if "vector" not in values:
            # im doing this here because its good for testing BUt i think lance should call the embedding function somehow
            values["vector"] = np.zeros(
                EmbeddingFunctions.openai.ndims()
            )  # EmbeddingFunctions.openai([values["content"]])[0]
        if "updated_at" not in values:
            values["updated_at"] = datetime.now()
        return values


class FunctionRegistryRecord(AbstractContentModel):
    description: str
    # search_embedding:
    type: str
    namespace: str
    entity_name: str
    metadata: str

    @model_validator(mode="before")
    def _init_content(cls, values):
        values[
            "content"
        ] = f"{values['description']} /n/nIn relation to entity {values['type']}, {values['entity_name']}, {values['namespace']}"

        return values

class SchemaOrgVectorEntity(AbstractContentModel):
    """
    A helper for mapping schema.org data usually scrapped in Json+LD
    we want tp put this in various pydantic types
    """

    model_config = ConfigDict(
        # these are excluded from the output we send to data stores for now
        EXCLUDE_ATTRIBUTES=["comment", "sameAs"],
        # these are mapped from objects that are complex with @id to just have the string value (s)
        PULL_IDs=["image", "video", "publisher", "aggregate_rating"],
        FLATTEN_COMPLEX_NAMED_TEXT_TYPES=["HowToStep"],
    )

    @model_validator(mode="before")
    def pull_ids(cls, values):
        for key in values.keys():
            if key in SchemaOrgVectorEntity.model_config["PULL_IDs"]:
                if not isinstance(values, dict):
                    raise NotImplementedError("multiples todo for mapping @id stuff")
                values[key] = values.get("name", values.get("@id"))
        return values

    @model_validator(mode="before")
    def _coerce_str(cls, values):
        # this is just for testing
        for key in values.keys():
            values[key] = str(values[key])

        # todo - here we will just dump everything for now
        values["content"] = json.dumps(values, default=str)

        return values

    @staticmethod
    def should_exclude(k, override_exclude=None):
        return k in (
            override_exclude or SchemaOrgVectorEntity.model_config["EXCLUDE_ATTRIBUTES"]
        )

    @classmethod
    def create_model_from_schema(
        cls,
        name,
        schema,
        namespace=None,
        exclude_fields_snake_case=None,
        text_fields=None,
    ):
        # from stringcase import snakecase
        def snakecase(s):
            # Replace spaces and underscores with a single underscore
            s = re.sub(r"[-_ ]+", "_", s)
            # Remove non-alphanumeric characters except underscores
            s = re.sub(r"[^a-zA-Z0-9_]", "", s)
            return s.lower()

        """
        schema org attributes but snake case and cleaned
        for now we will not type until we have a use case for columnar data      
        schema can be a type or an instance in the json format of python typed objects
        if its a type from schema org it will be better for generating but for simple use cases it does not matter  
        TODO: we will combine text fields into one that we use for the VStore - at the moment its assumed that there is at least a text field which is typical enough for schema.org
        """

        field_norm = lambda s: snakecase(s.lstrip("@"))

        fields = {
            field_norm(field_name): (
                # ToDo manage types properly
                str,
                Field(alias=field_name),
            )
            for field_name, type_info in schema.items()
        }

        # field names that we want to generate
        # if this were written out we could use underscores but this is a hidden dynamic type so we dont generate these excluded fields at all
        fields = {
            k: v
            for k, v in fields.items()
            if not SchemaOrgVectorEntity.should_exclude(k, exclude_fields_snake_case)
        }

        if text_fields:
            pass  # add the validator
            # we need to add a validator to merge these fields into a text field

        namespace = namespace or cls.__entity_namespace__
        return create_model(name, **fields, __module__=namespace, __base__=cls)
    
    

class InstructEmbeddingContentModel(AbstractContentModel):
    """ """

    vector: Vector(
        EmbeddingFunctions.instruct.ndims()
    ) = EmbeddingFunctions.instruct.VectorField()
    content: str = EmbeddingFunctions.instruct.SourceField()

    # this is a convenience for testing for now - todo inspect the embedding and embed the content if its blank
    @model_validator(mode="before")
    def _init_vector(cls, values):
        if "vector" not in values:
            # init a model for testing loads
            values["vector"] = np.zeros(
                EmbeddingFunctions.instruct.ndims()
            )  # EmbeddingFunctions.instruct([values["content"]])[0]
        if "updated_at" not in values:
            values["updated_at"] = datetime.now()
        return values
    
