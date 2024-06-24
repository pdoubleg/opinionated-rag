import os
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import json
import re
from src.utils.graph_logger import GraphLogger
from typing import Any, Dict, List, Union
import time
import openai
from dotenv import load_dotenv

load_dotenv()

green_logger = GraphLogger(name="GRAPH MAKER LOG", color="green_bright").getLogger()
json_parse_logger = GraphLogger(name="GRAPH MAKER ERROR", color="magenta").getLogger()
verbose_logger = GraphLogger(name="GRAPH MAKER VERBOSE", color="blue").getLogger()

class Document(BaseModel):
    text: str
    metadata: dict | None = None

class Ontology(BaseModel):
    labels: List[str | Dict]
    relationships: List[str]

    def dump(self):
        if len(self.relationships) == 0:
            return self.model_dump(exclude=["relationships"])
        else:
            return self.model_dump()
        
        
class Node(BaseModel):
    label: str
    name: str

class Edge(BaseModel):
    node_1: Node
    node_2: Node
    relationship: str
    metadata: dict = {}
    order: int | None = None
    
    
class KnowledgeGraph(BaseModel):
    edges: List[Edge] = Field(..., default_factory=list)
    
    @property
    def to_pandas(self):
        kg_dict = {
            "node_1": [n.node_1.name for n in self.edges],
            "node_2": [n.node_2.name for n in self.edges],
            "edge": [n.relationship for n in self.edges],
            "node_1_type": [n.node_1.label for n in self.edges],
            "node_2_type": [n.node_2.label for n in self.edges],
        }
        return pd.DataFrame(kg_dict)


default_ontology = Ontology(
    labels=[
        {"Organization": "Name of an organization."},
        {"Service": "Name of a provided service."},
        {"Team": "Name of the team providing a service."},
        {"Department": "Name of an internal department."},
        "Document",
        "Concept",
        "Issue",
        "Question",
        "Request",
    ],
    relationships=["Relationship between Any two labeled entities"],
)


class GraphMaker:

    def __init__(
        self,
        ontology: Ontology = default_ontology,
        llm_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        verbose: bool = False,
    ):
        self._ontology = ontology
        self._llm_client = llm_client
        self._model = "gpt-4o"
        self._verbose = verbose
        if self._verbose:
            verbose_logger.setLevel("INFO")
        else:
            verbose_logger.setLevel("DEBUG")

    def user_message(self, text: str) -> str:
        return f"input text: ```\n{text}\n```"

    def system_message(self) -> str:
        return (
            "You are an expert at creating Knowledge Graphs. "
            "Consider the following ontology. \n"
            f"{self._ontology} \n"
            "The user will provide you with an input text delimited by ```. "
            "Extract all the entities and relationships from the user-provided text as per the given ontology. Do not use any previous knowledge about the context."
            "Remember there can be multiple direct (explicit) or implied relationships between the same pair of nodes. "
            "Be consistent with the given ontology. Use ONLY the labels and relationships mentioned in the ontology. "
            "Format your output as a json with the following schema. \n"
            "[\n"
            "   {\n"
            '       node_1: Required, an entity object with attributes: {"label": "as per the ontology", "name": "Name of the entity"},\n'
            '       node_2: Required, an entity object with attributes: {"label": "as per the ontology", "name": "Name of the entity"},\n'
            "       relationship: Describe the relationship between node_1 and node_2 as per the context, in a few sentences.\n"
            "   },\n"
            "]\n"
            "Do not add any other comment before or after the json. Respond ONLY with a well formed json that can be directly read by a program."
        )

    def generate(self, text: str) -> str:
        response = self._llm_client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": self.system_message(),
                },
                {
                    "role": "user",
                    "content": self.user_message(text=text),
                },
            ],
        )
        return response.choices[0].message.content
       

    def parse_json(self, text: str):
        green_logger.info(f"Trying JSON Parsing: \n{text}")
        try:
            parsed_json = json.loads(text)
            green_logger.info(f"JSON Parsing Successful!")
            return parsed_json
        except json.JSONDecodeError as e:
            json_parse_logger.info(f"JSON Parsing failed with error: { e.msg}")
            verbose_logger.info(f"FAULTY JSON: {text}")
            return None

    def manually_parse_json(self, text: str):
        green_logger.info(f"Trying Manual Parsing: \n{text}")
        pattern = r"\}\s*,\s*\{"
        stripped_text = text.strip("\n[{]} ")
        # Split the json string into string of objects
        splits = re.split(pattern, stripped_text, flags=re.MULTILINE | re.DOTALL)
        # reconstruct object strings
        obj_string_list = list(map(lambda x: "{" + x + "}", splits))
        edge_list = []
        for string in obj_string_list:
            try:
                edge = json.loads(string)
                edge_list.append(edge)
            except json.JSONDecodeError as e:
                json_parse_logger.info(f"Failed to Parse the Edge: {string}\n{e.msg}")
                verbose_logger.info(f"FAULTY EDGE: {string}")
                continue
        green_logger.info(f"Manually extracted {len(edge_list)} Edges")
        return edge_list

    def json_to_edge(self, edge_dict):
        try:
            edge = Edge(**edge_dict)
        except ValidationError as e:
            json_parse_logger.info(
                f"Failed to parse the Edge: \n{e.errors(include_url=False, include_input=False)}"
            )
            verbose_logger.info(f"FAULTY EDGE: {edge_dict}")
            edge = None
        finally:
            return edge

    def from_text(self, text):
        response = self.generate(text)
        verbose_logger.info(f"LLM Response:\n{response}")

        json_data = self.parse_json(response)
        if not json_data:
            json_data = self.manually_parse_json(response)

        edges = [self.json_to_edge(edg) for edg in json_data]
        edges = list(filter(None, edges))
        return KnowledgeGraph(
            edges=edges,
        )

    def from_document(
        self, doc: Document, order: Union[int, None] = None
    ) -> List[Edge]:
        verbose_logger.info(f"Using Ontology:\n{self._ontology}")
        graph = self.from_text(doc.text)
        for edge in graph:
            edge.metadata = doc.metadata
            edge.order = order
        return KnowledgeGraph(
            edges=graph,
        )

    def from_documents(
        self,
        docs: List[Document],
        order_attribute: Union[int, None] = None,
        delay_s_between=0,
    ) -> List[Edge]:
        graph: List[Edge] = []
        for index, doc in enumerate(docs):
            # defines the chronology / order in which the documents will be interpreted.
            order = getattr(doc, order_attribute) if order_attribute else index
            green_logger.info(f"Document: {index+1}")
            subgraph = self.from_document(doc, order)
            graph = [*graph, *subgraph]
            time.sleep(delay_s_between)
        return KnowledgeGraph(
            edges=graph,
        )
    
