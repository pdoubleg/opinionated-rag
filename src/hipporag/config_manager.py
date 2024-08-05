from pydantic_settings import BaseSettings
from pathlib import Path


class ConfigManager(BaseSettings):
    """
    Configuration manager for the HippoRAG project.

    """

    dataset_name: str = "ho3_sample"
    input_filename: str = './hipporag_data/ho3_sample/corpus.json'
    string_filename: str = ""
    graph_type: str = "facts_and_sim"
    base_directory: Path = Path(".").resolve()
    data_directory: Path = base_directory / "hipporag_data/ho3_sample"
    output_directory: Path = data_directory / "output"
    vector_directory: Path = data_directory / "lm_vectors"
    llm_model: str = "gpt-4o-mini"
    run_ner: bool = True
    phrase_type: str = "ents_only"
    extraction_type: str = "ner"
    ner_temperature: float = 0.1
    num_passages: str | int = "all"
    text_to_embed_column: str = "strings"
    doc_column: str = "text"
    chunk_column: str = "chunk"
    retriever_name: str = "facebook/contriever"
    doc_ensemble: bool = False
    dpr_only: bool = False
    damping: float = 0.1
    recognition_threshold: float = 0.9
    num_processes: int = 1
    inter_triple_weight: float = 1.0
    similarity_max: float = 1.0
    create_graph_flag: bool = False
    cosine_sim_edges: bool = False
    sim_threshold: float = 0.9
    node_specificity: bool = False
    graph_alg: str = "ppr"

    @property
    def encoded_strings_path(self) -> Path:
        """Path to the encoded strings file."""
        return self.data_directory / "encoded_strings.txt"

    @property
    def vectors_path_pattern(self) -> str:
        """Pattern for vector files."""
        return str(self.data_directory / "vecs_*.p")

    @property
    def nearest_neighbors_path(self) -> Path:
        """Path to the nearest neighbors file."""
        return self.data_directory / "nearest_neighbors.p"

    @property
    def openie_results_path(self) -> Path:
        """Path to the OpenIE results file."""
        return self.output_directory / "openie_results.json"

    @property
    def named_entity_output_path(self) -> Path:
        """Path to the named entity output file."""
        return self.output_directory / "named_entity_output.tsv"

    @property
    def graph_path(self) -> Path:
        """Path to the G output file."""
        return self.output_directory / "graph_fact_doc_edges.p"

    @property
    def docs_to_facts_mat_path(self) -> Path:
        """Path to the docs_to_facts_mat matrix."""
        return self.output_directory / "docs_to_facts_mat.p"

    @property
    def facts_to_phrases_mat_path(self) -> Path:
        """Path to the facts_to_phrases_mat matrix."""
        return self.output_directory / "facts_to_phrases_mat.p"

    @property
    def kb_to_kb_path(self) -> Path:
        """Path to the kb_to_kb output file."""
        return self.output_directory / "kb_to_kb.tsv"

    @property
    def rel_kb_to_kb_path(self) -> Path:
        """Path to the rel_kb_to_kb output file."""
        return self.output_directory / "rel_kb_to_kb.tsv"

    @property
    def query_to_kb_path(self) -> Path:
        """Path to the query_to_kb output file."""
        return self.output_directory / "query_to_kb.tsv"

    @property
    def synonym_candidates_path(self) -> Path:
        """Path to the synonym_candidates file."""
        return self.output_directory / "similarity_edges.p"

    @property
    def relations_path(self) -> Path:
        """Path to the relations file."""
        return self.output_directory / "relation_dict.p"

    @property
    def graph_plus_path(self) -> Path:
        """Path to the graph_plus file."""
        return self.output_directory / "graph_plus.p"

    @property
    def graph_json_path(self) -> Path:
        """Path to the graph_json file."""
        return self.output_directory / "graph.json"

    @property
    def doc_to_phrases_mat_path(self) -> Path:
        """Path to the doc_to_phrases matrix."""
        return self.output_directory / "doc_to_phrases_mat.p"

    @property
    def phrase_to_num_doc_path(self) -> Path:
        """Path to the phrase_to_num_doc matrix."""
        return self.output_directory / "phrase_to_num_doc.p"

    @property
    def embeddings_cache_path(self) -> Path:
        """Path to the embeddings cache."""
        return self.output_directory / "embeddings_cache.p"
