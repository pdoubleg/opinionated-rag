from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field

from pydantic_settings import BaseSettings
from src.search.models import SearchType

from src.utils.gen_utils import DataFrameCache
from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


class SearchType(str, Enum):
    """Enum representing different types of search engines."""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    SPLADE = "splade"


class SearchEngineConfig(BaseSettings):
    """Configuration class for search engines.

    Attributes:
        type (SearchType): The type of search engine.
        data_path (Optional[Path]): Path to the data file.
        text_column (Optional[str]): Name of the text column in the data.
        embedding_column (Optional[str]): Name of the embedding column in the data.
    """
    type: SearchType
    data_path: Path | None = None
    text_column: str | None = None
    embedding_column: str | None = None
    
    
class SearchEngineBase(ABC):
    """Abstract base class for search engines."""
    
    @abstractmethod
    def create(config: SearchEngineConfig):

        pass
    
    @abstractmethod
    def query_similar_documents(query: str) -> pd.DataFrame:
        
        pass


class SearchEngine(SearchEngineBase):
    """Abstract factory class for creating search engines.

    This class follows the Dependency Inversion Principle, which promotes loose coupling and easier
    testability and maintainability. It allows swapping out different search engine implementations
    without affecting the code that uses the SearchEngine factory to create objects.

    To add a new search engine:
    1. Create a configuration class subclassing SearchEngineConfig.
    2. Implement the specific search engine class inheriting from SearchEngineBase.
    3. Update the create() method in this class to handle the new search engine type.
    4. Optionally, add the name of your search engine to the SearchType enum.

    Attributes:
        config (SearchEngineConfig): Configuration for the search engine.
    """

    def __init__(self, config: SearchEngineConfig):
        self.config = config

    @staticmethod
    def create(config: SearchEngineConfig) -> Optional["SearchEngine"]:
        """Create a new instance of the search engine based on the provided configuration.

        Args:
            config (SearchEngineConfig): Configuration for the search engine.

        Returns:
            Optional[SearchEngine]: The created search engine instance, or None if not supported.
        """
        from src.agent.tools.semantic_search import SemanticSearch
        from src.agent.tools.splade_search import SPLADESparseSearch
        from src.agent.tools.hybrid_search import HybridSearch
        from src.agent.tools.full_text_search import FTSSearch

        cached_df = DataFrameCache(config.data_path).df

        if config.type == SearchType.SEMANTIC:
            return SemanticSearch(
                df=cached_df,
                text_column=config.text_column,
                embedding_column=config.embedding_column,
            )
        if config.type == SearchType.SPLADE:
            return SPLADESparseSearch(
                df=cached_df,
                text_column=config.text_column,
                embedding_column=config.embedding_column,
            )
        if config.type == SearchType.HYBRID:
            return HybridSearch(
                df=cached_df,
                text_column=config.text_column,
                dense_embedding_column=config.dense_embedding_column,
                sparse_embedding_column=config.sparse_embedding_column,
            )
        if config.type == SearchType.KEYWORD:
                return FTSSearch(
                df=cached_df,
                text_column=config.text_column,
                embedding_column=config.embedding_column,
            )
                
    def query_similar_documents(self, query: str) -> pd.DataFrame:

        raise NotImplementedError("query_similar_documents is not implemented in the base SearchEngine class")

