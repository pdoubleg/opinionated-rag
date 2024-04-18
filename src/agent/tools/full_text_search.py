


from typing import List, Optional
import lancedb
import pandas as pd
from src.search.base import SearchEngine, SearchEngineConfig, SearchType

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


DATA_PATH = "data/splade.parquet"


class FTSConfig(SearchEngineConfig):
    """Configuration for Full Text Search (FTS) engine."""
    
    type: SearchType = SearchType.KEYWORD
    data_path: str = DATA_PATH
    text_column: str = "body"
    embedding_column: str = "openai_embeddings"


class FTSSearchEngine(SearchEngine):
    """Full Text Search (FTS) engine using Tantivy via LanceDB."""
    
    def __init__(self, config: FTSConfig = FTSConfig()):
        super().__init__(config)
        self.config: FTSConfig = config
        
        
class FTSSearch:
    """
    Full Text Search (FTS) class using Tantivy via LanceDB.
    
    At query time, constructs a FTS index from one or more DataFrame columns containing text.
    Even though this class does not use embeddings, they are required by lance 
    to build a database. 
        
    Note: This class accepts SQL-like filter statements at query time.
        Supported SQL expressions:
        *  >, >=, <, <=, =
        *  AND, OR, NOT
        *  IS NULL, IS NOT NULL
        *  IS TRUE, IS NOT TRUE, IS FALSE, IS NOT FALSE
        *  IN
        *  LIKE, NOT LIKE 
        
    General info: https://lancedb.github.io/lancedb/fts/
    Filtering info: https://lancedb.github.io/lancedb/sql/#pre-and-post-filtering

    Args:
        df (pd.DataFrame): Input DataFrame containing the documents.
        embedding_column (str): Name of the column containing the embeddings.
        text_column (str): Name of the column containing the text to index.
        text_column_list (list[str], optional): List of columns containing text to index.
            Columns will be combined into a single index for querying.

    Raises:
        ValueError: If the specified embedding_column is not present in the DataFrame.
        ValueError: If neither text_column nor text_column_list is provided.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        embedding_column: str,
        text_column: Optional[str] = None,
        text_column_list: Optional[List[str]] = None,
    ) -> None:
        if embedding_column not in df.columns:
            raise ValueError(f"Embedding column '{embedding_column}' not found in the DataFrame.")
        if text_column is None and text_column_list is None:
            raise ValueError("At least one of text_column or text_column_list must be provided.")
        
        self._df = df
        self.embedding_column = embedding_column
        self.text_column = text_column
        self.text_column_list = text_column_list
        self.uri = "../temp-lancedb/temp_table.lance"

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns the DataFrame with the embedding column renamed to 'vector'.
        
        This is required by LanceDB for using pre-computed embeddings.

        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        df = self._df.rename(columns={self.embedding_column: "vector"})
        return df

    def query_similar_documents(
        self,
        query: str,
        top_k: int = 20,
        filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query similar documents using Full Text Search (FTS).

        Args:
            query (str): Query string for searching similar documents.
            top_k (int): Number of top similar documents to retrieve (default: 20).
            filter (Optional[str]): SQL-like filter condition for the search (default: None).

        Returns:
            pd.DataFrame: DataFrame containing the top k similar documents.

        Example:
            >>> fts_search = FTSSearch(df, "embeddings", "text")
            >>> results = fts_search.query_similar_documents("example query", top_k=10)
        """
        db = lancedb.connect(self.uri)
        try:
            table = db.create_table("temp_lance", data=self.df, mode="create")
        except:
            table = db.create_table("temp_lance", data=self.df, mode="overwrite")

        if self.text_column_list:
            text_to_index = self.text_column_list
        else:
            text_to_index = self.text_column
        
        table.create_fts_index(text_to_index, replace=True)

        # Clean up query: replace all newlines with spaces in query,
        # force special search keywords to lower case, remove quotes,
        # so it's not interpreted as search syntax
        query_clean = (
            query.replace("\n", " ")
            .replace("AND", "and")
            .replace("OR", "or")
            .replace("NOT", "not")
            .replace("'", "")
            .replace('"', "")
        )
        if filter is not None:
            result = table.search(query_clean, query_type="fts").where(filter).limit(top_k).to_pandas()
        else:
            result = table.search(query_clean, query_type="fts").limit(top_k).to_pandas()
        result.rename(columns={"vector": self.embedding_column}, inplace=True)
        logger.info(
            f"Full Text Search (FTS) search yielded a DataFrame with {len(result):,} rows"
        )

        return result