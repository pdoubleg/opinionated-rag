


from typing import List, Optional
import lancedb
import pandas as pd
from src.search.base import SearchEngine, SearchEngineConfig, SearchType

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


DATA_PATH = "data/splade.parquet"


class FTSConfig(SearchEngineConfig):
    type: SearchType = SearchType.KEYWORD
    data_path: str = DATA_PATH
    text_column: str = "body"
    embedding_column: str = "openai_embeddings"


class FTSSearchEngine(SearchEngine):
    def __init__(self, config: FTSConfig = FTSConfig()):
        super().__init__(config)
        self.config: FTSConfig = config
        
        
class FTSSearch:
    """
    Full Text Search (FTS) class for querying similar documents.

    Args:
        df (pd.DataFrame): Input DataFrame containing the documents.
        embedding_column (str): Name of the column containing the embeddings.
        text_column (str): Name of the column containing the text data.

    Raises:
        ValueError: If the specified embedding_column is not present in the DataFrame.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        embedding_column: str,
        text_column: str,
    ) -> None:
        if embedding_column not in df.columns:
            raise ValueError(f"Embedding column '{embedding_column}' not found in the DataFrame.")
        self._df = df
        self.embedding_column = embedding_column
        self.text_column = text_column
        self.uri = "../temp-lancedb/temp_table.lance"

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns the DataFrame with the embedding column renamed to 'vector'.

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
            filter (Optional[str]): Filter condition for the search (default: None).

        Returns:
            pd.DataFrame: DataFrame containing the top similar documents.

        Example:
            >>> fts_search = FTSSearch(df, "embeddings", "text")
            >>> results = fts_search.query_similar_documents("example query", top_k=10)
        """
        db = lancedb.connect(self.uri)
        cols = self.df.columns.tolist()
        try:
            table = db.create_table("temp_lance", data=self.df[cols], mode="create")
        except:
            table = db.create_table("temp_lance", data=self.df[cols], mode="overwrite")

        table.create_fts_index(self.text_column, replace=True)

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