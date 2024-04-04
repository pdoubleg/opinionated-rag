import asyncio
import pandas as pd
import numpy as np
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import patch, MagicMock

from src.agent.tools.semantic_search import SemanticSearch, Filter


class TestFilter(TestCase):
    def test_display_filter_with_where(self):
        """Test display_filter property with where condition."""
        filter_instance = Filter(where={"state": "active"}, name="Active State")
        self.assertEqual(filter_instance.display_filter, "Search Criteria: State = active")

    def test_display_filter_without_where(self):
        """Test display_filter property without where condition."""
        filter_instance = Filter(name="All States")
        self.assertEqual(filter_instance.display_filter, "Search Criteria: All States")

    def test_filter_key_value_with_where(self):
        """Test filter_key and filter_value properties with where condition."""
        filter_instance = Filter(where={"state": "active"}, name="Active State")
        self.assertEqual(filter_instance.filter_key, "state")
        self.assertEqual(filter_instance.filter_value, "active")

    def test_filter_key_value_without_where(self):
        """Test filter_key and filter_value properties without where condition."""
        filter_instance = Filter(name="All States")
        self.assertEqual(filter_instance.filter_key, "")
        self.assertEqual(filter_instance.filter_value, "")


class TestSemanticSearch(IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up test case with a sample dataframe and SemanticSearch instance."""
        self.df = pd.DataFrame({
            "text": ["This is a test", "Another test", "Yet another test"],
            "embeddings": [np.random.rand(512) for _ in range(3)]
        })
        self.semantic_search = SemanticSearch(df=self.df)

    @patch("src.agent.tools.semantic_search.OpenAI")
    def test_get_embedding(self, mock_openai):
        """Test get_embedding method."""
        mock_openai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=np.random.rand(512))])
        text = "This is a test"
        embedding = self.semantic_search.get_embedding(text)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (512,))

    async def test_aget_embedding(self):
        """Test aget_embedding method."""
        with patch("src.agent.tools.semantic_search.AsyncOpenAI") as mock_aopenai:
            mock_aopenai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=np.random.rand(512))])
            text = "This is a test"
            embedding = await self.semantic_search.aget_embedding(text)
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (512,))

    def test_normalize_embeddings(self):
        """Test normalize_embeddings static method."""
        embeddings = np.random.rand(3, 512)
        normalized_embeddings = SemanticSearch.normalize_embeddings(embeddings)
        norms = np.linalg.norm(normalized_embeddings, axis=1)
        np.testing.assert_almost_equal(norms, np.ones_like(norms))

    def test_build_faiss_index(self):
        """Test build_faiss_index method."""
        embeddings = np.random.rand(3, 512)
        index = self.semantic_search.build_faiss_index(embeddings, use_cosine_similarity=True)
        self.assertIsNotNone(index)

    @patch("src.agent.tools.semantic_search.faiss.IndexFlatIP")
    def test_search_faiss_index(self, mock_index):
        """Test search_faiss_index method."""
        mock_index.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))
        embeddings = np.random.rand(3, 512)
        index = self.semantic_search.build_faiss_index(embeddings, use_cosine_similarity=True)
        embedding = np.random.rand(512)
        indices, sim_scores = self.semantic_search.search_faiss_index(index, embedding, top_n=2, use_cosine_similarity=True, similarity_threshold=0.95)
        self.assertEqual(len(indices), 2)
        self.assertEqual(len(sim_scores), 2)
