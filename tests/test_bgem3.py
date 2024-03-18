import unittest
from typing import List

from src.embedding_models.models import BGE_M3Embeddings


class TestBGE_M3Embeddings(unittest.TestCase):

    def setUp(self):
        self.model = BGE_M3Embeddings(model_name='BAAI/bge-m3', use_fp16=False, batch_size=12, max_length=8192)

    def test_instance_creation(self):
        self.assertIsInstance(self.model, BGE_M3Embeddings)

    def test_query_embedding(self):
        result = self.model._get_query_embedding("test query")
        self.assertIsInstance(result, List)
        self.assertTrue(all(isinstance(x, float) for x in result))

    def test_text_embedding(self):
        result = self.model._get_text_embedding("test text")
        self.assertIsInstance(result, List)
        self.assertTrue(all(isinstance(x, float) for x in result))

if __name__ == '__main__':
    unittest.main()
