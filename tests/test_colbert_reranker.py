import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch

from src.embedding_models.models import ColbertReranker

class TestColbertReranker(unittest.TestCase):
    @patch('src.embedding_models.models.AutoTokenizer.from_pretrained')
    @patch('src.embedding_models.models.AutoModel.from_pretrained')
    def test_rerank_hybrid(self, mock_model, mock_tokenizer):
        # Prepare the mock for tokenizer and model
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        
        # Setup the return value for the tokenizer's __call__
        mock_tokenizer_instance.return_value = {'input_ids': torch.tensor([[101, 102]])}
        
        # Setup the return value for the model's __call__
        mock_model_output = MagicMock()
        mock_model_output.last_hidden_state = torch.tensor([[[0.1, 0.2, 0.3]]])
        mock_model_instance.return_value = mock_model_output
        
        # Assign the mock instances to the return_value of the from_pretrained method
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model.return_value = mock_model_instance
        
        # Instantiate the ColbertReranker with default parameters
        reranker = ColbertReranker()
        
        # Create mock data for vector_results and fts_results
        vector_results = pd.DataFrame({'content': ['document 1'], 'score': [0.9]})
        fts_results = pd.DataFrame({'content': ['document 2'], 'score': [0.8]})
        
        # Call the rerank_hybrid method
        reranked_results = reranker.rerank_hybrid('query', vector_results, fts_results)
        
        # Assertions to ensure the reranked results are as expected
        self.assertIsInstance(reranked_results, pd.DataFrame)
        self.assertTrue('_relevance_score' in reranked_results.columns)
        self.assertEqual(len(reranked_results), 2)  # Ensure two documents are returned
        self.assertTrue(reranked_results.iloc[0]['_relevance_score'] >= reranked_results.iloc[1]['_relevance_score'])

if __name__ == '__main__':
    unittest.main()