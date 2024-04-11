import unittest
import pandas as pd
from src.search.doc_joiner import DocJoinerDF

class TestDocJoinerDF(unittest.TestCase):
    def setUp(self):
        self.df1 = pd.DataFrame({
            'text': ['doc1', 'doc2', 'doc3'],
            'score': [0.8, 0.6, 0.4],
            'extra': [1, 2, 3]
        })
        self.df2 = pd.DataFrame({
            'text': ['doc2', 'doc3', 'doc4'],
            'score': [0.7, 0.5, 0.3],
            'extra': [4, 5, 6]
        })
        self.df3 = pd.DataFrame({
            'text': ['doc1', 'doc4', 'doc5'],
            'score': [0.9, 0.2, 0.1],
            'extra': [7, 8, 9]
        })

    def test_concatenate(self):
        joiner = DocJoinerDF(join_mode='concatenate')
        result = joiner.run([self.df1, self.df2, self.df3])
        expected = pd.DataFrame({
            'text': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5'],
            'score': [0.9, 0.7, 0.5, 0.3, 0.1],
            'extra': [7, 4, 5, 6, 9]
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_merge(self):
        joiner = DocJoinerDF(join_mode='merge', weights=[0.4, 0.3, 0.3])
        result = joiner.run([self.df1, self.df2, self.df3])
        expected = pd.DataFrame({
            'text': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5'],
            'score': [0.86, 0.64, 0.44, 0.24, 0.03],
            'extra': [7, 4, 5, 8, 9]
        })
        pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=0.01)

    def test_reciprocal_rank_fusion(self):
        joiner = DocJoinerDF(join_mode='reciprocal_rank_fusion', weights=[0.4, 0.3, 0.3])
        result = joiner.run([self.df1, self.df2, self.df3])
        expected = pd.DataFrame({
            'text': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5'],
            'score': [0.0492, 0.0246, 0.0164, 0.0082, 0.0049],
            'extra': [7, 4, 5, 8, 9]
        })
        pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=0.001)
        
    def test_top_k(self):
        joiner = DocJoinerDF(join_mode='concatenate', top_k=3)
        result = joiner.run([self.df1, self.df2, self.df3])
        expected = pd.DataFrame({
            'text': ['doc1', 'doc2', 'doc3'],
            'score': [0.9, 0.7, 0.5],
            'extra': [7, 4, 5]
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_invalid_join_mode(self):
        with self.assertRaises(ValueError):
            DocJoinerDF(join_mode='invalid')

    def test_missing_score_column(self):
        df_missing_score = pd.DataFrame({'text': ['doc1'], 'extra': [1]})
        joiner = DocJoinerDF(join_mode='concatenate')
        with self.assertRaises(ValueError):
            joiner.run([self.df1, df_missing_score])

if __name__ == '__main__':
    unittest.main()
