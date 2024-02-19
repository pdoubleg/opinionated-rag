import os
import re
import numpy as np
import pandas as pd
import tiktoken
import openai
from collections import Counter
from typing import List, Literal, Tuple
from thefuzz import fuzz, process

import warnings

from src.embedding_models.models import OpenAIEmbeddingsConfig, OpenAIEmbeddings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.filterwarnings('ignore')
import logging
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    
from pydantic import BaseModel, Field


df = pd.read_parquet('data/forward_citations.parquet')
df.dropna(subset=['Relevant Excerpt'], inplace=True)
print(df.isna().sum())

embedding_cofig = OpenAIEmbeddingsConfig(
    model_type="openai",
    model_name="text-embedding-ada-002",
    dims=1536,
)

embedding_model = OpenAIEmbeddings(embedding_cofig)

