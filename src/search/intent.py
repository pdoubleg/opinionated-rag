import os
from typing import Optional
import numpy as np
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)


# Danswer custom Deep Learning Models
INTENT_MODEL_VERSION = "danswer/intent-model"
# Intent model max context size
QUERY_MAX_CONTEXT_SIZE = 256


def get_local_intent_model() -> "TFDistilBertForSequenceClassification":
    return TFDistilBertForSequenceClassification.from_pretrained(INTENT_MODEL_VERSION)


def get_intent_model_tokenizer() -> "AutoTokenizer":
    return AutoTokenizer.from_pretrained(INTENT_MODEL_VERSION)


class IntentModel:
    def __init__(
        self,
        model_name: str = INTENT_MODEL_VERSION,
        max_seq_length: int = QUERY_MAX_CONTEXT_SIZE,
    ) -> None:
        self.model_name = model_name
        self.max_seq_length = max_seq_length

    def predict(
        self,
        query: str,
    ) -> list[float]:
        tokenizer = get_intent_model_tokenizer()
        intent_model = get_local_intent_model()

        model_input = tokenizer(
            query, return_tensors="tf", truncation=True, padding=True
        )
        predictions = intent_model(model_input)[0]
        probabilities = tf.nn.softmax(predictions, axis=-1)
        class_percentages = np.round(probabilities.numpy() * 100, 2)

        return list(class_percentages.tolist()[0])
