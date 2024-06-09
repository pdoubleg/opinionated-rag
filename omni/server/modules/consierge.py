import json
from pathlib import Path
from textwrap import fill
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated
from annotated_types import Gt, Lt
from pydantic import BaseModel, Field, ValidationError, model_validator
from openai.types.completion_usage import CompletionUsage
import instructor
import openai
import time

# Constants
DEFAULT_PROMPT_ROOT_DIR = "prompts"
DEFAULT_PROMPT_FILE = "prompt_concierge.txt"
DEFAULT_TOPIC_DIR = "cons"
DEFAULT_TOPIC = "Legal Services Department Inquiries"


def build_omni_prompt(
    input_value: str,
    topic: str = DEFAULT_TOPIC,
    topic_dir: str = DEFAULT_TOPIC_DIR,
) -> str:
    """
    Constructs a complete prompt.

    Args:
        input_value (str): The user's input question or query.
        topic str: The specific topic under which the prompt is categorized.
        topic_dir str: The directory associated with the topic.

    Returns:
        str: A fully constructed prompt ready for use in the autocomplete system.
    """
    try:
        prompt_path = Path(DEFAULT_PROMPT_ROOT_DIR) / DEFAULT_PROMPT_FILE
        previous_completions_path = (
            Path(DEFAULT_PROMPT_ROOT_DIR)
            / "knowledge_bases"
            / topic_dir
            / "previous_completions.json"
        )
        domain_knowledge_path = (
            Path(DEFAULT_PROMPT_ROOT_DIR)
            / "knowledge_bases"
            / topic_dir
            / "domain_knowledge.txt"
        )

        with prompt_path.open("r") as file:
            prompt = file.read()

        with previous_completions_path.open("r") as file:
            previous_completions = file.read()

        with domain_knowledge_path.open("r") as file:
            domain_knowledge = file.read()

        prompt = prompt.replace("{{topic}}", topic)
        prompt = prompt.replace("{{previous_completions}}", previous_completions)
        prompt = prompt.replace("{{domain_knowledge}}", domain_knowledge)
        prompt = prompt.replace("{{input_value}}", input_value)

        return prompt

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {e.filename}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while building the prompt: {e}")


class Prediction(BaseModel):
    """Predicted optimal department for a new user query."""

    chain_of_thought: str = Field(
        description="Reasoning behind the prediction based on PREVIOUS_PREDICTIONS and DOMAIN_KNOWLEDGE.",
        exclude=True,
    )
    predicted_department: str = Field(
        ...,
        description="The predicted department.",
    )
    confidence: Annotated[int, Gt(0), Lt(11)] = Field(
        ...,
        description="An integer score from 1-10 indicating prediction confidence.",
    )

    def __str__(self):
        wrapped_thought = fill(self.chain_of_thought, width=100)
        thought = f"Thought: {wrapped_thought}\n\n"
        thought += f"Predicted Department: {self.predicted_department}\n\n"
        thought += f"Score: {self.confidence}\n"
        return thought


class MultiPredict(BaseModel):
    """
    Class containing multiple (THREE) predictions.

    Args:
        predictions (List[Prediction]): The list of predicted departments.
    """

    predictions: List[Prediction] = Field(
        ...,
        description="List of predictions.",
    )

    @model_validator(mode="after")
    def sort_predictions_by_confidence(cls, values):
        values.predictions = sorted(
            values.predictions, key=lambda p: p.confidence, reverse=True
        )
        return values

    @property
    def print_preds(self):
        output_string = ""
        for pred in self.predictions:
            output_string += str(pred)
            output_string += "\n------------------------------------------------\n\n"
        return output_string


class PredictionRequest(BaseModel):
    """Request model for making predictions."""

    user_query: str
    ground_truth: Optional[str] = None
    model_output: Optional[MultiPredict] = None
    token_usage: Optional[CompletionUsage] = None
    model_name: Optional[str] = None
    run_time: Optional[float] = None

    @property
    def cost(self):
        return (
            self.token_usage.prompt_tokens * 5 / 1000000
            + self.token_usage.completion_tokens * 15 / 1000000
        )

    @property
    def prediction_rank(self) -> int:
        """
        Returns the rank of the ground truth department in the predictions list.

        Returns:
            int: The rank of the ground truth department, or 0 if not found.
        """
        if not self.ground_truth or not self.model_output:
            return 0

        for rank, prediction in enumerate(self.model_output.predictions, start=1):
            if prediction.predicted_department == self.ground_truth:
                return rank

        return 0
    
        
    @property
    def correct_top_one(self) -> bool:
        """Checks if the ground truth department is the top prediction.

        Returns:
            bool: True if the ground truth department is the top prediction, else False.
        """
        return self.prediction_rank == 1
    
    @property
    def correct_top_two(self) -> bool:
        """Checks if the ground truth department is within the top two predictions.

        Returns:
            bool: True if the ground truth department is within the top two predictions, else False.
        """
        return self.prediction_rank in [1, 2]
    


def get_predictions(
    input_data: PredictionRequest,
    topic: str = DEFAULT_TOPIC,
    topic_dir: str = DEFAULT_TOPIC_DIR,
) -> MultiPredict:
    start_time = time.time()
    input_value = input_data.user_query
    prompt = build_omni_prompt(
        input_value=input_value, topic=topic, topic_dir=topic_dir
    )
    client = instructor.from_openai(openai.OpenAI())

    response, completion = client.chat.completions.create_with_completion(
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model="gpt-4o",
        response_model=MultiPredict,
    )

    return PredictionRequest(
        user_query=input_data.user_query,
        ground_truth=input_data.ground_truth or None,
        model_output=response,
        token_usage=completion.usage,
        model_name=completion.model,
        run_time=time.time() - start_time,
    )


class PastPredictionEntry(BaseModel):
    input: str
    correct_department: str


def increment_previous_predictions(input: str, completion: str) -> List[Dict[str, Any]]:
    """
    Helper function to update a "knowledge base" JSON file of few-shot-examples.
    Checks if a new "completion" exists in a specified knowledge base and if not adds it.

    Args:
        input (str): The input text.
        completion (str): The prediction.

    Returns:
        List[Dict[str, Any]]: The updated list of previous completions.
    """
    # Define the path to the previous completions file
    previous_completions_file = f"{DEFAULT_PROMPT_ROOT_DIR}/knowledge_bases/{DEFAULT_TOPIC_DIR}/previous_completions.json"

    # Attempt to load the previous completions from the file
    try:
        with open(previous_completions_file, "r") as file:
            data = json.load(file)
            previous_completions = [PastPredictionEntry(**item) for item in data]
    except (FileNotFoundError, json.JSONDecodeError, ValidationError):
        previous_completions = []

    # Search for a matching case in the previous completions
    matching_case = None
    for item in previous_completions:
        if item.input.lower() == input.lower():
            matching_case = item
            break

    # If no matching case is found, create a new entry
    if not matching_case:
        new_entry = PastPredictionEntry(input=input, correct_department=completion)
        previous_completions.append(new_entry)

    # Write the updated list of previous completions back to the file
    with open(previous_completions_file, "w") as file:
        json.dump(
            [entry.model_dump() for entry in previous_completions], file, indent=4
        )

    return previous_completions
