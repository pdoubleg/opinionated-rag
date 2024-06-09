import os
import json
from pathlib import Path
import warnings
from typing import Any, Dict, List
import instructor
from pydantic import BaseModel, Field, ValidationError
import tiktoken
import openai
from dotenv import load_dotenv


DEFAULT_PROMPT_ROOT_DIR = "prompts"
DEFAULT_PROMPT_FILE = "prompt_concierge.txt"
DEFAULT_TOPIC_DIR = "cons"
DEFAULT_TOPIC = "Legal Services Department Inquiries"


class AutoCompletionEntry(BaseModel):
    input: str
    completions: List[str]
    correct_department: str | None = None
    hits: int = 1

    def __str__(self):
        return f"{self.input} {self.completions[0]}"


class AutoCompletions(BaseModel):
    """Auto-completions for a new user query."""

    input: str = Field(
        ...,
        description="The user provided INPUT_VALUE.",
    )
    completions: List[str] = Field(
        default_factory=list,
        description="A list of potential completions based on the GENERATION_RULES.",
    )
    correct_department: str = Field(
        ...,
        description="The predicted department based on ALL available information.",
    )


class AutoCompleteSystem:
    def __init__(
        self,
        prompt_root_dir: str = DEFAULT_PROMPT_ROOT_DIR,
        topic_dir: str = DEFAULT_TOPIC_DIR,
        topic: str = DEFAULT_TOPIC,
    ):
        """
        Initializes the AutoCompleteSystem with the given directories and topic.

        Args:
            prompt_root_dir (str): The root directory for prompts.
            topic_dir (str): The directory for the specific topic.
            topic (str): The topic for the autocomplete system.
        """
        self.prompt_root_dir = prompt_root_dir
        self.topic_dir = topic_dir
        self.topic = topic
        self.prompt_template = self._load_prompt_template()
        self.previous_completions = self._load_previous_completions()
        self.domain_knowledge = self._load_domain_knowledge()
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        warnings.filterwarnings("ignore")

    def _load_prompt_template(self) -> str:
        """
        Loads the prompt template from a file.

        Returns:
            str: The content of the prompt template.
        """
        with open(f"{self.prompt_root_dir}/prompt.txt", "r") as file:
            return file.read()

    def _load_previous_completions(self) -> str:
        """
        Loads the previous completions from a JSON file.

        Returns:
            str: The content of the previous completions file.
        """
        with open(
            f"{self.prompt_root_dir}/knowledge_bases/{self.topic_dir}/previous_completions.json",
            "r",
        ) as file:
            return file.read()

    def _load_domain_knowledge(self) -> str:
        """
        Loads the domain knowledge from a file.

        Returns:
            str: The content of the domain knowledge file.
        """
        with open(
            f"{self.prompt_root_dir}/knowledge_bases/{self.topic_dir}/domain_knowledge.txt",
            "r",
        ) as file:
            return file.read()

    def build_prompt(self, input_value: str) -> str:
        """
        Builds the prompt by replacing placeholders with actual values.

        Args:
            input_value (str): The input value to be included in the prompt.

        Returns:
            str: The constructed prompt.
        """
        prompt = self.prompt_template.replace("{{topic}}", self.topic)
        prompt = prompt.replace("{{previous_completions}}", self.previous_completions)
        prompt = prompt.replace("{{domain_knowledge}}", self.domain_knowledge)
        prompt = prompt.replace("{{input_value}}", input_value)
        return prompt

    @staticmethod
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

    def get_autocompletions(self, input_data: str) -> AutoCompletions:
        """
        Retrieves auto-completions for a given input using a language model.

        Args:
            input_data (str): The user's input data for which completions are needed.

        Returns:
            AutoCompletion: An object containing the input, completions, and predicted department.
        """
        prompt = self.build_omni_prompt(
            input_data, topic=self.topic, topic_dir=self.topic_dir
        )
        client = instructor.from_openai(openai.OpenAI())

        return client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="gpt-4o",
            response_model=AutoCompletions,
        )

    def increment_previous_completions(
        self, input: str, completion: str
    ) -> List[Dict[str, Any]]:
        """
        Updates the previous completions by adding a new completion if no matching case is found.

        Args:
            input (str): The input text.
            completion (str): The completion text.

        Returns:
            List[Dict[str, Any]]: The updated list of previous completions.
        """
        # Define the path to the previous completions file
        previous_completions_file = f"{self.prompt_root_dir}/knowledge_bases/{self.topic_dir}/previous_completions.json"

        # Attempt to load the previous completions from the file
        try:
            with open(previous_completions_file, "r") as file:
                data = json.load(file)
                previous_completions = [AutoCompletionEntry(**item) for item in data]
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
            new_entry = AutoCompletionEntry(input=input, completions=[completion])
            previous_completions.append(new_entry)

        # Write the updated list of previous completions back to the file
        with open(previous_completions_file, "w") as file:
            json.dump(
                [entry.model_dump() for entry in previous_completions], file, indent=4
            )

        return previous_completions
