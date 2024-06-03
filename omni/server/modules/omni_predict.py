import os
import json
import warnings
from typing import Any, Dict, List
import instructor
from pydantic import BaseModel, Field, ValidationError
import tiktoken
import openai
from dotenv import load_dotenv


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
    def __init__(self, prompt_root_dir: str, topic_dir: str, topic: str):
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
        warnings.filterwarnings('ignore')

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
        with open(f"{self.prompt_root_dir}/knowledge_bases/{self.topic_dir}/previous_completions.json", "r") as file:
            return file.read()

    def _load_domain_knowledge(self) -> str:
        """
        Loads the domain knowledge from a file.

        Returns:
            str: The content of the domain knowledge file.
        """
        with open(f"{self.prompt_root_dir}/knowledge_bases/{self.topic_dir}/domain_knowledge.txt", "r") as file:
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
    
    def truncate_json_by_tokens(self, json_string: str, threshold: int, model: str = "gpt-3.5-turbo") -> List[Dict[str, Any]]:
        """
        Truncates a JSON string into a list of dictionaries based on a token threshold using a specified model.

        Args:
            json_string (str): The JSON string to be truncated.
            threshold (int): The maximum number of tokens allowed in the truncated output.
            model (str): The model used to count tokens, default is "gpt-3.5-turbo".

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the truncated JSON data.
        """
        data = json.loads(json_string)
        truncated_data = []
        current_tokens = 0

        for item in data:
            item_string = json.dumps(item, indent=4)
            item_tokens = tiktoken.count_tokens(item_string, model)
            
            if current_tokens + item_tokens > threshold:
                break
            
            truncated_data.append(item)
            current_tokens += item_tokens

        return truncated_data

    def build_prompt_with_token_limit(self, input_value: str, token_threshold: int) -> str:
        """
        Builds the prompt with a token limit by truncating the previous completions.

        Args:
            input_value (str): The input value to be included in the prompt.
            token_threshold (int): The maximum number of tokens allowed in the prompt.

        Returns:
            str: The constructed prompt with token limit.
        """
        truncated_completions = self.truncate_json_by_tokens(self.previous_completions, token_threshold)
        truncated_completions_str = json.dumps(truncated_completions, indent=4)
        
        prompt = self.prompt_template.replace("{{topic}}", self.topic)
        prompt = prompt.replace("{{previous_completions}}", truncated_completions_str)
        prompt = prompt.replace("{{domain_knowledge}}", self.domain_knowledge)
        prompt = prompt.replace("{{input_value}}", input_value)
        return prompt
    
    def get_autocompletions(self, input_data: str) -> AutoCompletions:
        """
        Retrieves auto-completions for a given input using a language model.

        Args:
            input_data (str): The user's input data for which completions are needed.

        Returns:
            AutoCompletion: An object containing the input, completions, and predicted department.
        """
        prompt = self.build_prompt(
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
                model="gpt-4-turbo",
                response_model=AutoCompletions,
            )
    
    def increment_or_create_previous_completions(self, input: str, completion: str) -> List[Dict[str, Any]]:
        """
        Increments the hit count for an existing completion or creates a new one.

        Args:
            input (str): The input text.
            completion (str): The completion text.

        Returns:
            List[Dict[str, Any]]: The updated list of previous completions.
        """
        previous_completions_file = f"{self.prompt_root_dir}/knowledge_bases/{self.topic_dir}/previous_completions.json"
        try:
            with open(previous_completions_file, "r") as file:
                data = json.load(file)
                previous_completions = [AutoCompletionEntry(**item) for item in data]
        except (FileNotFoundError, json.JSONDecodeError, ValidationError):
            previous_completions = []

        matching_case = None
        for item in previous_completions:
            if item.input.lower() == input.lower():
                matching_case = item
                break

        if matching_case:
            matching_case.hits += 1
            matching_case.completions.append(completion)
        else:
            new_entry = AutoCompletionEntry(input=input, completions=[completion])
            previous_completions.append(new_entry)

        with open(previous_completions_file, "w") as file:
            json.dump([entry.model_dump() for entry in previous_completions], file, indent=4)

        return previous_completions
