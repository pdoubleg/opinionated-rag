import os
from typing import Any, Dict
from markdown import markdown
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
import openai
from html2text import html2text


class NotebookDocGenerator:
    def __init__(self, api_key: str):
        """
        Initialize the NotebookDocGenerator.

        Args:
            api_key (str): The OpenAI API key.
        """
        openai.api_key = api_key

    def generate_documentation(
        self, file_path: str, output_path: str, instructions: str = None
    ) -> None:
        """
        Generate documentation from a Jupyter Notebook or Python module and write it to a file.

        Args:
            file_path (str): The path to the Jupyter Notebook (.ipynb) or Python module (.py) file.
            output_path (str): The path to the output file (markdown or text).
            instructions (str, optional): Additional instructions for generating the documentation.
                Defaults to a predefined set of instructions based on the file type.
        """
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == ".ipynb":
            nb_obj = self._read_notebook(file_path)
            nb_html = self._retrieve_html(nb_obj)
            markdown_from_html_text = html2text(nb_html)
            text = self._markdown_to_text(markdown_from_html_text)
            if instructions is None:
                instructions = (
                    "You are a Python project documentation AI specializing in drafting clear wiki-style documentation "
                    "in markdown that is easy to read for technical and non-technical readers. Users will provide "
                    "contents of a Jupyter Notebook that has been converted to text. Your task is to generate a "
                    "structured summary of the code and its output. Try to infer intent so that we explain the benefits "
                    "and takeaways. Include example Python code blocks for the core functionality only, and add "
                    "docstrings and examples using toy/fake data where needed to enhance readability."
                )
            prompt = f"Please review the following Jupyter Notebook and generate a markdown summary suitable to use as a README file. Notebook: {text}"
        
        elif file_extension.lower() == ".py":
            text = self.read_python_module(file_path)
            if instructions is None:
                instructions = (
                    "You are a Python project documentation AI specializing in drafting clear wiki-style documentation "
                    "in markdown that is easy to read for technical and non-technical readers. Users will provide "
                    "contents of a Python module. Your task is to generate a structured summary of the code. "
                    "Try to infer intent so that we explain the benefits and takeaways. Include example Python code blocks "
                    "for the core functionality only, and add docstrings and examples using toy/fake data where needed to enhance readability."
                )
            prompt = f"Please review the following Python module and generate a markdown summary suitable to use as a README file. Module: {text}"
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        nb_doc = self._prompt(
            instructions=instructions,
            prompt=prompt,
        )

        _, file_extension = os.path.splitext(output_path)
        if file_extension.lower() == ".md":
            self.write_to_markdown(nb_doc, output_path)
        else:
            self.write_to_text(nb_doc, output_path)

    def write_to_markdown(self, documentation: str, output_path: str) -> None:
        """
        Write the generated documentation to a markdown file.

        Args:
            documentation (str): The generated documentation in markdown format.
            output_path (str): The path to the output markdown file.
        """
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(documentation)

    def write_to_text(self, documentation: str, output_path: str) -> None:
        """
        Write the generated documentation to a text file.

        Args:
            documentation (str): The generated documentation in markdown format.
            output_path (str): The path to the output text file.
        """
        markdown_text = self._markdown_to_text(documentation)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(markdown_text)

    @staticmethod
    def _read_notebook(notebook_path: str) -> nbformat.NotebookNode:
        """Read a Jupyter Notebook file and return the notebook object."""
        with open(notebook_path, encoding="utf-8") as file:
            return nbformat.read(file, as_version=4)

    @staticmethod
    def _retrieve_html(notebook: nbformat.NotebookNode) -> str:
        """Convert a notebook object to HTML."""
        html_exporter = HTMLExporter()
        html_output, _ = html_exporter.from_notebook_node(notebook)
        return html_output

    @staticmethod
    def _markdown_to_text(markdown_string: str) -> str:
        """Convert a markdown string to plain text."""
        html = markdown(markdown_string)
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        return text

    @staticmethod
    def _safe_get(data: Dict[str, Any], dot_chained_keys: str) -> Any:
        """
        Safely retrieve a value from a nested dictionary using dot-chained keys.

        Args:
            data (Dict[str, Any]): The nested dictionary.
            dot_chained_keys (str): The dot-chained keys to access the value.

        Returns:
            Any: The retrieved value or None if the keys are not found.
        """
        keys = dot_chained_keys.split(".")
        for key in keys:
            try:
                if isinstance(data, list):
                    data = data[int(key)]
                else:
                    data = data[key]
            except (KeyError, TypeError, IndexError):
                return None
        return data

    @staticmethod
    def _response_parser(response: Dict[str, Any]) -> str:
        """Parse the response from the OpenAI API."""
        return NotebookDocGenerator._safe_get(response, "choices.0.message.content")

    def _prompt(
        self, prompt: str, instructions: str, model: str = "gpt-4-1106-preview"
    ) -> str:
        """Generate a response from a prompt using the OpenAI API."""
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return self._response_parser(response.model_dump())

    @staticmethod
    def read_python_module(file_path: str) -> str:
        """
        Read a Python module file and return its contents as a string.

        Args:
            file_path (str): The path to the Python module file.

        Returns:
            str: The contents of the Python module file as a string.
        """
        with open(file_path, "r") as file:
            module_contents = file.read()
        return module_contents
