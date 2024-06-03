import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

import openai

# Initialize the OpenAI client with the API key from environment variables
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def prompt_json(prompt: str) -> str:
    """Generates a JSON response from the OpenAI API based on the given prompt.

    This function sends a prompt to the OpenAI API and retrieves a response in JSON format.

    Args:
        prompt (str): The input prompt to be sent to the OpenAI API.

    Returns:
        str: The content of the response message from the OpenAI API.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
        model="gpt-4o",
        response_format={
            "type": "json_object",
        },
    )
    return chat_completion.choices[0].message.content


def prompt_llm(
    prompt: str,
    model: str = "gpt-4o",
    instructions: str = "You are a helpful assistant.",
) -> str:
    """
    Generate a response from a prompt using the OpenAI API without specifying a response format.
    """
    response = client.chat.completions.create(
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
    return response.choices[0].message.content


