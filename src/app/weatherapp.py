import base64
from datetime import datetime
import re
from typing import Any, Dict
from urllib.parse import urlparse
import openai
import requests
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from openai.types.images_response import ImagesResponse

load_dotenv()

def safe_get(data, dot_chained_keys):
    """
    {'a': {'b': [{'c': 1}]}}
    safe_get(data, 'a.b.0.c') -> 1
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


def response_parser(response: Dict[str, Any]):
    return safe_get(response, "choices.0.message.content")


def is_url(image_path: str) -> bool:
    """
    Check if the given string is a valid URL.

    Args:
        image_path (str): The string to check.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    try:
        result = urlparse(image_path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image file to a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ------------------ content generators ------------------


def prompt(
    prompt: str,
    model: str = "gpt-4-turbo",
    instructions: str = "You are a helpful assistant.",
) -> str:

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instructions,  # Added instructions as a system message
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response_parser(response.model_dump())

def prompt_image_gen(
    prompt: str,
    openai_key: str = os.getenv("OPENAI_API_KEY"),
    model: str = "dall-e-3", # or 'dall-e-2'
    size_category: str = "square",
    style: str = "vivid", # or 'natural'
    quality: str = "standard",
) -> Dict[str, str]:
    """
    Generate an image from a prompt using the OpenAI API, dynamically save it to a file path based on the prompt and current datetime,
    and return a dictionary with the file path and URL for display purposes.

    Args:
        prompt (str): The prompt to generate the image from.
        openai_key (str): The OpenAI API key.
        model (str, optional): The model to use for image generation. Defaults to "dall-e-3".
        size (str, optional): The size of the generated image. Defaults to "512x512".
        quality (str, optional): The quality of the generated image. Defaults to "standard".

    Returns:
        Dict[str, str]: A dictionary containing the file path and URL of the generated image.
    """   
    d2_size_mapping = {
        "small": "256x256",
        "medium": "512x512",
        "large": "1024x1024",
    }
    d3_size_mapping = {
        "square": "1024x1024",
        "wide": "1792x1024",
        "tall": "1024x1792"
    }
    if model == "dall-e-2":
            size_mapping = d2_size_mapping
    elif model == "dall-e-3":
        size_mapping = d3_size_mapping
    else:
        raise ValueError("Unsupported model. Choose either 'dall-e-2' or 'dall-e-3'.")
    
    # Set the OpenAI API key
    client = openai.OpenAI(
    )
    
    # Get the size from the mapping
    size = size_mapping.get(size_category, "512x512")

    # Generate the image
    response: ImagesResponse = client.images.generate(
        prompt=prompt,
        model=model,
        n=1,
        quality=quality,
        size=size,
        style=style,
    )

    # Extract the image URL from the response
    image_data = response.data[0]
    image_url = image_data.url

    # Create a sanitized version of the prompt for the file name
    sanitized_prompt = re.sub(r'[^A-Za-z0-9]+', '', prompt)[:8]
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data/dalle_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = f"{output_dir}/{sanitized_prompt}_{datetime_str}.jpeg"

    # Download and save the image
    with requests.get(image_url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return {"file_path": file_path, "image_url": image_url}

def convert_timestamp_to_datetime(timestamp: str) -> str:
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d")

timestamp = datetime.timestamp(datetime.now())

timestamp_string = convert_timestamp_to_datetime(timestamp)

def image_gen_prompt(
    prompt: str,
    model: str = "gpt-4-turbo",
):
    return openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a master of the visual arts, adept at vividly describing scenic locations while emphasizing the effects of the weather. You will receive a CITY and a WEATHER REPORT. Use then to make am awesome Dalle-3 prompt that reflects the actual weather, but make it extreme and try to incorporate elements of the city if they are known. We really want the user to **feel** the weather in their home town. Make the prompt sound like a story so do not say 'generate an image'. Make sure to keep it under 4000 characters",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=True
    )



def initialize_weather_agent():
    """ Initialize the weather agent with specified LLM and tools. """
    llm = OpenAI()
    tools = load_tools(['openweathermap-api'], llm)
    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

def run_weather_app(weather_agent):
    """ Run the Streamlit app for weather forecasting. """
    st.title('Weather Forecast')
    city = st.text_input('Enter a city name:', '')

    if st.button('Get Weather'):
        if city:
            query = f"What's the weather in {city} in the next 3 hours?"
            report = weather_agent.run(query)
            st.write(report)
            with st.status(label=" ", expanded=True) as status:
                full_response = ""
                message_placeholder = st.empty()
                dalle3_prompt_stream = image_gen_prompt(f"Please help me write an awesome prompt for Dalle-3 that depicts the weather in {city}. Here is the weather report: {report}")
                # st.write_stream(dalle3_prompt_stream)
                for chunk in dalle3_prompt_stream:
                    if chunk.choices[0].delta.content is not None:
                        response = chunk.choices[0].delta.content
                        full_response += response
                        message_placeholder.markdown(full_response + "▌")
                status.update(expanded=False)
            message_placeholder.markdown(full_response)

            gen_image = prompt_image_gen(prompt=full_response)
            st.image(gen_image['image_url'], caption=f"{city}, {timestamp_string}")
            st.link_button("image link", gen_image['image_url'], use_container_width =True)
        else:
            st.error("Please enter a city name to check the weather.")

if __name__ == '__main__':
    weather_agent = initialize_weather_agent()
    run_weather_app(weather_agent)