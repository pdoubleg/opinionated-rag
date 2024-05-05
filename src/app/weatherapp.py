
import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_extras.add_vertical_space import add_vertical_space as avs
import os
import base64
from datetime import datetime
import hashlib
from pathlib import Path
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse
import numpy as np
import openai
from pydantic import BaseModel, ConfigDict, HttpUrl, model_validator
import requests
from PIL import Image
from dotenv import load_dotenv
from openai.types.images_response import ImagesResponse

load_dotenv()

# ------------------ utility functions ------------------

def celsius_to_fahrenheit(temp_celsius: float) -> float:
    return (temp_celsius * 9/5) + 32

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
    
def thumbnail(image, scale=3):
    return image.resize(np.array(image.size)//scale)

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()    
    
# ------------------ pydantic models ------------------


class WeatherData(BaseModel):
    temp: float
    temp_max: float
    temp_min: float
    feels_like: float
    description: str
    icon: str
    wind_speed: float
    wind_direction: int
    humidity: int
    rain: str
    cloud_cover: int
    sunset_local: str
    city_name: str
    date_stamp: str

    def __str__(self):
        return (
            f"{self.date_stamp}\n"
            f"In {self.city_name}, the weather is currently:\n"
            f"Status: {self.description.title()}\n"
            f"Wind speed: {self.wind_speed} m/s, direction: {self.wind_direction}°\n"
            f"Humidity: {self.humidity}%\n"
            f"Temperature: \n"
            f"  - Current: {self.temp}°F\n"
            f"  - High: {self.temp_max}°F\n"
            f"  - Low: {self.temp_min}°F\n"
            f"  - Feels like: {self.feels_like}°F\n"
            f"Rain: {self.rain if self.rain else 'No rain'}\n"
            f"Cloud cover: {self.cloud_cover}%"
        )
    
    @property
    def to_markdown(self):
        return (
            f"{self.date_stamp}\n\n"
            f"The weather in **{self.city_name}** is currently:\n\n"
            f"Status: {self.description.title()}\n\n"
            f"Wind speed: {self.wind_speed} m/s, direction: {self.wind_direction}°\n\n"
            f"Humidity: {self.humidity}%\n\n"
            f"Temperature: \n\n"
            f"  - Current: {self.temp}°F\n"
            f"  - High: {self.temp_max}°F\n"
            f"  - Low: {self.temp_min}°F\n"
            f"  - Feels like: {self.feels_like}°F\n\n"
            f"Rain: {self.rain if self.rain else 'No rain'}\n\n"
            f"Cloud cover: {self.cloud_cover}%"
        )

class TheWeather(BaseModel):
    model_config = ConfigDict(
        extra='allow'
    )
    image_url: HttpUrl
    query: Optional[str] = None
    timestamp: Optional[datetime | str] = None
    hash_id: Optional[str] = None
    
    @model_validator(mode="before")
    def generate_hash(cls, values):
        unique_string = str(values["ebay_id"])+str(values["image_url"])
        values['hash_id'] = hash_text(unique_string)
        return values
    
    @model_validator(mode="before")
    def generate_tiemstamp(cls, values):
        timestamp = datetime.now()
        values['timestamp'] = timestamp.isoformat(timespec='minutes')
        return values
    
    @property
    def filename(self):
        filename = re.sub(
            r'[\\/*?:"<>|]', "", str(self.item.title)
        )  # Remove invalid file name characters
        filename = re.sub(r"\s+", "_", filename)  # Replace spaces with underscores
        filename += ".jpg"  # Append file extension
        return filename

    @property
    def full_path(self):
        folder_path: str = "./data/theweatherapp"
        # Ensure the folder exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        # Combine folder path and filename
        file_path = Path(folder_path) / self.filename
        return file_path
    
    @property
    def image(self) -> Image.Image:
        if not Path(self.full_path).exists():
            self.download_image()
        return Image.open(self.full_path)

    def download_image(
        self,
        folder_path: str = "./data/theweatherapp",
    ) -> None:
        """
        Downloads an image from a given URL and saves it to a specified folder with a filename
        based on the cleaned title attribute.

        Args:
            folder_path (str): The path to the folder where the image will be saved.

        Returns:
            None
        """
        # Ensure the folder exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        # Combine folder path and filename
        file_path = Path(folder_path) / self.filename

        # Download and save the image
        response = requests.get(self.item.image_url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download image from {self.image_url}")


# ------------------ content generators ------------------

def fetch_weather(search_query: str, search_type: str = "city") -> WeatherData:
    API_key = os.getenv("OPENWEATHERMAP_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    
    if search_type == "city":
        final_url = f"{base_url}appid={API_key}&q={search_query}"
    elif search_type == "zip":
        final_url = f"{base_url}appid={API_key}&zip={search_query}"
    else:
        raise ValueError(f"Invalid search type: {search_type}. Must be either 'city' or 'zip'.")
    
    owm_response_json = requests.get(final_url).json()
    
    sunset_utc = datetime.fromtimestamp(owm_response_json["sys"]["sunset"])
    sunset_local = sunset_utc.strftime("%I:%M %p")
    
    temp_celsius = owm_response_json["main"]["temp"] - 273.15
    temp_max_celsius = owm_response_json["main"]["temp_max"] - 273.15
    temp_min_celsius = owm_response_json["main"]["temp_min"] - 273.15
    temp_feels_like_celsius = owm_response_json["main"]["feels_like"] - 273.15

    temp_fahrenheit = round(celsius_to_fahrenheit(temp_celsius), 2)
    temp_max_fahrenheit = round(celsius_to_fahrenheit(temp_max_celsius), 2)
    temp_min_fahrenheit = round(celsius_to_fahrenheit(temp_min_celsius), 2)
    temp_feels_like_fahrenheit = round(celsius_to_fahrenheit(temp_feels_like_celsius), 2)

    rain = owm_response_json.get("rain", "No rain")
    
    owm_dict = {
        "temp": temp_fahrenheit,
        "temp_max": temp_max_fahrenheit,
        "temp_min": temp_min_fahrenheit,
        "feels_like": temp_feels_like_fahrenheit,
        "description": owm_response_json["weather"][0]["description"],
        "icon": owm_response_json["weather"][0]["icon"],
        "wind_speed": owm_response_json["wind"]["speed"],
        "wind_direction": owm_response_json["wind"]["deg"],
        "humidity": owm_response_json["main"]["humidity"],
        "rain": rain,
        "cloud_cover": owm_response_json["clouds"]["all"],
        "sunset_local": sunset_local,
        "city_name": owm_response_json["name"],
        "date_stamp": datetime.utcnow().strftime("%A, %B %d, %Y")
    }
    
    return WeatherData(**owm_dict)


def prompt(
    prompt: str,
    model: str = "gpt-4-turbo",
    instructions: str = "You are a helpful assistant.",
):
    return openai.chat.completions.create(
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
        stream=True,
    )

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
                "content": "You are a master of the visual arts, adept at vividly describing locations while emphasizing the effects of the weather. You will receive a CITY and a WEATHER REPORT, along with important USER_NOTES. Use then to make am awesome Dalle-3 prompt that emphasizes the weather, make it extreme and try to incorporate elements of the city if they are known. We really want the user to **feel** the weather in their home town. Make the prompt sound like a short story so do not say 'generate an image'. Make sure to keep it under 4000 characters",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=True
    )



def run_weather_app():
    """ Run the Streamlit app for weather forecasting. """
    st.markdown('# Weather Forecast', unsafe_allow_html=True)
    # TODO: Add option to search by zip code
    city = ui.input(default_value=None, type='text', placeholder="Enter a city name:", key="city_input")
    notes = ui.input(default_value=None, type='text', placeholder="Add some notes:", key="notes_input")

    if ui.button('Get Weather', key="clk_btn", className="bg-cyan-950 text-white"):
        if city:
            # Get basic weather forecast
            weather_data = fetch_weather(city)
            report = str(weather_data.to_markdown)
            full_report = ""
            report_message_placeholder = st.empty()
            weather_report_stream = prompt(
                prompt=f"Please concisely summarize the following weather report. Include a concise narrative with highlights and end with a markdown table summary. Also please note the following important user feedback: **{notes}**\n\nHere is the report: {report}"
            )
            for chunk in weather_report_stream:
                if chunk.choices[0].delta.content is not None:
                    response = chunk.choices[0].delta.content
                    full_report += response
                    report_message_placeholder.markdown(full_report + "▌", unsafe_allow_html=True)
            # Capture the completed response tokens
            report_message_placeholder.markdown(full_report, unsafe_allow_html=True)
            avs(1)
            # Create prompt for image gen and stream in response
            with st.status(label="dreaming about the weather ...", expanded=False) as status:
                full_response = ""
                message_placeholder = st.empty()
                dalle3_prompt_stream = image_gen_prompt(f"Please help write an awesome prompt for Dalle-3 that depicts the weather in {city}. Also please note the following important user feedback: **{notes}**\n\nHere is the weather report, and please be mindful of the season we are in based on the report. Here is the report: {report}")
                for chunk in dalle3_prompt_stream:
                    if chunk.choices[0].delta.content is not None:
                        response = chunk.choices[0].delta.content
                        full_response += response
                        message_placeholder.markdown(full_response + "▌")
                status.update(label=' ', expanded=False)
            # Capture the completed response tokens
            message_placeholder.markdown(full_response)
            # Send image gen prompt to dalle-3
            gen_image = prompt_image_gen(prompt=full_response)
            # Render image using native url
            avs(1)
            st.image(gen_image['image_url'], caption=f"{city.title()}, {weather_data.date_stamp}")
            # Render button link to the image
            avs(1)
            st.link_button("image link", gen_image['image_url'], use_container_width=True)
        else:
            st.error("Please enter a city name to check the weather.")

if __name__ == '__main__':
    run_weather_app()