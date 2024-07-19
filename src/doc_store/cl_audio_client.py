import os
import re
from typing import List, Optional, Union
from dotenv import load_dotenv
import openai
from pydantic import BaseModel, Field, HttpUrl, conint, field_validator, model_validator
from datetime import datetime
from pydub import AudioSegment
from pydub.silence import split_on_silence
import requests

from src.doc_store.utils import download_mp3_tempfile
from src.schema.docket import Docket
from src.doc_store.courtlistener import COURTLISTENER_WEB_URL
from src.schema.audio import Source, SOURCES

class AudioResource(BaseModel):
    resource_uri: HttpUrl
    id: int
    frontend_url: HttpUrl
    absolute_url: str
    panel: List[str] = []
    docket: HttpUrl
    date_created: datetime
    date_modified: datetime
    source: Source
    case_name_short: str
    case_name: str
    case_name_full: Optional[str] = ""
    judges: Optional[str] = ""
    sha1: str
    download_url: HttpUrl
    local_path_mp3: HttpUrl
    local_path_original_file: str
    filepath_ia: HttpUrl
    duration: int
    processing_complete: bool
    date_blocked: Optional[datetime] = None
    blocked: bool
    stt_status: conint(ge=0)
    stt_google_response: Optional[str] = ""
    
    @model_validator(mode="before")
    def generate_frontend_url(cls, values):
        absolute_url = values.get('absolute_url')
        if absolute_url:
            base_url = COURTLISTENER_WEB_URL
            values['frontend_url'] = f"{base_url}{absolute_url}"
        return values
    
    @model_validator(mode="before")
    def generate_mp3_url(cls, values):
        local_path_mp3 = values.get('local_path_mp3')
        if local_path_mp3:
            base_url = "https://www.courtlistener.com/"
            values['local_path_mp3'] = f"{base_url}{local_path_mp3}"
        return values
    
    @property
    def whisper_cost(self):
        audio_duration_in_minutes = self.duration / 60
        return round(audio_duration_in_minutes, 2) * 0.006
    
    def __str__(self) -> str:
        """
        Generates a string representation of the object, combining various attributes.

        Returns:
            str: A string representation of the object.
        """
        text_parts = []
        case_id = f"Case ID: {self.docket_num}"
        text_parts.append(case_id)
        source = self.get_source_description
        source_str = f"Source: [{source}]({self.download_url})"
        text_parts.append(source_str)
        name_short = str(self.case_name)
        text_parts.append(name_short)
        date_argued_str = self.serialize_date(self.date_argued)
        date = f"Date Argued: {date_argued_str[:10]}"
        text_parts.append(date)
        court = f"Court: {str(self.court)}"
        text_parts.append(court)
        cleaned_snippet = re.sub(r'\n+', '\n', self.snippet).strip()
        text_parts.append(cleaned_snippet)
        return "\n\n".join(text_parts)

   
class OralArgumentSearchResult(BaseModel):
    absolute_url: str
    frontend_url: HttpUrl
    case_name: str = Field(
        ...,
        alias="caseName",
    )
    court: str | None = None
    court_citation_string: str | None = None
    court_exact: str | None = None
    court_id: str | None = None
    date_argued: datetime | None= Field(
        None,
        alias="dateArgued"
    )
    docket_num: str | None = Field(
        None,
        alias="docketNumber"
    )
    docket_id: int | None = None
    download_url: HttpUrl 
    duration: int | None = None
    file_size_mp3: int | None = None
    id: int | None = None
    judge: Optional[str] = ''
    local_path: str | None = None
    pacer_case_id: Optional[int] = None
    panel_ids: Optional[int | List[int]] = None
    sha1: Optional[str] = None
    snippet: str | None = None
    source: Source | None = None
    timestamp: datetime | None = None
    
    @property
    def get_source_description(self) -> Optional[str]:
        """
        Returns the description of the current source enum value.

        Returns:
            Optional[str]: The description of the selected source, or None if no source is selected.
        """
        if self.source is None:
            return None
        # Find the description in SOURCES.NAMES using the current source value
        for code, description in SOURCES.NAMES:
            if code == self.source.value:
                return description
        return None
    
    @model_validator(mode="before")
    def generate_frontend_url(cls, values):
        absolute_url = values.get('absolute_url')
        if absolute_url:
            base_url = COURTLISTENER_WEB_URL
            values['frontend_url'] = f"{base_url}{absolute_url}"
        return values
    
    @field_validator("date_argued", mode="before")
    def serialize_date(cls, v: Union[datetime, str]) -> str:
        """
        Serializes the date_argued field to a string format.

        Args:
            date_argued (Union[date, str]): The decision date to be serialized.

        Returns:
            str: The serialized decision date as a string.
        """
        if isinstance(v, datetime):
            return v.isoformat()
        return v
    
    @property
    def whisper_cost(self):
        audio_duration_in_minutes = self.duration / 60
        return round(audio_duration_in_minutes, 2) * 0.006
    
    def __str__(self) -> str:
        """
        Generates a string representation of the object, combining various attributes.

        Returns:
            str: A string representation of the object.
        """
        text_parts = []
        case_id = f"Case ID: {self.docket_num}"
        text_parts.append(case_id)
        source = self.get_source_description
        source_str = f"Source: [{source}]({self.frontend_url})"
        text_parts.append(source_str)
        name_short = str(self.case_name)
        text_parts.append(name_short)
        if self.date_argued is not None:
            date_argued_str = self.serialize_date(self.date_argued)
            date = f"Date Argued: {date_argued_str[:10]}"
        else:
            date = "Date Argued: Unknown"
        text_parts.append(date)
        court = f"Court: {str(self.court)}"
        text_parts.append(court)
        # Clean snippet by replacing all sequences of multiple line breaks with a single line break
        cleaned_snippet = re.sub(r'\n+', '\n', self.snippet).strip()
        text_parts.append(cleaned_snippet)
        return "\n\n".join(text_parts)


class CourtListenerAudio:
    """
    A class to interact with the CourtListener Audio API endpoint.
    """
    AUDIO_URL = 'https://www.courtlistener.com/api/rest/v3/audio/'
    SEARCH_URL = 'https://www.courtlistener.com/api/rest/v3/search/'
    
    def __init__(self):
        """
        Initializes the CourtListenerAudio with the necessary API token.

        """
        self.api_token = os.getenv("COURTLISTENER_API_KEY")
        self.headers = {'Authorization': f'Token {self.api_token}'}
        
    def get_docket(self, url: HttpUrl):
        response = requests.get(url, headers=self.headers)
        response_dict = response.json()
        return Docket(**response_dict)
    
    def get_docket_id(self, docket_id: int):
        url = f"https://www.courtlistener.com/api/rest/v3/dockets/{docket_id}/"
        response = requests.get(url, headers=self.headers)
        response_dict = response.json()
        return Docket(**response_dict)
    
    def get_audio(self, url: HttpUrl):
        response = requests.get(url, headers=self.headers)
        response_dict = response.json()
        
        return AudioResource(**response_dict)

    def query_audio(self, filters: dict = None, ordering: str = "dateArgued") -> List[AudioResource]:
        """
        Queries the CourtListener Audio endpoint with optional filters and ordering.

        Args:
            filters (dict, optional): A dictionary of filters to apply to the query.
            ordering (str, optional): A string specifying the ordering of the results.

        Returns:
            AudioResource: A list of AudioResource objects.
        """
        params = filters if filters else {}
        if ordering:
            params['ordering'] = f"{ordering} desc"

        response = requests.get(self.AUDIO_URL, headers=self.headers, params=params)
        response.raise_for_status()
        search_res = response.json()["results"]
        return [AudioResource(**s) for s in search_res]
    
    
    def search_oral_arguments(self, query: str, filters: dict = None, ordering: str = "dateArgued") -> dict:
        """
        Searches for oral arguments using the CourtListener search endpoint with the type "oa".

        Args:
            query (str): The search query string.
            filters (dict, optional): Additional filters to apply to the search.
            ordering (str, optional): A string specifying the ordering of the results.

        Returns:
            dict: The JSON response from the CourtListener API.
        """
        params = {'q': query, 'type': 'oa'}
        if filters:
            params.update(filters)
        params['order_by'] = f"{ordering} desc"

        response = requests.get(self.SEARCH_URL, headers=self.headers, params=params)
        response.raise_for_status()
        search_res = response.json()["results"]
        return [OralArgumentSearchResult(**s) for s in search_res]
    
    
    def process_and_transcribe(self, filepath: str, min_file_size: int = 1 * 1024 * 1024, max_file_size: int = 15 * 1024 * 1024):
        """
        Processes an MP3 file by splitting it based on silence, then dynamically combining or splitting chunks
        to ensure each is within a specified size range before transcribing. Saves the final transcription to a text file
        and returns the transcription objects.

        Args:
            filepath (str): Path to the MP3 file to be processed.
            min_file_size (int, optional): Minimum file size for each chunk in bytes. Defaults to 1MB.
            max_file_size (int, optional): Maximum file size for each chunk in bytes. Defaults to 15MB.

        Returns:
            List[dict]: A list of transcription objects returned by the transcription API.
        """
        sound = AudioSegment.from_file(filepath)
        chunks = split_on_silence(
            sound,
            min_silence_len=500,
            silence_thresh=sound.dBFS - 16,
            keep_silence=250,
        )

        transcriptions = []  # To store transcription texts
        combined_chunk = AudioSegment.empty()
        for _, chunk in enumerate(chunks):
            combined_chunk += chunk
            combined_chunk_length = len(combined_chunk)

            if combined_chunk_length >= min_file_size:
                if combined_chunk_length > max_file_size:
                    # Split and transcribe large chunks
                    half = combined_chunk_length // 2
                    transcriptions.append(self.transcribe_chunk(combined_chunk[:half]))
                    transcriptions.append(self.transcribe_chunk(combined_chunk[half:]))
                    combined_chunk = AudioSegment.empty()
                else:
                    # Transcribe suitable chunks
                    transcriptions.append(self.transcribe_chunk(combined_chunk))
                    combined_chunk = AudioSegment.empty()

        if len(combined_chunk) > 0:
            transcriptions.append(self.transcribe_chunk(combined_chunk))

        # Save transcriptions to a text file
        with open("final_transcription.txt", "w") as f:
            for transcription in transcriptions:
                f.write(transcription + "\n\n")

        return transcriptions

    @staticmethod
    def transcribe_chunk(chunk: AudioSegment) -> dict:
        """
        Transcribes a given audio chunk using the OpenAI Whisper model and returns the transcription object.

        Args:
            chunk (AudioSegment): The audio chunk to be transcribed.

        Returns:
            dict: The transcription object returned by the transcription API.
        """
        load_dotenv()
        client = openai.OpenAI()
        temp_file = "temp_chunk.mp3"
        chunk.export(temp_file, format="mp3")

        audio_file = open(temp_file, "rb")
        # Transcribe the audio file
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="json",
            )
        audio_file.close()
        os.remove(temp_file)
        return transcription.text
    
    @staticmethod
    def download_mp3_tempfile(url: str) -> str:
        """
        Downloads an MP3 file from the given URL and saves it to a temporary file path.

        Args:
            url (str): The URL of the MP3 file.

        Returns:
            str: The file path of the downloaded MP3 file.
        """
        import requests
        from tempfile import NamedTemporaryFile

        try:
            # Send a GET request to the URL
            response = requests.get(url, stream=True)

            # Raise an exception if the request was unsuccessful
            response.raise_for_status()

            # Create a named temporary file in binary write mode, which will be automatically deleted
            with NamedTemporaryFile(delete=False, suffix='.mp3', mode='wb') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    # Write the content chunk by chunk to the file
                    temp_file.write(chunk)
                file_size = os.path.getsize(temp_file.name) / (1024 * 1024)
                print(f"File downloaded and saved as {temp_file.name}\nFile Size: {file_size:.2f} MB")
                return temp_file.name
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return ""
      
    @staticmethod  
    def save_texts_to_files(search_results: List[OralArgumentSearchResult], output_dir: str):
        """
        Saves the string representation of OralArgumentSearchResult to a text file.

        Args:
            search_results (List[OralArgumentSearchResult]): List of search results.
            output_dir (str): Directory to save the text files.

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for result in search_results:
            # Extract the filename from the local_path attribute
            filename = os.path.basename(result.local_path)
            # Change the extension to .txt
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            # Create the full output path
            output_path = os.path.join(output_dir, txt_filename)
            
            # Write the snippet to the file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(result))
            print(f"Saved snippet to {output_path}")