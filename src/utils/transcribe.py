from typing import List

from pydantic import BaseModel
from faster_whisper import WhisperModel

class TranscriptWord(BaseModel):
    """Represents a word in the transcript."""

    start: float
    end: float
    word: str
    
class TranscriptSegment(BaseModel):
    """Represents a segment in the transcript."""

    start: float
    end: float
    text: str
    words: List[TranscriptWord]


class Transcription(BaseModel):
    """Represents the entire transcription output."""

    entire_script: str
    segments: List[TranscriptSegment]
    words: List[TranscriptWord]
    
    

def create_transcription(
    video_path: str,
    output_path: str,
    duration_limit_in_seconds: int = None,
    create_json_file: bool = False,
) -> Transcription:
    """Transcribes a video using Faster Whisper and returns a pydantic Transcription object.

    Args:
        video_path (str): The path to the video file to transcribe.
        duration_limit_in_seconds (int, optional): The maximum duration in seconds to process. Defaults to None.
    """

    print(
        f"create_transcription(video_path={output_path}.txt, duration_limit_in_seconds={duration_limit_in_seconds}, create_json_file={output_path}.json)"
    )

    model_size = "large-v3"
    # model_size = "medium.en"
    model = WhisperModel(model_size, device="auto", compute_type="int8")

    raw_segments, info = model.transcribe(
        video_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        language="en",
    )

    entire_script = ""
    segments: List[TranscriptWord] = []
    words: List[TranscriptWord] = []
    processed_duration = 0

    for segment in raw_segments:

        text = segment.text
        start = segment.start
        end = segment.end

        print(f"Processing segment (At {start}s-{end}s): '{text}'")

        _segment = TranscriptSegment(start=start, end=end, text=text, words=[])

        for word in segment.words:
            start = word.start
            end = word.end
            word = word.word

            # print(f"Processing word: {word} at {start} to {end}")

            _word = TranscriptWord(start=start, end=end, word=word)
            words.append(_word)

            _segment.words.append(_word)

        segments.append(_segment)

        entire_script += text + " "

        processed_duration = end

        if duration_limit_in_seconds is not None and processed_duration > int(
            duration_limit_in_seconds
        ):
            print(
                f"Duration limit of {duration_limit_in_seconds} seconds reached. Stopping transcription."
            )
            break

    # clean up script, remove double spaces and trim start and end
    entire_script = entire_script.strip().replace("  ", " ")

    transcript = Transcription(
        entire_script=entire_script,
        segments=segments,
        words=words,
    )

    # Always write the transcript text to a file
    with open(f'{output_path}.txt', "w") as f:
        f.write(transcript.entire_script)
        print(f"Transcription text written to ./data/transcripts.output.txt")

    # Only write the transcript to a JSON file if the flag is present
    if create_json_file:
        with open(f'{output_path}.json', "w") as f:
            f.write(transcript.model_dump_json())
            print(f"Transcription json written to ./data/transcripts.output.json")
            
    return transcript