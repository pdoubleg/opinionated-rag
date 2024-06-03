import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from modules import llm, omnicomplete

logging.basicConfig(level=logging.INFO)

topics = [("Legal Questions", "LSS")]
topic_index = 0

load_dotenv()
app = FastAPI()

client = AsyncOpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def test():
    return {"hello": "world"}


@app.post("/get-autocomplete")
async def get_autocomplete(request: Request) -> JSONResponse:
    input_data = (await request.json())["input"]
    prompt = omnicomplete.build_omni_complete_prompt(
        input_data, topic=topics[topic_index][0], topic_dir=topics[topic_index][1]
    )
    response = llm.prompt_json(prompt)
    return JSONResponse(content=response)


@app.post("/use-autocomplete")
async def do_autocomplete(request: Request) -> JSONResponse:
    """Handles the use autocomplete request.

    Args:
        request (Request): The request object containing the autocomplete data.

    Returns:
        JSONResponse: JSON response indicating success or failure.
    """
    autocomplete_object = await request.json()
    if "input" not in autocomplete_object or "completion" not in autocomplete_object:
        raise HTTPException(status_code=400, detail="Invalid autocomplete object")

    input_data = autocomplete_object["input"]
    completion = autocomplete_object["completion"]
    print(f"Received autocomplete object: input={input_data}, completion={completion}")

    omnicomplete.increment_or_create_previous_completions(
        input_data, completion, topics[topic_index][1]
    )

    return JSONResponse(content={"success": True})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="debug")
