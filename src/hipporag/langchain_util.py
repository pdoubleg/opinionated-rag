import os

import tiktoken

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_by_tiktoken(text: str):
    return len(enc.encode(text))


class LangChainModel:
    def __init__(self, provider: str, model_name: str, **kwargs):
        self.provider = provider
        self.model_name = model_name
        self.kwargs = kwargs


def init_langchain_model(
    model_name: str,
    temperature: float = 0.0,
    max_retries=5,
    timeout=60,
    **kwargs,
):
    """
    Initialize a language model from the langchain library.
    :param llm: The LLM to use, e.g., 'openai', 'together'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    """
    # https://python.langchain.com/v0.1/docs/integrations/chat/openai/
    from langchain_openai import ChatOpenAI

    assert model_name.startswith("gpt-")
    return ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=model_name,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
        **kwargs,
    )
