from pydantic_settings import BaseSettings


class PromptsConfig(BaseSettings):
    max_tokens: int = 1000
