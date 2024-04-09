import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logging import setup_colored_logging

logger = setup_colored_logging(__name__)

class CommonSettings(BaseSettings):
    openai_api_key: str = Field(
        default=os.getenv("OPENAI_API_KEY"),
        validation_alias='api_key',
    )
    environment: str = Field(
        default=os.getenv("ENVIRONMENT", "base")
    )
    model_config = SettingsConfigDict(
        env_file = ".env",
        validate_default=True,
        env_file_encoding='utf-8',
        extra='ignore',
    )

class DevelopmentSettings(CommonSettings):
    api_key: str
    model_config = SettingsConfigDict(
        env_prefix = "TEST_"
    )
    
class NonProductionSettings(CommonSettings):
    api_key: str
    model_config = SettingsConfigDict(
        env_prefix = "NP_"
    )

class ProductionSettings(CommonSettings):
    api_key: str
    model_config = SettingsConfigDict(
        env_prefix = "PROD_"
    )


def get_settings() -> BaseSettings:
    import os
    env = os.getenv("ENVIRONMENT", "base").lower()
    if env == "base":
        logger.info(f"Loaded env: {env}")
        return CommonSettings()
    if env == "test":
        logger.info(f"Loaded env: {env}")
        return DevelopmentSettings()
    elif env == "np":
        logger.info(f"Loaded env: {env}")
        return NonProductionSettings()
    elif env == "prod":
        logger.info(f"Loaded env: {env}")
        return ProductionSettings()
    else:
        raise ValueError(f"Unknown environment: {env}")



# Example usage:
# os.environ["ENVIRONMENT"] = "base"
# settings = get_settings('base')
