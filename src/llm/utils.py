# from openai-cookbook
import asyncio
import functools
import inspect
from itertools import islice
import logging
import random
import time
from typing import Any, Callable, Dict, Iterable, List

import aiohttp
import openai
from pydantic import BaseModel
import requests
from diskcache import Cache

# from src.utils.system import friendly_error

logger = logging.getLogger(__name__)
# setlevel to warning
logger.setLevel(logging.INFO)


# cache = Cache("./my_cache_directory")


# def instructor_cache(func):
#     """Cache a function that returns a Pydantic model"""
#     return_type = inspect.signature(func).return_annotation
#     if not issubclass(return_type, BaseModel):
#         raise ValueError("The return type must be a Pydantic model")

#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
#         # Check if the result is already cached
#         if (cached := cache.get(key)) is not None:
#             # Deserialize from JSON based on the return type
#             if issubclass(return_type, BaseModel):
#                 return return_type.model_validate_json(cached)

#         # Call the function and cache its result
#         result = func(*args, **kwargs)
#         serialized_result = result.model_dump_json()
#         cache.set(key, serialized_result)

#         return result

#     return wrapper

# Example:
#
# class UserDetail(BaseModel):
#     name: str
#     age: int


# @instructor_cache
# def extract(data) -> UserDetail:
#     return client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         response_model=UserDetail,
#         messages=[
#             {"role": "user", "content": data},
#         ],
#     )

# model = extract("Extract jason is 25 years old")


# def cache_decorator_factory(cache: Cache):
#     def instructor_cache(func):
#         """Cache decorator for methods returning Pydantic models."""
#         @functools.wraps(func)
#         def wrapper(self, *args, **kwargs):
#             # Generate a unique key based on the function name and arguments
#             key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
#             # Check if the result is already cached
#             if (cached := cache.get(key)) is not None:
#                 # Deserialize from JSON based on the return type
#                 return_type = inspect.signature(func).return_annotation
#                 if issubclass(return_type, BaseModel):
#                     return return_type.parse_raw(cached)
#             # Call the function and cache its result
#             result = func(self, *args, **kwargs)
#             serialized_result = result.json()
#             cache.set(key, serialized_result)
#             return result
#         return wrapper
#     return instructor_cache


def async_cache_decorator_factory(cache: Cache):
    def instructor_acache(func):
        """Cache decorator for asynchronous methods returning Pydantic models."""

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Generate a unique key based on the function name and arguments
            key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
            # Check if the result is already cached
            if (cached := cache.get(key)) is not None:
                # Deserialize from JSON based on the return type
                return_type = inspect.signature(func).return_annotation
                if issubclass(return_type, BaseModel):
                    return return_type.parse_raw(cached)
            # Call the function and cache its result
            result = await func(self, *args, **kwargs)
            serialized_result = result.json()
            cache.set(key, serialized_result)
            return result

        return wrapper

    return instructor_acache


def batched(iterable: Iterable[Any], n: int) -> Iterable[Any]:
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


# define a retry decorator
def retry_with_exponential_backoff(
    func: Callable[..., Any],
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (  # type: ignore
        requests.exceptions.RequestException,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.APIError,
        aiohttp.ServerTimeoutError,
        asyncio.TimeoutError,
    ),
) -> Callable[..., Any]:
    """Retry a function with exponential backoff."""

    def wrapper(*args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            except openai.BadRequestError as e:
                # do not retry when the request itself is invalid,
                # e.g. when context is too long
                logger.error(f"OpenAI API request failed with error: {e}.")
                logger.error(friendly_error(e))
                raise e

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                logger.warning(
                    f"""OpenAI API request failed with error: 
                    {e}. 
                    Retrying in {delay} seconds..."""
                )
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def async_retry_with_exponential_backoff(
    func: Callable[..., Any],
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (  # type: ignore
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.APIError,
        aiohttp.ServerTimeoutError,
        asyncio.TimeoutError,
    ),
) -> Callable[..., Any]:
    """Retry a function with exponential backoff."""

    async def wrapper(*args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or exception is raised
        while True:
            try:
                result = await func(*args, **kwargs)
                return result

            except openai.BadRequestError as e:
                # do not retry when the request itself is invalid,
                # e.g. when context is too long
                logger.error(f"OpenAI API request failed with error: {e}.")
                logger.error(friendly_error(e))
                raise e

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                logger.warning(
                    f"""OpenAI API request failed with error{e}. 
                    Retrying in {delay} seconds..."""
                )
                # Sleep for the delay
                await asyncio.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper
