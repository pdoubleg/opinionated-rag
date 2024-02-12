import asyncio
from itertools import chain
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import os
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
import aiohttp
from httpx import Timeout
from openai import AsyncOpenAI, OpenAI
from openai.types.completion import Completion
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion
import openai
import requests
from rich.markup import escape
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from diskcache import Cache
import functools
import inspect

from src.llm.utils import (
    async_retry_with_exponential_backoff,
    retry_with_exponential_backoff,
    async_cache_decorator_factory,
)

import instructor
from src.utils.configuration import Settings
from src.utils.constants import NO_ANSWER
from src.utils.system import friendly_error

logger = logging.getLogger(__name__)

settings = Settings(debug=True)


class OpenAIChatModel(str, Enum):
    """Enum for OpenAI Chat models"""

    GPT3_5_TURBO = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo-preview"


_context_length: Dict[str, int] = {
    OpenAIChatModel.GPT3_5_TURBO: 16_385,
    OpenAIChatModel.GPT4: 8192,
    OpenAIChatModel.GPT4_TURBO: 128_000,
}

_cost_per_1k_tokens: Dict[str, Tuple[float, float]] = {
    # model => (prompt cost, generation cost) in USD
    OpenAIChatModel.GPT3_5_TURBO: (0.001, 0.002),
    OpenAIChatModel.GPT4: (0.03, 0.06),  # 8K context
    OpenAIChatModel.GPT4_TURBO: (0.01, 0.03),  # 128K context
}


openAIChatModelPreferenceList = [
    OpenAIChatModel.GPT4_TURBO,
    OpenAIChatModel.GPT4,
    OpenAIChatModel.GPT3_5_TURBO,
]

# Initialize Cache from config
cache = Cache("/.mycache")

# Create an async decorator with the configured cache
instructor_acache = async_cache_decorator_factory(cache)


if "OPENAI_API_KEY" in os.environ:
    try:
        availableModels = set(map(lambda m: m.id, OpenAI().models.list()))
    except openai.AuthenticationError as e:
        if settings.debug:
            logger.warning(
                f"""
            OpenAI Authentication Error: {e}.
            ---
            If you intended to use an OpenAI Model, you should fix this.
            """
            )
        availableModels = set()
else:
    availableModels = set()


defaultOpenAIChatModel = next(
    chain(
        filter(
            lambda m: m.value in availableModels,
            openAIChatModelPreferenceList,
        ),
        [OpenAIChatModel.GPT3_5_TURBO],
    )
)


class OpenAICallParams(BaseModel):
    """
    Various params that can be sent to an OpenAI API chat-completion call.
    When specified, any param here overrides the one with same name in the
    OpenAIGPTConfig.
    """

    max_tokens: int = 1024
    temperature: float = 0.2
    frequency_penalty: float | None = 0.0  # between -2 and 2
    presence_penalty: float | None = 0.0  # between -2 and 2
    logit_bias: Dict[int, float] | None = None  # token_id -> bias
    logprobs: bool = False
    top_p: int | None = 1
    top_logprobs: int | None = None  # if int, requires logprobs=True
    stop: str | List[str] | None = None  # (list of) stop sequence(s)
    user: str | None = None  # user id for tracking

    def to_dict_exclude_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class OpenAIConfig(BaseSettings):
    type: str = "openai"
    api_key: str = os.getenv("OPENAI_API_KEY")
    api_base: Optional[str] = None
    model: str = defaultOpenAIChatModel
    response_model: Optional[BaseModel] = None
    max_retries: Optional[int] = 1
    validation_context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    context_length: int = 1024
    temperature: float = 0.0
    max_tokens: int = 4096
    params: OpenAICallParams | None = None
    # Dict of model -> (input/prompt cost, output/completion cost)
    cost_per_1k_tokens: Tuple[float, float] = (0.0, 0.0)
    timeout: int = 20
    cache_config: Optional[str] | None = "./my_cache_directory"
    cache: Optional[Cache] | None = None

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        env_prefix="OPENAI_",
    )
    # class Config:
    #     env_prefix = "OPENAI_"

    @classmethod
    def create(cls, prefix: str) -> Type["OpenAIConfig"]:
        """Create a config class whose params can be set via a desired
        prefix from the .env file or env vars.
        """

        class DynamicConfig(OpenAIConfig):
            pass

        DynamicConfig.env_prefix = prefix.upper() + "_"
        if cls.cache_config:
            DynamicConfig.cache = Cache(cls.cache_config)

        return DynamicConfig


class LLMTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    calls: int = 0  # how many API calls

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0
        self.calls = 0

    def __str__(self) -> str:
        return (
            f"Tokens = "
            f"(prompt {self.prompt_tokens}, completion {self.completion_tokens}), "
            f"Cost={self.cost}, Calls={self.calls}"
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Role(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class LLMMessage(BaseModel):
    """
    Class representing message sent to, or received from, LLM.
    """

    role: Role
    name: Optional[str] = None
    tool_id: str = ""  # used by OpenAIAssistant
    content: Optional[str] = None
    response_model: Optional[BaseModel | List[BaseModel]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def api_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for API request.
        DROP the tool_id, since it is only for use in the Assistant API,
        not the completion API.
        Returns:
            dict: dictionary representation of LLM message
        """
        d = self.model_dump()
        # drop None values since API doesn't accept them
        dict_no_none = {k: v for k, v in d.items() if v is not None}
        if "name" in dict_no_none and dict_no_none["name"] == "":
            # OpenAI API does not like empty name
            del dict_no_none["name"]
        if "function_call" in dict_no_none:
            # arguments must be a string
            if "arguments" in dict_no_none["function_call"]:
                dict_no_none["function_call"]["arguments"] = json.dumps(
                    dict_no_none["function_call"]["arguments"]
                )
        dict_no_none.pop("tool_id", None)
        dict_no_none.pop("timestamp", None)
        return dict_no_none

    def __str__(self) -> str:
        if self.response_model is not None:
            content = "FUNC: " + self.response_model.model_dump_json(indent=2)
        else:
            content = self.content
        name_str = f" ({self.name})" if self.name else ""
        return f"{self.role} {name_str}: {content}"


class LLMResponse(BaseModel):
    """
    Class representing response from LLM.
    """

    message: Optional[str] = None
    tool_id: str = ""  # used by OpenAIAssistant
    response_model: Optional[BaseModel | List[BaseModel]] = (None,)
    usage: Optional[LLMTokenUsage] = None

    def __str__(self) -> str:
        if self.response_model is not None:
            return str(self.response_model.model_dump_json(indent=2))
        else:
            return self.message

    def to_LLMMessage(self) -> LLMMessage:
        content = self.message
        role = Role.ASSISTANT if self.response_model is None else Role.FUNCTION
        name = None if self.response_model is None else self.response_model.name
        return LLMMessage(
            role=role,
            name=name,
            content=content,
            response_model=self.response_model,
        )


# Define an abstract base class for language models
class LanguageModel(ABC):
    """
    Abstract base class for language models.
    """

    # usage cost by model, accumulates here
    usage_cost_dict: Dict[str, LLMTokenUsage] = {}

    def __init__(self, config: OpenAIConfig = OpenAIConfig()):
        self.config = config

    @staticmethod
    def create(config: Optional[OpenAIConfig]) -> Optional["LanguageModel"]:
        """
        Create a language model.
        Args:
            config: configuration for language model
        Returns: instance of language model
        """
        from openai import OpenAI, AsyncOpenAI

        if config is None or config.type is None:
            return None

        openai: Union[Type[OpenAIGPT], Type[AsyncOpenAI]]

        if config.type == "async":
            openai = AsyncOpenAI
        else:
            openai = OpenAI
        cls = dict(
            openai=openai,
        ).get(config.type, openai)
        return cls(config)  # type: ignore

    @staticmethod
    def user_assistant_pairs(lst: List[str]) -> List[Tuple[str, str]]:
        """
        Given an even-length sequence of strings, split into a sequence of pairs

        Args:
            lst (List[str]): sequence of strings

        Returns:
            List[Tuple[str,str]]: sequence of pairs of strings
        """
        evens = lst[::2]
        odds = lst[1::2]
        return list(zip(evens, odds))

    @staticmethod
    def get_chat_history_components(
        messages: List[LLMMessage],
    ) -> Tuple[str, List[Tuple[str, str]], str]:
        """
        From the chat history, extract system prompt, user-assistant turns, and
        final user msg.

        Args:
            messages (List[LLMMessage]): List of messages in the chat history

        Returns:
            Tuple[str, List[Tuple[str,str]], str]:
                system prompt, user-assistant turns, final user msg

        """
        # Handle various degenerate cases
        messages = [m for m in messages]  # copy
        DUMMY_SYS_PROMPT = "You are a helpful assistant."
        DUMMY_USER_PROMPT = "Follow the instructions above."
        if len(messages) == 0 or messages[0].role != Role.SYSTEM:
            logger.warning("No system msg, creating dummy system prompt")
            messages.insert(0, LLMMessage(content=DUMMY_SYS_PROMPT, role=Role.SYSTEM))
        system_prompt = messages[0].content

        # now messages = [Sys,...]
        if len(messages) == 1:
            logger.warning(
                "Got only system message in chat history, creating dummy user prompt"
            )
            messages.append(LLMMessage(content=DUMMY_USER_PROMPT, role=Role.USER))

        # now we have messages = [Sys, msg, ...]

        if messages[1].role != Role.USER:
            messages.insert(1, LLMMessage(content=DUMMY_USER_PROMPT, role=Role.USER))

        # now we have messages = [Sys, user, ...]
        if messages[-1].role != Role.USER:
            logger.warning(
                "Last message in chat history is not a user message,"
                " creating dummy user prompt"
            )
            messages.append(LLMMessage(content=DUMMY_USER_PROMPT, role=Role.USER))

        # now we have messages = [Sys, user, ..., user]
        # so we omit the first and last elements and make pairs of user-asst messages
        conversation = [m.content for m in messages[1:-1]]
        user_prompt = messages[-1].content
        pairs = LanguageModel.user_assistant_pairs(conversation)
        return system_prompt, pairs, user_prompt

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        pass

    @abstractmethod
    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        pass

    @abstractmethod
    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 1200,
        response_model: Optional[BaseModel | List[BaseModel]] = None,
        max_retries: Optional[int] = 1,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def achat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        response_model: Optional[BaseModel | List[BaseModel]] = None,
        max_retries: Optional[int] = None,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        pass

    def __call__(self, prompt: str, max_tokens: int) -> LLMResponse:
        return self.generate(prompt, max_tokens)

    def context_length(self) -> int:
        return self.config.context_length

    def usage_cost(self) -> Tuple[float, float]:
        return self.config.cost_per_1k_tokens

    def reset_usage_cost(self) -> None:
        for mdl in [self.config.model]:
            if mdl is None:
                return
            if mdl not in self.usage_cost_dict:
                self.usage_cost_dict[mdl] = LLMTokenUsage()
            counter = self.usage_cost_dict[mdl]
            counter.reset()

    @classmethod
    def usage_cost_summary(cls) -> str:
        s = ""
        for model, counter in cls.usage_cost_dict.items():
            s += f"{model}: {counter}\n"
        return s


class OpenAIGPT(LanguageModel):
    """
    Class for OpenAI LLMs
    """

    def __init__(self, config: OpenAIConfig = OpenAIConfig()):
        """
        Args:
            config: configuration for openai-gpt model
        """
        # copy the config to avoid modifying the original
        config = config.model_copy()
        super().__init__(config)
        self.config: OpenAIConfig = config

        self.api_key = config.api_key
        self.client = instructor.patch(
            OpenAI(
                api_key=self.api_key,
                timeout=Timeout(self.config.timeout),
            )
        )
        self.async_client = instructor.patch(
            AsyncOpenAI(
                api_key=self.api_key,
                timeout=Timeout(self.config.timeout),
            )
        )

    def _openai_api_call_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prep the params to be sent to the OpenAI API

        Order of priority:
        - (1) Params (mainly max_tokens) in the chat/achat call
                (these are passed in via kwargs)
        - (2) Params in OpenAIGPTConfi.params (of class OpenAICallParams)
        - (3) Specific Params in OpenAIGPTConfig (just temperature for now)
        """
        params = dict(
            temperature=self.config.temperature,
        )
        if self.config.params is not None:
            params.update(self.config.params.to_dict_exclude_none())
        params.update(kwargs)
        return params

    def context_length(self) -> int:
        """
        Context-length for chat-completion models/endpoints
        Get it from the dict, otherwise fail-over to general method
        """
        model = self.config.model

        return _context_length.get(model, super().context_length())

    def usage_cost(self) -> Tuple[float, float]:
        """
        (Prompt, Generation) cost per 1000 tokens, for chat-completion
        models/endpoints.
        Get it from the dict, otherwise fail-over to general method
        """
        return _cost_per_1k_tokens.get(self.config.model, super().usage_cost())

    def _model_cost(self, prompt: int, completion: int) -> float:
        price = self.usage_cost()
        return (price[0] * prompt + price[1] * completion) / 1000

    def _get_token_usage(self, response: Dict[str, Any]) -> LLMTokenUsage:
        """
        Extracts token usage from ``response`` and computes cost, only when NOT
        in streaming mode, since the LLM API (OpenAI currently) does not populate the
        usage fields in streaming mode. In streaming mode, these are set to zero for
        now, and will be updated later by the fn ``update_token_usage``.
        """
        cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0

        if isinstance(response, BaseModel):
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        else:
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
        cost = self._model_cost(prompt_tokens, completion_tokens)

        return LLMTokenUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost
        )

    def generate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        try:
            return self._generate(prompt, max_tokens)
        except Exception as e:
            # capture exceptions not handled by retry, so we don't crash
            logging.error(friendly_error(e, "Error in OpenAIGPT.generate: "))
            return LLMResponse(message=NO_ANSWER)

    def _generate(self, prompt: str, max_tokens: int) -> LLMResponse:
        return self.chat(messages=prompt, max_tokens=max_tokens)

    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        try:
            return await self._agenerate(prompt, max_tokens)
        except Exception as e:
            # capture exceptions not handled by retry, so we don't crash
            logger.error(friendly_error(e, "Error in OpenAIGPT.agenerate: "))
            return LLMResponse(message=NO_ANSWER)

    async def _agenerate(self, prompt: str, max_tokens: int) -> LLMResponse:
        return await self.achat(messages=prompt, max_tokens=max_tokens)

    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        response_model: Optional[BaseModel | List[BaseModel]] = None,
        max_retries: Optional[int] = 1,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        try:
            return self._chat(
                messages, max_tokens, response_model, max_retries, validation_context
            )
        except Exception as e:
            # capture exceptions not handled by retry, so we don't crash
            logger.error(friendly_error(e, "Error in OpenAIGPT.chat: "))
            return LLMResponse(message=NO_ANSWER)

    async def achat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        response_model: Optional[BaseModel | List[BaseModel]] = None,
        max_retries: Optional[int] = 1,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        try:
            result = await self._achat(
                messages, max_tokens, response_model, max_retries, validation_context
            )
            return result
        except Exception as e:
            # capture exceptions not handled by retry, so we don't crash
            logger.error(friendly_error(e, "Error in OpenAIGPT.achat: "))
            return LLMResponse(message=NO_ANSWER)

    @retry_with_exponential_backoff
    def _chat_completions_with_backoff(self, **kwargs) -> Completion:
        completion_call = self.client.chat.completions.create
        result: Completion = completion_call(**kwargs)
        return result

    @async_retry_with_exponential_backoff
    async def _achat_completions_with_backoff(self, **kwargs) -> Completion:
        acompletion_call = self.async_client.chat.completions.create
        result: Completion = await acompletion_call(**kwargs)
        return result

    def _prep_chat_completion(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        response_model: Optional[BaseModel | List[BaseModel]] = None,
        max_retries: Optional[int] = 1,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if isinstance(messages, str):
            llm_messages = [
                LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
                LLMMessage(role=Role.USER, content=messages),
            ]
        else:
            llm_messages = messages

        model = self.config.model
        if self.config.type == "azure":
            if hasattr(self, "deployment_name"):
                model = self.deployment_name

        args: Dict[str, Any] = dict(
            model=model,
            messages=[m.api_dict() for m in llm_messages],
            max_tokens=max_tokens,
            # response_model=response_model,
            # max_retries=max_retries,
            # validation_context=validation_context,
        )
        args.update(self._openai_api_call_params(args))
        # only include functions-related args if functions are provided
        # since the OpenAI API will throw an error if `functions` is None or []
        if response_model is not None:
            args.update(
                dict(
                    response_model=response_model,
                    max_retries=max_retries,
                    validation_context=validation_context,
                )
            )
        return args

    def _process_chat_completion_response(
        self,
        response: BaseModel | Dict[str, Any],
    ) -> LLMResponse:
        if isinstance(response, dict):
            message = response["choices"][0]["message"]
            msg = message["content"] or ""
            response_ = response
            fun_call = message["function_call"]["arguments"]

        elif isinstance(response, ChatCompletion):
            response_ = response
            msg = response.choices[0].message.content
            fun_call = response.choices[0].message.function_call

        elif isinstance(response, BaseModel):
            try:
                fun_call = response
                response_ = response._raw_response
                message = response_.choices[0].message
                msg = message.content or ""

            except (ValueError, SyntaxError, AssertionError):
                logger.warning(
                    "Could not parse function arguments: "
                    f"{message['function_call']['arguments']} "
                    f"for function {message['function_call']['name']} "
                    "treating as normal non-function message"
                )
                fun_call = None
                args_str = message["function_call"]["arguments"] or ""
                msg_str = message["content"] or ""
                msg = msg_str + args_str

        return LLMResponse(
            message=msg.strip() if msg is not None else "",
            response_model=fun_call,
            usage=self._get_token_usage(response_),
        )

    def _chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        response_model: Optional[BaseModel | List[BaseModel]] = None,
        max_retries: Optional[int] = 1,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        ChatCompletion API call to OpenAI.
        Args:
            messages: list of messages  to send to the API, typically
                represents back and forth dialogue between user and LLM, but could
                also include "function"-role messages.
            max_tokens: max output tokens to generate
        Returns:
            LLMResponse object
        """
        args = self._prep_chat_completion(
            messages,
            max_tokens,
            response_model,
            max_retries,
            validation_context,
        )
        response = self._chat_completions_with_backoff(**args)

        if isinstance(response, Union[dict, BaseModel]):
            response_to_process = response

        else:
            response_to_process = response.model_dump()
        return self._process_chat_completion_response(response_to_process)

    async def _achat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        response_model: Optional[BaseModel | List[BaseModel]] = None,
        max_retries: Optional[int] = 1,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Async version of _chat(). See that function for details.
        """
        args = self._prep_chat_completion(
            messages,
            max_tokens,
            response_model,
            max_retries,
            validation_context,
        )
        response = await self._achat_completions_with_backoff(**args)
        if isinstance(response, Union[dict, BaseModel]):
            response_to_process = response

        else:
            response_to_process = response.model_dump()
        return self._process_chat_completion_response(response_to_process)

#     async def _openai_api_call(
#         self, 
#         func: Callable[..., Awaitable[Any]],
#         errors: tuple = (  # type: ignore
#         requests.exceptions.RequestException,
#         openai.APITimeoutError,
#         openai.RateLimitError,
#         aiohttp.ServerTimeoutError,
#         asyncio.TimeoutError,
#     ),
#         **kwargs
#     ) -> Optional[Any]:
#         """
#         Makes an asynchronous API call with automatic retries on failure.

#         Args:
#             func (Callable[..., Awaitable[Any]]): The asynchronous function to call.
#             **kwargs: Arbitrary keyword arguments passed to the function.

#         Returns:
#             Optional[Any]: The result of the API call, or None if the call fails after retries.
#         """
#         max_retries: int = 5
#         retry_count: int = 0
#         base_wait_time: int = 1

#         while retry_count <= max_retries:
#             try:
#                 return await func(**kwargs)
#             except errors as e:
#                 logger.error(f"Attempt {retry_count} failed with error: {e}", exc_info=True)
#                 logger.warning(f"{friendly_error(e)} Retrying... {2 ** retry_count}s")
#                 retry_count += 1
#                 await asyncio.sleep(base_wait_time * (2**retry_count))
#             except (openai.BadRequestError, openai.PermissionDeniedError) as e:
#                 logger.error(friendly_error(e))
#                 return None
#             except Exception as e:
#                 logger.error(f"Unexpected error: {e}. Exiting...")
#                 return None

#         logger.error("Max retries reached. Exiting...")
#         return None

#     async def handle_multiple_prompts(self, prompts, response_model):
#         tasks = [
#             self._create_chat_completion(prompt, response_model) for prompt in prompts
#         ]
#         responses = await asyncio.gather(*tasks, return_exceptions=True)

#         return responses

#     @instructor_acache
#     async def _create_chat_completion(
#         self,
#         prompt,
#         response_model=None,
#     ):
#         async def _api_call():
#             return await self.async_client.chat.completions.create(
#                 model=self.config.model,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt},
#                 ],
#                 response_model=response_model,
#             )

#         response = await self._openai_api_call(_api_call)

#         return response


# # Usage example:
# class Variable(BaseModel):
#     variable: str = Field(..., description="Variable name")
#     value: Optional[str] = Field(..., description="Variable value")
#     unit: str = Field(..., description="Variable unit")


# class SimpleCalculation(BaseModel):
#     expression: str = Field(..., description="Expression to calculate")
#     lead_variable: Variable = Field(
#         ...,
#         description="Lead variable. You must calculate the value using the 'expression' and 'values'",
#     )
#     variables: List[Variable] = Field(..., description="List of variables")


# # Define the prompts
# beam = "5m simply supported beam with a central point load of P = 10kN and constant EI"
# prompts = [
#     f"Bending moment formula: {beam}",
#     f"Deflection formula: {beam}",
#     f"Shear force formula: {beam}",
# ]

# # Define model
# model = "gpt-4-1106-preview"
# openai_client = OpenAIGPT()


# async def main() -> None:
#     # Get the responses
#     responses = await openai_client.handle_multiple_prompts(
#         prompts,
#         response_model=SimpleCalculation,
#     )

#     for response in responses:
#         print("-"*100)
#         print(response._raw_response.usage)
#         print(response.model_dump_json(indent=2))


# asyncio.run(main())
