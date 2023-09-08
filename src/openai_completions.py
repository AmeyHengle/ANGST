import os
import logging
import openai
from openai import error
import aiolimiter
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
from litellm import (
    completion, 
    acompletion
)



ERROR_ERRORS_TO_MESSAGES = {
    error.InvalidRequestError: "OpenAI API Invalid Request: Prompt was filtered",
    error.RateLimitError: "OpenAI API rate limit exceeded. Sleeping for 10 seconds.",
    error.APIConnectionError: "OpenAI API Connection Error: Error Communicating with OpenAI",  # noqa E501
    error.Timeout: "OpenAI APITimeout Error: OpenAI Timeout",
    error.ServiceUnavailableError: "OpenAI service unavailable error: {e}",
    error.APIError: "OpenAI API error: {e}",
}



async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        for _ in range(3):
            try:
                return await completion(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:
                if isinstance(e, (error.ServiceUnavailableError, error.APIError)):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                elif isinstance(e, error.InvalidRequestError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Invalid Request: Prompt was filtered"
                                }
                            }
                        ]
                    }
                else:
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}



async def generate_from_openai_completion(
    prompts: list,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    requests_per_minute: int = 150,
):
    """Generate from OpenAI Completion API.

    Args:
        prompts: List of prompts to generate from.
        model: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of completions to generate for each API call.
        api_key: the OPENAI_API_KEY
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if api_key not in os.environ:
        raise ValueError(
            f"{api_key} environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ[api_key]
    print(f"\nUsing {API_KEY}\n")
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    return [x["choices"][0]["text"] for x in responses]



async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        for _ in range(3):
            try:
                return await acompletion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:
                if isinstance(e, (error.ServiceUnavailableError, error.APIError)):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                elif isinstance(e, error.InvalidRequestError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Invalid Request: Prompt was filtered"
                                }
                            }
                        ]
                    }
                else:
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}



async def generate_from_openai_chat_completion(
    messages_list: list,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    requests_per_minute: int = 150
):
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of responses to generate for each API call.
        context_length: Length of context to use.
        api_key: the OPENAI_API_KEY
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if api_key not in os.environ:
        raise ValueError(
            f"{api_key} environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ[api_key]
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model,
            messages=full_context,
            temperature=temperature,
            max_tokens=max_tokens,
            limiter=limiter,
        )
        for full_context in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    return [x["choices"][0]["message"]["content"] for x in responses]