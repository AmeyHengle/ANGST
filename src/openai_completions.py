import os
import logging
import openai
import aiolimiter
from aiohttp import ClientSession
from typing import Optional
import asyncio
from tqdm.asyncio import tqdm_asyncio


ERROR_ERRORS_TO_MESSAGES = {
    openai.error.InvalidRequestError: "OpenAI API Invalid Request: Prompt was filtered",
    openai.error.RateLimitError: "OpenAI API rate limit exceeded. Sleeping for 10 seconds.",
    openai.error.APIConnectionError: "OpenAI API Connection Error: Error Communicating with OpenAI",  # noqa E501
    openai.error.Timeout: "OpenAI APITimeout Error: OpenAI Timeout",
    openai.error.ServiceUnavailableError: "OpenAI service unavailable error: {e}",
    openai.error.APIError: "OpenAI API error: {e}",
}



async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        for _ in range(3):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:
                if isinstance(e, (openai.error.ServiceUnavailableError, openai.error.APIError)):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                elif isinstance(e, openai.error.InvalidRequestError):
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
    top_p: float,
    max_tokens: int,
    api_key: str,
    requests_per_minute: int = 150,
    org_key: Optional[str] = None,
):
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        top_p: Top p probabilities to use.
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
    
    if org_key is not None:
        if org_key not in os.environ:
            raise ValueError(
                f"{org_key} environment variable must be set when using OpenAI API."
            )
        openai.organization = os.environ[org_key]
        
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model,
            messages=full_context,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            limiter=limiter,
        )
        for full_context in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    return [x["choices"][0]["message"]["content"] for x in responses]