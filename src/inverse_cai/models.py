from typing import Any
import json
import numpy as np
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks import get_openai_callback
import inverse_cai.local_secrets  # required to load env vars
from loguru import logger

NULL_LOGPROB_VALUE = -1000000


# Custom OpenRouter headers
# https://openrouter.ai/docs/api-reference/overview#headers
OPENROUTER_HEADERS = {
    "X-Title": "ICAI",
    # We need to set HTTP-Referer in addition to X-Title since otherwise openrouter does
    # not show an App name in the activity overview. If we set both, it shows the X-Title
    # and links to HTTP-Referer.
    "HTTP-Referer": "https://github.com/rdnfn/icai",
}


def get_model(
    name: str,
    temp: float = 0.0,
    enable_logprobs: bool = False,
    max_tokens: int = 1000,
) -> Any:
    """Get a language model instance.

    Args:
        name: Model name with provider prefix (e.g. "openai/gpt-4o-2024-05-13")
        temp: Temperature for generation (default: 0.0)
        enable_logprobs: Whether to enable logprobs for token probabilities (default: False)
        max_tokens: Maximum tokens to generate (default: 1000)

    Returns:
        LogWrapper-wrapped language model instance
    """
    if enable_logprobs:
        model_kwargs = {"logprobs": True, "top_logprobs": 10}
    else:
        model_kwargs = {}

    if name.startswith("openai"):
        return ChatOpenAI(
            model=name.split("/")[1],
            max_tokens=max_tokens,
            temperature=temp,
            model_kwargs=model_kwargs,
        )

    if name.startswith("anthropic"):
        return ChatAnthropic(
            model=name.split("/")[1],
            max_tokens=max_tokens,
            temperature=temp,
            model_kwargs=model_kwargs,
        )

    if name.startswith("openrouter"):

        # Extract the actual model from openrouter/provider/model format
        parts = name.split("/", 2)
        if len(parts) < 3:
            raise ValueError(
                "OpenRouter model format should be 'openrouter/provider/model'"
            )

        # Use the provider/model as the model name for OpenRouter
        model_id = "/".join(parts[1:])

        # Get the OpenRouter API key from environment variable
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable must be set for OpenRouter models"
            )

        return ChatOpenAI(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temp,
            openai_api_key=openrouter_api_key,  # Use the OpenRouter API key
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers=OPENROUTER_HEADERS,
            model_kwargs=model_kwargs,
        )


def get_embeddings_model(name):
    # TODO: support other embeddings?

    openrouter = name.startswith("openrouter")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openrouter and not openai_api_key:
        raise ValueError(
            "OpenRouter doesn't support embedding models, OPENAI_API_KEY still required"
        )

    return OpenAIEmbeddings()


def get_token_probs(tokens: list[str], model: str, messages: list) -> dict[float]:
    """
    Get the token probabilities for a list of tokens.
    """

    def get_first_generation(llm_generate_return_val):
        return llm_generate_return_val.generations[0][0]

    def get_log_prob_for_token(generation, token):
        top_logprobs = generation.generation_info["logprobs"]["content"][0][
            "top_logprobs"
        ]
        for logprob in top_logprobs:
            if logprob["token"] == token:
                return logprob["logprob"]
        logger.warning(
            f"Token {token} not found in top logprobs. Returning {NULL_LOGPROB_VALUE} logprob (close to 0 probability for token)."
        )
        return NULL_LOGPROB_VALUE

    def get_norm_probs_for_tokens(generation, tokens):
        logprobs = [get_log_prob_for_token(generation, token) for token in tokens]
        errors = []

        if any(logprob == NULL_LOGPROB_VALUE for logprob in logprobs):
            for token, logprob in zip(tokens, logprobs):
                if logprob == NULL_LOGPROB_VALUE:
                    errors.append(f"token_not_found_in_top_logprobs_{token}")

        if all(logprob == NULL_LOGPROB_VALUE for logprob in logprobs):
            logger.warning(
                f"All tokens not found in top logprobs. Returning equal probabilities for all tokens ({tokens})."
            )
            normalised_probs = [0.5 for _ in tokens]
            errors.append("neither_tokens_found_in_top_logprobs")
        else:
            probs = [np.exp(logprob) for logprob in logprobs]
            normalised_probs = [prob / sum(probs) for prob in probs]
        prob_dict = dict(zip(tokens, normalised_probs))
        return prob_dict, errors

    return_val = model.generate([messages])
    generation = get_first_generation(return_val)

    token_probs, errors = get_norm_probs_for_tokens(
        generation=generation, tokens=tokens
    )

    return token_probs, errors
