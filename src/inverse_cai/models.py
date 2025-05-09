from typing import Any
import json
import numpy as np
import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks import get_openai_callback
import inverse_cai.local_secrets
from loguru import logger

NULL_LOGPROB_VALUE = -1000000


class LogWrapper:
    def __init__(self, wrapped_instance):
        """Initialize the wrapper with the instance to wrap."""
        self._wrapped_instance = wrapped_instance
        self._logger = logger

        if hasattr(wrapped_instance, "model_name"):
            self.model_name = wrapped_instance.model_name
        elif hasattr(wrapped_instance, "model"):
            self.model_name = wrapped_instance.model

    def __getattr__(self, attr):
        """
        Transparently handle method calls and property access.
        """
        original_attr = getattr(self._wrapped_instance, attr)
        if callable(original_attr):

            def hooked(*args, **kwargs):
                # Before calling the original method
                self._logger.log(
                    "LM_API_CALL",
                    f'{{"type":"input","class_method":"{attr}","args":{json.dumps(str(args))}, "kwargs":{json.dumps(str(kwargs))}}}, "model": "{self.model_name}", "model_class": "{self._wrapped_instance.__class__.__name__}", "model_class_str": "{str(self._wrapped_instance)}"}}',
                )

                # check if openai model used,
                # then add token_tracking
                if "openai" in self._wrapped_instance.__class__.__name__.lower():
                    get_callback = get_openai_callback
                else:
                    get_callback = None

                if get_callback is not None:
                    with get_callback() as callback:
                        # Call the original method
                        result = original_attr(*args, **kwargs)
                else:
                    result = original_attr(*args, **kwargs)

                if get_callback is not None:
                    token_usage = json.dumps(
                        {
                            "str": json.dumps(str(callback)),
                            "cost": callback.total_cost,
                            "tokens": callback.total_tokens,
                            "prompt_tokens": callback.prompt_tokens,
                            "completion_tokens": callback.completion_tokens,
                        }
                    )
                else:
                    token_usage = '"(not available)"'

                # After calling the original method
                self._logger.log(
                    "LM_API_CALL",
                    f'{{"type":"return_value","class_method":"{attr}", "value": {json.dumps(str(result))}, "model": "{self.model_name}", "model_class": "{self._wrapped_instance.__class__.__name__}", "model_class_str": {json.dumps(str(self._wrapped_instance))}, "token_usage": {token_usage}}}',
                )
                return result

            return hooked
        else:
            return original_attr


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
        return LogWrapper(
            ChatOpenAI(
                model=name.split("/")[1],
                max_tokens=max_tokens,
                temperature=temp,
                model_kwargs=model_kwargs,
            )
        )
    if name.startswith("anthropic"):
        return LogWrapper(
            ChatAnthropic(
                model=name.split("/")[1],
                max_tokens=max_tokens,
                temperature=temp,
                model_kwargs=model_kwargs,
            )
        )
    if name.startswith("openrouter"):
        # Custom OpenRouter headers
        # https://openrouter.ai/docs/api-reference/overview#headers
        custom_headers = {
            "X-Title": "ICAI",
            # We need to set HTTP-Referer in addition to X-Title since otherwise openrouter does
            # not show an App name in the activity overview. If we set both, it shows the X-Title
            # and links to HTTP-Referer.
            "HTTP-Referer": "https://github.com/rdnfn/icai",
        }

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

        return LogWrapper(
            ChatOpenAI(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temp,
                openai_api_key=openrouter_api_key,  # Use the OpenRouter API key
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers=custom_headers,
                model_kwargs=model_kwargs,
            )
        )


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
