from typing import Any, Sequence, Mapping, Union
from functools import partial
import json
from filelock import FileLock

import asyncio
import langchain_core.messages.base
import numpy as np
import pickle
import os
import os.path
import hashlib

from functools import lru_cache
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


def serializable(obj):
    if isinstance(obj, Union[str, int, float, bool, type(None)]):
        return obj
    elif isinstance(obj, Sequence):
        return tuple(serializable(x) for x in obj)
    elif isinstance(obj, Mapping):
        return {k: serializable(v) for k, v in obj.items()}
    elif isinstance(obj, langchain_core.messages.base.BaseMessage):
        return serializable(langchain_core.messages.base.message_to_dict(obj))
    else:
        logger.warning(
            f"{obj} ({type(obj).__name__}) is not serializable, converting to string for cache key"
        )
        return str(obj)


def hash_obj(obj):
    m = hashlib.sha256()
    m.update(json.dumps(serializable(obj)).encode())
    return m.hexdigest()


def partial_hash(obj):
    """
    This function is for humans, to make it easier to inspect the differences
    between objects
    """
    if isinstance(obj, str):
        if len(obj) < 20:
            return obj
        else:
            return f"hash:{hash_obj(obj)[:8]}"
    elif isinstance(obj, Sequence):
        return tuple(partial_hash(x) for x in obj)
    elif isinstance(obj, Mapping):
        return {k: partial_hash(v) for k, v in obj.items()}
    else:
        return f"hash:{hash_obj(obj)[:8]}"


class CachedObject:
    def __init__(self, obj, cache_dir, cached_funcs, seed=0):
        self.obj = obj
        self.cache_dir = cache_dir
        self.cached_funcs = cached_funcs
        self.seed = seed

    @property
    def cache_seed_path(self):
        return f"{self.cache_dir}/{self.seed}"

    def cache_key_path(self, key):
        return f"{self.cache_seed_path}/{key}.pkl"

    @lru_cache(maxsize=128)
    def get_from_cache(self, key):
        if os.path.isfile(self.cache_key_path(key)):
            with open(self.cache_key_path(key), "rb") as cache_file:
                return pickle.load(cache_file)

        return None

    def save_to_cache(self, key, value):
        try:
            with open(self.cache_key_path(key), "wb") as cache_file:
                pickle.dump(value, cache_file)
        except FileNotFoundError:
            # using try-except means os.makedirs will only be run once at most
            os.makedirs(self.cache_seed_path, exist_ok=True)
            return self.save_to_cache(key, value)

    @staticmethod
    async def _arun(coroutine):
        await coroutine
        return coroutine

    async def async_cache_run(self, func: str, *args, **kwargs):
        h = hash_obj((func, args, kwargs))
        result = self.get_from_cache(h)

        if result is None:
            logger.debug(f"cache {self.seed} miss ({h[:8]})")
            result = getattr(self.obj, func)(*args, **kwargs)

            if asyncio.iscoroutine(result):
                result = await result

            self.save_to_cache(h, result)
        else:
            logger.debug(f"cache {self.seed} hit ({h[:8]})")

        return result

    def cache_run(self, *args, **kwargs):
        return asyncio.run(self.async_cache_run(*args, **kwargs))

    def __getattr__(self, attr):
        if attr in self.cached_funcs:
            if asyncio.iscoroutinefunction(getattr(self.obj, attr)):
                return partial(self.async_cache_run, attr)
            else:
                return partial(self.cache_run, attr)
        else:
            return self.obj.__getattr__(attr)


def get_model(
    name: str,
    temp: float = 0.0,
    enable_logprobs: bool = False,
    max_tokens: int = 1000,
    cache: bool = True,
    cache_seed: int = 0,
) -> Any:
    """Get a language model instance.

    Args:
        name: Model name with provider prefix (e.g. "openai/gpt-4o-2024-05-13")
        temp: Temperature for generation (default: 0.0)
        enable_logprobs: Whether to enable logprobs for token probabilities (default: False)
        max_tokens: Maximum tokens to generate (default: 1000)
        cache: enable model cache

    Returns:
        (possibly CachedObject-wrapped) LogWrapper-wrapped language model instance
    """
    if enable_logprobs:
        model_kwargs = {"logprobs": True, "top_logprobs": 10}
    else:
        model_kwargs = {}

    if name.startswith("openai"):
        model = ChatOpenAI(
            model=name.split("/")[1],
            max_tokens=max_tokens,
            temperature=temp,
            model_kwargs=model_kwargs,
        )

    elif name.startswith("anthropic"):
        model = ChatAnthropic(
            model=name.split("/")[1],
            max_tokens=max_tokens,
            temperature=temp,
            model_kwargs=model_kwargs,
        )

    elif name.startswith("openrouter"):
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

        model = ChatOpenAI(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temp,
            openai_api_key=openrouter_api_key,  # Use the OpenRouter API key
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers=OPENROUTER_HEADERS,
            model_kwargs=model_kwargs,
        )
    else:
        raise ValueError(f"{name} is not a recognised model name")

    if cache:
        model = CachedObject(
            model,
            cache_dir=f"exp/cache/models/{name}_{temp}_{enable_logprobs}_{max_tokens}",
            cached_funcs=("invoke", "ainvoke"),
            seed=cache_seed,
        )

    return model


def get_embeddings_model(
    self,
    name: str,
    cache: bool = True,
    cache_seed: int = 0,
):
    openrouter = name.startswith("openrouter")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openrouter and not openai_api_key:
        raise ValueError(
            "OpenRouter doesn't support embedding models, OPENAI_API_KEY still required"
        )

    model = OpenAIEmbeddings()

    if cache:
        model = CachedObject(
            model,
            cache_dir=f"exp/cache/models/{name}",
            cached_funcs=("embed_documents",),
            seed=cache_seed,
        )

    return model


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
