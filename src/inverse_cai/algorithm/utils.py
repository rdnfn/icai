import shutil
from pathlib import Path
from loguru import logger
from typing import Awaitable
import openai
import backoff
import logging
import re


def parse_prompt(
    prompt_str: str, prompt_kwargs, prompt_optional_kwargs={}
) -> list[dict]:
    """Parse prompt str to list of messages."""

    # check kwargs in prompt_str and log warnings for unused keys
    for key in prompt_kwargs:
        if f"{{{key}}}" not in prompt_str:
            logger.error(f"Key '{key}' not found in prompt_str. Check your prompts.")

    # It's ok if optional kwargs aren't present in the prompt
    prompt_kwargs |= prompt_optional_kwargs

    # check prompt_str keys are in kwargs and log warnings for invalid keys
    for m in re.finditer(r"{([^{} ]+)}", prompt_str):
        key = m.group(1)
        if key not in prompt_kwargs:
            logger.error(f"Key '{key}' is not a valid key. Check your prompts.")
            prompt_str = prompt_str.replace(f"{{{key}}}", "")

    messages = ae_prompt_to_chatml(prompt_str)

    # add values to individual messages AFTER prompt to chatml
    # note: this is necessary if prompt_kwargs contain chatml syntax
    # that we do not want to parse as chatml
    for message in messages:
        if "content" in message:
            content = message["content"]
            # Filter kwargs to only include keys that exist in this specific message's content
            filtered_kwargs = {
                k: v for k, v in prompt_kwargs.items() if f"{{{k}}}" in content
            }
            message["content"] = content.format(**filtered_kwargs)

    return messages


prompt_warned = False


def get_prompt_from_row(row) -> str:
    """
    Get the prompt of a row
    """

    if "prompt" in row:
        return row["prompt"]
    else:
        return get_prompt_from_two_samples(row["text_a"], row["text_b"])


def get_prompt_from_two_samples(text_a, text_b) -> str:
    """
    Extract prompt from two samples

    In some datasets, the prompt is not available in the data but rather
    included in the text_a and text_b columns.
    """

    prompt_a = (
        text_a.split("Instruction:\n")[-1]
        .split("Response:\n")[0]
        .split("Assistant:\n")[0]
    )
    prompt_b = (
        text_b.split("Instruction:\n")[-1]
        .split("Response:\n")[0]
        .split("Assistant:\n")[0]
    )
    if prompt_a != prompt_b:
        global prompt_warned
        if not prompt_warned:
            # TODO: there must be a neater way to do this
            logger.warning(
                "ICAI doesn't know how to get prompts from this data: "
                'there is no "prompt" column and it couldn\'t be figured '
                "out from text_a and text_b"
            )
            prompt_warned = True
        return ""

    return prompt_a.strip()


def copy_cache(source_results_path: Path, target_results_path: Path):
    """Copy over cache from source results dir to target results dir.

    Args:
        source_results_path: Path to source results directory (e.g YYYY_MM_DD_HH_MM_SS/results/)
        target_results_path: Path to target results directory (similar as above)
    """
    # copy over prior cache file/directory
    shutil.copytree(
        source_results_path / "cache", target_results_path / "cache", dirs_exist_ok=True
    )
    logger.info(f"Copied over prior cache from '{source_results_path}'")


# These errors seem to come from multiple parts of the stack, and
# probably differ depending on the provider, etc, so this is awkward...
def _get_http_code(err):
    if isinstance(err, openai.APIStatusError):
        return err.status_code

    if len(err.args) > 0 and isinstance(err.args[0], dict) and "code" in err.args[0]:
        return err.args[0]["code"]

    try:
        return int(re.sub(r".*Error code: (\d*).*", r"\1", str(err)))
    except ValueError:
        return None


def _get_error_message(err):
    if (
        isinstance(err, openai.APIStatusError)
        and isinstance(err.body, dict)
        and "message" in err.body
    ):
        return err.body["message"]

    if len(err.args) > 0 and isinstance(err.args[0], dict) and "message" in err.args[0]:
        return err.args[0]["messag"]

    return str(err)


def _fatal_model_error(err):
    code = _get_http_code(err)
    if code:
        logger.info(f"Got HTTP error {code}")

    if code in (403,):
        # OpenAI returns this when it moderates
        logger.warning(f"403 Forbidden: {_get_error_message(err)}")
        return False
    elif code in (408, 420, 429, 444, 498, 499):
        # 4xx client errors
        # see https://http.cat/<code>
        # some of these are very obscure, but include all retryable ones just in case
        return False
    elif code // 100 == 5:
        # 5xx server errors
        return False
    else:
        return True


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=10,
    giveup=_fatal_model_error,
)
async def run_with_http_retries(fn, *args, **kwargs):
    return await fn(*args, **kwargs)


# The following two functions (string_to_dict and prompt_to_chatml) are taken from alpaca_eval.utils
# Replicated here to reduce dependency on alpaca_eval
# Licensed under Apache License 2.0 (https://github.com/tatsu-lab/alpaca_eval/blob/cd543a149df89434d8a54582c0151c0b945c3d20/LICENSE)
# Copyright 2023 Yann Dubois and Xuechen Li and Rohan Taori and Tianyi Zhang and Ishaan Gulrajani


def ae_string_to_dict(to_convert):
    r"""Converts a string with equal signs to dictionary. E.g.
    >>> _string_to_dict(" name=user university=stanford")
    {'name': 'user', 'university': 'stanford'}
    """
    return {
        s.split("=", 1)[0]: s.split("=", 1)[1]
        for s in to_convert.split(" ")
        if len(s) > 0
    }


def ae_prompt_to_chatml(
    prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"
):
    r"""Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = (
    ... "<|im_start|>system\n"
    ... "You are a helpful assistant.\n<|im_end|>\n"
    ... "<|im_start|>system name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\n"
    ... "Who's there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    ... )
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'content': 'You are a helpful assistant.', 'role': 'system'},
      {'content': 'Knock knock.', 'role': 'system', 'name': 'example_user'},
      {'content': "Who's there?", 'role': 'system', 'name': 'example_assistant'},
      {'content': 'Orange.', 'role': 'user'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = ae_string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message


## End of AE functions
