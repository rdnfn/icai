import shutil
from pathlib import Path
from loguru import logger
from typing import Awaitable
import openai
import alpaca_eval.utils
import backoff
import logging
import re


def parse_prompt(prompt_str: str, prompt_kwargs, prompt_optional_kwargs) -> list[dict]:
    """Parse prompt str to list of messages."""

    # check kwargs in prompt_str and log warnings for unused keys
    for key in prompt_kwargs:
        if f"{{{key}}}" not in prompt_str:
            logger.error(f"Key '{key}' not found in prompt_str. Check your prompts.")

    # It's ok if optional kwargs aren't present in the prompt
    prompt_kwargs |= prompt_optional_kwargs

    messages = alpaca_eval.utils.prompt_to_chatml(prompt_str)

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

    In some datasets, the prompt is not available in the data but rather
    included in the text_a and text_b columns. This function attempts to
    extract the prompt from the two samples.
    """
    if "prompt" in row:
        return row["prompt"]

    prompt_a = (
        row["text_a"].split("Instruction:\n")[-1]
        .split("Response:\n")[0]
        .split("Assistant:\n")[0]
    )
    prompt_b = (
        row["text_b"].split("Instruction:\n")[-1]
        .split("Response:\n")[0]
        .split("Assistant:\n")[0]
    )
    if prompt_a != prompt_b:
        global prompt_warned
        if not prompt_warned:
            # TODO: there's probably a neater way to do this
            logger.warning(
                "ICAI doesn't know how to get prompts from this data: "
                "there is no \"prompt\" column and it couldn't be figured "
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
    if isinstance(err, openai.APIStatusError) and isinstance(err.body, dict) and "message" in err.body:
        return err.body["message"]

    if len(err.args) > 0 and isinstance(err.args[0], dict) and "message" in err.args[0]:
        return err.args[0]["messag"]

    return str(error)


def _fatal_model_error(err):
    code = _get_http_code(err)
    if code: print(f"Got HTTP error {code}") # debug

    if code in (403,):
        # OpenAI returns this when it moderates
        logger.warning(f'403 Forbidden: {_get_error_message(err)}')
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
