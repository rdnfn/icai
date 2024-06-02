from loguru import logger
import alpaca_eval.utils


def parse_prompt(prompt_str: str, prompt_kwargs) -> list[dict]:
    """Parse prompt str to list of messages."""

    # check kwargs in prompt_str
    for key in prompt_kwargs:
        if f"{{{key}}}" not in prompt_str:
            logger.error(f"Key '{key}' not found in prompt_str. Check your prompts.")

    formatted_prompt = prompt_str.format(**prompt_kwargs)
    messages = alpaca_eval.utils.prompt_to_chatml(formatted_prompt)

    return messages
