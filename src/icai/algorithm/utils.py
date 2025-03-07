from loguru import logger
import alpaca_eval.utils


def parse_prompt(prompt_str: str, prompt_kwargs) -> list[dict]:
    """Parse prompt str to list of messages."""

    # check kwargs in prompt_str and log warnings for unused keys
    for key in prompt_kwargs:
        if f"{{{key}}}" not in prompt_str:
            logger.error(f"Key '{key}' not found in prompt_str. Check your prompts.")

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
