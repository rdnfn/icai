"""Generate Chatbot data for testing the inverse CAI algorithm."""

import random
import tqdm
import pandas as pd
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

import inverse_cai.models

DIVERSITY_LIST = [
    "cup",
    "wallet",
    "shoe",
    "camera",
    "umbrella",
    "phone",
    "mouse",
    "watch",
    "pillow",
    "chair",
    "dog",
    "pen",
    "lamp",
    "bag",
    "bottle",
    "clock",
    "sunglasses",
    "book",
    "keyboard",
    "plant",
]


def generate_data_set(
    num_comparisons: int = 10,
    generation_prompt: str = None,
    principles: list = None,
    diversity_list: list = None,
    save_path: str = "./data.csv",
    model_name: str = "openai/gpt-3.5-turbo",
):
    """Generate a synthetic data set to test the inverse CAI algorithm.

    Args:
        num_comparisons (int, optional): Number of comparisons to generate. Defaults to 10.
        generation_prompt (str, optional): The prompt to generate the data. Defaults to None.
        principles (list, optional): The principles to test. Defaults to None.
        diversity_list (list, optional): The list of objects to use
            in the generation prompt. Defaults to None.
        save_path (str, optional): The path to save the data.
            Defaults to "./data.csv".
        model_name (str, optional): The language model to use for
            generation. Defaults to "openai/gpt-3.5-turbo".
    """
    if diversity_list is None:
        diversity_list = DIVERSITY_LIST

    if num_comparisons % len(principles) != 0:
        logger.warning(
            "Number of comparisons is not divisible by number of principles. "
            f"Will create {(num_comparisons // len(principles))*len(principles)} comparisons instead of {num_comparisons}."
        )

    # generate input
    if generation_prompt is None:
        generation_prompt = "Create a short dialog of a user enquiring an chatbot assistant about {object} and an assistant answering. Keep it short, max one response each, max 20 words each."

    preferred_prompt = "Make sure that the assistant's answer would be choosen/preferred according to the following rule: {principle}."
    reject_prompt = "Make sure that the assistants answer would NEVER be chosen/preferred according to the following rule: {principle}\n\nIt is very important that the assistant's answer is CLEARLY rejected according to the rule. Be very explicit!"

    df = pd.DataFrame(
        columns=["preferred_text", "rejected_text", "tie", "ground_truth_principle"]
    )

    model = inverse_cai.models.get_model(model_name)

    for _ in tqdm.tqdm(range(num_comparisons // len(principles))):
        for principle in principles:
            diversity_object_a = random.choice(diversity_list)
            pref_complete_prompt = generation_prompt.format(
                object=diversity_object_a
            ) + preferred_prompt.format(principle=principle)

            diversity_object_b = random.choice(diversity_list)
            reject_complete_prompt = generation_prompt.format(
                object=diversity_object_b
            ) + reject_prompt.format(principle=principle)

            print(pref_complete_prompt)
            print(reject_complete_prompt)
            preferred_text = model.invoke(
                [HumanMessage(content=pref_complete_prompt)]
            ).content
            rejected_text = model.invoke(
                [HumanMessage(content=reject_complete_prompt)]
            ).content

            new_row = pd.DataFrame(
                {
                    "preferred_text": preferred_text,
                    "rejected_text": rejected_text,
                    "tie": False,
                    "ground_truth_principle": principle,
                },
                index=[0],
            )

            df = pd.concat(
                [df, new_row],
                ignore_index=True,
            )

    df.to_csv(save_path, index=False)

    return df
