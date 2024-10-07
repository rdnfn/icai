import ast
import pandas as pd
import tqdm
from loguru import logger

import inverse_cai as icai
from inverse_cai.data.utils import get_preferred_text, get_rejected_text
from inverse_cai.experiment.config import ExpConfig
import inverse_cai.algorithm.utils


def generate_principles_from_feedback(
    feedback: pd.DataFrame,
    num_principles_per_ranking,
    model_name: str,
    config: ExpConfig,
) -> list:
    """
    Generate principles from feedback.

    Args:
        feedback: feedback data as a pandas DataFrame.
        num_principles_per_ranking: The number of principles to generate per ranking.

    Returns:
        A list of principles.
    """
    logger.info("Generating principles from feedback")
    logger.info(f"Number of rankings: {len(feedback)}")
    logger.info(
        "Number of different prompts used for generation: "
        f"{len(config.alg_prompts.generator_prompts)}"
    )
    logger.info(
        f"Number of principles per ranking per prompt: {num_principles_per_ranking}"
    )
    overall_num_principles = (
        len(feedback)
        * len(config.alg_prompts.generator_prompts)
        * num_principles_per_ranking
    )
    logger.info(
        f"Will overall generate {overall_num_principles} principles from the feedback"
    )

    if num_principles_per_ranking == 1:
        logger.warning(
            (
                "Generating a single principle per sample currently "
                "not well supported. The generation prompt likely refers to "
                "multiple principles by default. "
                "Consider changing the generation prompt "
                "to work well in this scenario via the config."
            )
        )

    # initialize the principles column
    feedback["principles"] = None

    for index, row in tqdm.tqdm(feedback.iterrows(), total=len(feedback)):
        principles = generate_principles_from_single_ranking(
            preferred_text=get_preferred_text(row),
            rejected_text=get_rejected_text(row),
            num_principles=num_principles_per_ranking,
            model_name=model_name,
            config=config,
        )
        # note that principles here is a list of strings
        feedback.at[index, "principles"] = principles

    return feedback


def generate_principles_from_single_ranking(
    preferred_text: str,
    rejected_text: str,
    num_principles,
    model_name: str,
    config: ExpConfig,
) -> list:
    """
    Generate principles from a single ranking.

    Args:
        preferred_text: The preferred text.
        rejected_text: The rejected text.
        num_principles: The number of principles to generate.

    Returns:
        A list of principles.
    """
    assert num_principles > 0, "Number of principles must be greater than 0"

    # get the model
    model = icai.models.get_model(model_name)
    principles: list = []

    for prompt in config.alg_prompts.generator_prompts:
        messages = inverse_cai.algorithm.utils.parse_prompt(
            prompt_str=prompt,
            prompt_kwargs=dict(
                preferred_sample=preferred_text,
                rejected_sample=rejected_text,
                num_principles=num_principles,
            ),
        )

        # generate principles
        principle_output = model.invoke(messages).content

        # parse the principles
        try:
            principle_output = clean_principle_str(principle_output)
            parsed_output = ast.literal_eval(principle_output)["principles"]
            principles += parsed_output
            if len(parsed_output) < num_principles:
                logger.warning(
                    f"Generated fewer ({len(parsed_output)}) principles "
                    f"than expected ({num_principles})"
                )
        except Exception as e:
            logger.error(f"Failed to parse principles: {principle_output}")
            logger.error(e)

    return principles


def clean_principle_str(principle_str: str) -> str:
    """
    Clean a principle string.

    Especially necessary for GPT-4o models that
    like to add markdown code markers.

    Args:
        principle_str: The principle string to clean.

    Returns:
        The cleaned principle string.
    """
    if principle_str.startswith("```"):
        principle_str = principle_str[3:]
    if principle_str.startswith("json"):
        principle_str = principle_str[4:]
    if principle_str.endswith("\n```"):
        principle_str = principle_str[:-3]
    principle_str = principle_str.strip()
    return principle_str
