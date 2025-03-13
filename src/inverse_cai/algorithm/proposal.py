import ast
from joblib import Parallel, delayed
import pandas as pd
import tqdm
from loguru import logger
import numpy as np

import inverse_cai.models
from inverse_cai.data.utils import get_preferred_text, get_rejected_text
from inverse_cai.experiment.config import ExpConfig
import inverse_cai.algorithm.utils


def generate_principles_from_feedback(
    feedback: pd.DataFrame,
    num_principles_per_sampling_step,
    model_name: str,
    config: ExpConfig,
    num_rankings_per_sampling_step: int,
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
        "Number of different prompt templates used for generation: "
        f"{len(config.alg_prompts.generator_prompts)}"
    )
    logger.info(
        f"Number of rankings per sampling step: {num_rankings_per_sampling_step}"
    )
    logger.info(
        f"Number of principles per sampling step per prompt: {num_principles_per_sampling_step}"
    )
    overall_num_principles = (
        -(len(feedback) // -num_rankings_per_sampling_step)  # ceiling division
        * len(config.alg_prompts.generator_prompts)
        * num_principles_per_sampling_step
    )
    logger.info(
        f"Will overall generate {overall_num_principles} principles from the feedback"
    )

    if num_principles_per_sampling_step == 1:
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

    if num_rankings_per_sampling_step == 1:

        def process_row(
            index, row, num_principles_per_sampling_step, model_name, config
        ):
            principles = generate_principles_from_single_ranking(
                preferred_text=get_preferred_text(row),
                rejected_text=get_rejected_text(row),
                num_principles=num_principles_per_sampling_step,
                model_name=model_name,
                config=config,
            )
            return index, principles

        # parallelize the process
        results = Parallel(n_jobs=config.parallel_workers)(
            delayed(process_row)(
                index, row, num_principles_per_sampling_step, model_name, config
            )
            for index, row in tqdm.tqdm(feedback.iterrows(), total=feedback.shape[0])
        )
        # update the feedback DataFrame
        for index, principles in results:
            feedback.at[index, "principles"] = principles

    elif num_rankings_per_sampling_step > 1:

        # allocate group_ids, such that each row is assigned to a random equally
        # sized group
        n_groups = -(len(feedback) // -num_rankings_per_sampling_step)
        feedback["group_id"] = pd.qcut(
            np.random.permutation(len(feedback)), q=n_groups, labels=False
        )
        assert (
            feedback["group_id"].value_counts().max() <= num_rankings_per_sampling_step
        ), f"Number of rankings per group must be less than or equal to {num_rankings_per_sampling_step}, max is {feedback['group_id'].value_counts().max()}"
        assert (
            feedback["group_id"].unique().shape[0] == n_groups,
            f"Number of groups ({feedback['group_id'].unique().shape[0]}) must be equal to {n_groups}",
        )

        def process_multiple_rows(
            index, rows, num_principles_per_sampling_step, model_name, config
        ):
            principles = generate_principles_from_multiple_rankings(
                preferred_texts=[get_preferred_text(row) for _, row in rows.iterrows()],
                rejected_texts=[get_rejected_text(row) for _, row in rows.iterrows()],
                num_principles=num_principles_per_sampling_step,
                model_name=model_name,
                config=config,
            )
            return index, principles

        # parallelize the process
        results = Parallel(n_jobs=config.parallel_workers)(
            delayed(process_multiple_rows)(
                index, rows, num_principles_per_sampling_step, model_name, config
            )
            for index, rows in tqdm.tqdm(
                feedback.groupby("group_id"), total=feedback.shape[0]
            )
        )

        # update the feedback DataFrame, adding groups principles to each row in group
        for index, principles in results:
            # add principles to each row in the group
            feedback.loc[feedback["group_id"] == index]["principles"] = [
                principles
            ] * len(feedback[feedback["group_id"] == index])

    # get list of all principles (note that in results
    # principles are lists of principles)
    principles = [
        principle for _, principle_list in results for principle in principle_list
    ]

    logger.info(
        f"Generated {len(principles)} principles (expected {overall_num_principles})"
    )

    return feedback, principles


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
        model_name: The name of the model to use.
        config: The experiment configuration.

    Returns:
        A list of principles.
    """
    assert num_principles > 0, "Number of principles must be greater than 0"

    # get the model
    model = inverse_cai.models.get_model(model_name)
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


def generate_principles_from_multiple_rankings(
    preferred_texts: list[str],
    rejected_texts: list[str],
    num_principles: int,
    model_name: str,
    config: ExpConfig,
) -> list:
    """
    Generate principles from multiple rankings.

    Args:
        preferred_texts: List of preferred texts.
        rejected_texts: List of rejected texts.
        num_principles: The number of principles to generate.
        model_name: The name of the model to use.
        config: The experiment configuration.

    Returns:
        A list of principles.
    """
    assert num_principles > 0, "Number of principles must be greater than 0"
    assert len(preferred_texts) == len(
        rejected_texts
    ), "Number of preferred and rejected texts must match"
    assert len(preferred_texts) > 0, "At least one ranking must be provided"

    # get the model
    model = inverse_cai.models.get_model(model_name)
    principles: list = []

    rankings_str = "\n".join(
        [
            f"## Ranking {i+1}:\n### Preferred: {preferred_text}\n\n### Rejected: {rejected_text}\n\n-----\n\n"
            for i, (preferred_text, rejected_text) in enumerate(
                zip(preferred_texts, rejected_texts)
            )
        ]
    )

    for prompt in config.alg_prompts.generator_prompts:
        messages = inverse_cai.algorithm.utils.parse_prompt(
            prompt_str=prompt,
            prompt_kwargs=dict(
                rankings=rankings_str,
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
