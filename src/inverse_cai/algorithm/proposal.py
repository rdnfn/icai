import ast
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm
from loguru import logger

import inverse_cai.models
from inverse_cai.data.utils import get_preferred_text, get_rejected_text
from inverse_cai.experiment.config import ExpConfig
import inverse_cai.algorithm.utils


def generate_principles_from_feedback(
    feedback: pd.DataFrame,
    num_principles_per_sampling_step,
    model_name: str,
    config: ExpConfig,
    num_rankings_per_sampling_step: int = 1,
    max_concurrent_tasks: int = 10,
) -> list:
    """
    Generate principles from feedback.

    Args:
        feedback: feedback data as a pandas DataFrame.
        num_principles_per_sampling_step: The number of principles to
            generate per sampling step.
        model_name: The name of the model to use.
        config: The experiment configuration.
        num_rankings_per_sampling_step: The number of rankings to use per
            principle sampling step. Only implemented for
            num_rankings_per_sampling_step=1 at the moment.
        max_concurrent_tasks: Maximum number of concurrent tasks to run.

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

    async def _async_generate_principles():
        # initialize the principles column
        feedback["principles"] = None
        feedback["prompt_principles"] = None

        if num_rankings_per_sampling_step == 1:
            # Create semaphore for controlling concurrency
            semaphore = asyncio.Semaphore(max_concurrent_tasks)

            async def process_row(
                index, row, num_principles_per_sampling_step, model_name, config
            ):
                async with semaphore:
                    principles, prompt_principles = (
                        await generate_principles_from_single_ranking(
                            preferred_text=get_preferred_text(row),
                            rejected_text=get_rejected_text(row),
                            num_principles=num_principles_per_sampling_step,
                            model_name=model_name,
                            config=config,
                        )
                    )
                    return index, principles, prompt_principles

            # create async tasks for parallel processing
            tasks = [
                process_row(
                    index, row, num_principles_per_sampling_step, model_name, config
                )
                for index, row in feedback.iterrows()
            ]

            # execute all tasks concurrently
            results = await tqdm.gather(*tasks)

            # update the feedback DataFrame
            for index, principles, prompt_principles in results:
                feedback.at[index, "principles"] = principles
                feedback.at[index, "prompt_principles"] = prompt_principles

        elif num_rankings_per_sampling_step > 1:
            raise NotImplementedError

        # get list of all principles (note that in results
        # principles are lists of principles)
        principles = [
            principle
            for _, principle_list, _ in results
            for principle in principle_list
        ]

        prompt_principles = [
            principle
            for _, _, prompt_principle_list in results
            for principle in prompt_principle_list
        ]

        logger.info(
            f"Generated {len(principles)} principles, {len(prompt_principles)} prompt principles (expected {overall_num_principles})"
        )

        return feedback, principles, prompt_principles

    # Run the async function
    return asyncio.run(_async_generate_principles())


async def generate_principles_from_single_ranking(
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
    prompt_principles: list = []

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
        principle_output = (await model.ainvoke(messages)).content

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

    for prompt in config.alg_prompts.prompt_generator_prompts:
        messages = inverse_cai.algorithm.utils.parse_prompt(
            prompt_str=prompt,
            prompt_kwargs=dict(
                preferred_sample=preferred_text,
                rejected_sample=rejected_text,
                prompt=inverse_cai.algorithm.utils.get_prompt_from_two_samples(
                    sample_a=preferred_text,
                    sample_b=rejected_text,
                ),
                num_principles=num_principles,
            ),
        )

        # generate principles
        principle_output = (await model.ainvoke(messages)).content

        # parse the principles
        try:
            principle_output = clean_principle_str(principle_output)
            parsed_output = ast.literal_eval(principle_output)["features"]
            prompt_principles += parsed_output
            if len(parsed_output) < num_principles:
                logger.warning(
                    f"Generated fewer ({len(parsed_output)}) principles "
                    f"than expected ({num_principles})"
                )
        except Exception as e:
            logger.error(f"Failed to parse principles: {principle_output}")
            logger.error(e)

    return principles, prompt_principles


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
