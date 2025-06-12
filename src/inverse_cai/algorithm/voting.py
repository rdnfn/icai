import ast
import asyncio
import random
import time
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm.asyncio import tqdm
from typing import Literal

from inverse_cai.data.utils import get_preferred_text, get_rejected_text
from inverse_cai.experiment.config import ExpConfig
import inverse_cai.algorithm.utils
import inverse_cai.models
from inverse_cai.algorithm.cache import VoteCache, get_vote_hash


def get_votes_for_principles(
    feedback_df: pd.DataFrame,
    max_votes_in_single_prompt: int,
    summaries: dict,
    config: ExpConfig,
    model_name: str,
    cache_path: Path,
    max_concurrent_tasks: int,
    num_seeds: int,
    voting_method_cross_seed: Literal["majority", "unanimous"],
    prompt_principles=False,
) -> tuple[pd.DataFrame, dict]:
    """Get votes for principles.

    Distributed over multiple passes if necessary.

    Args:
        feedback_df: DataFrame of feedback
        max_votes_in_single_prompt: Maximum number of votes in a single prompt
        summaries: Dictionary of summaries
        config: Configuration
        model_name: Name of the model to use
        cache_path: Path to cache (without .jsonl suffix)
        max_concurrent_tasks: Maximum number of concurrent tasks
        num_seeds: Number of seeds, i.e. how often to re-annotate the same data
        prompt_principles: Whether to vote for prompt principles
    """

    logger.info("Getting votes for principles")
    if prompt_principles:
        logger.info("Voting for prompt principles")
    else:
        logger.info("Voting for regular principles")

    if len(summaries) == 0:
        logger.warning("No principles to vote on, skipping voting.")
        return pd.Series(), {}

    num_passes = len(summaries) // max_votes_in_single_prompt + 1
    logger.info(f"Split voting into {num_passes} passes over entire dataset.")
    logger.info(
        f"Running up to {max_concurrent_tasks} LLM calls asynchronously at the same time."
    )

    hashed_votes_per_seed = {seed: [] for seed in range(1, num_seeds + 1)}

    for seed in range(1, num_seeds + 1):
        # Shuffle summary keys and split into chunks
        summary_keys = list(summaries.keys())
        random.seed(seed)
        random.shuffle(summary_keys)
        summaries_parts = [
            {k: summaries[k] for k in summary_keys[i : i + max_votes_in_single_prompt]}
            for i in range(0, len(summary_keys), max_votes_in_single_prompt)
        ]
        assert sum(len(part) for part in summaries_parts) == len(summaries), (
            f"Sum of lengths of summaries parts ({sum(len(part) for part in summaries_parts)}) "
            f"does not match length of summaries ({len(summaries)})"
            f"Full summaries: {summaries}\nFull summaries parts: {summaries_parts}"
            f"max votes in single prompt: {max_votes_in_single_prompt}"
        )
        logger.info(f"Running voting for seed {seed}/{num_seeds}.")
        for i, summary_part in enumerate(summaries_parts):
            logger.info(f"Starting pass {i+1}/{len(summaries_parts)}")
            cache_path_seed = Path(f"{cache_path}_seed_{seed}")
            votes = asyncio.run(
                run_pass_to_get_votes_for_principles(
                    feedback_df=feedback_df,
                    summaries=summary_part,
                    config=config,
                    model_name=model_name,
                    cache_path=cache_path_seed,
                    prompt_principles=prompt_principles,
                    max_concurrent_tasks=max_concurrent_tasks,
                )
            )
            hashed_votes_per_seed[seed].append(votes)

    logger.info(f"Post-processing votes")
    postprocess_start_time = time.time()

    # combine hashed votes across all passes into single dictionary
    for seed in hashed_votes_per_seed:
        hashed_votes_per_seed[seed] = {
            k: v for part in hashed_votes_per_seed[seed] for k, v in part.items()
        }

    def _get_cross_seed_votes(hashed_votes_per_seed, voting_method_cross_seed):
        """Combine votes per datapoint across seeds with voting method."""

        num_seeds = len(hashed_votes_per_seed)

        # get all vote hashes (even if failed on some seed)
        all_vote_hashes = set()
        for seed in hashed_votes_per_seed:
            all_vote_hashes.update(hashed_votes_per_seed[seed].keys())

        hashed_votes = {}
        for vote_hash in all_vote_hashes:

            # collect vote value counts
            value_counts = {}
            for seed in hashed_votes_per_seed:
                vote = hashed_votes_per_seed[seed][vote_hash]
                value_counts[vote] = value_counts.get(vote, 0) + 1

            # find if any vote value succeeded according to voting method
            for vote, count in value_counts.items():
                if voting_method_cross_seed == "majority":
                    if count > 0.5 * num_seeds:
                        hashed_votes[vote_hash] = vote
                        break
                elif voting_method_cross_seed == "unanimous":
                    if count == num_seeds:  # all votes agree
                        hashed_votes[vote_hash] = vote
                        break

            # if no vote succeeded according to voting method,
            # set to None (principle not applicable)
            if vote_hash not in hashed_votes:
                hashed_votes[vote_hash] = None

        return hashed_votes

    hashed_votes = _get_cross_seed_votes(
        hashed_votes_per_seed, voting_method_cross_seed
    )

    # Transform to standard vote representation used throughout ICAI codebase

    def _get_per_comparison_votes(row):
        """Returns a per-comparison vote containing all votes for all principles."""
        hash_per_principle = {}
        for key, principle in summaries.items():
            vote_hash = get_vote_hash(
                preferred=get_preferred_text(row),
                rejected=get_rejected_text(row),
                principle=principle,
                model_name=model_name,
            )
            hash_per_principle[key] = vote_hash

        return {key: hashed_votes[hash_per_principle[key]] for key in summaries.keys()}

    # ### First output: raw_votes ###
    # Raw votes take the following pd series form:
    # index,votes
    # 0,"{0: False, 1: True, 2: None, 3: None, 4: None, ...}"
    # Where each key is the index of the principle in the summaries.
    # Eventually, e.g. saved to 040_votes_per_comparison.csv (by algorithm.main.py).
    raw_votes = feedback_df.apply(_get_per_comparison_votes, axis=1)

    # ### Second output: combined_votes ###
    # Takes the following json form:
    # {
    # "0": {"for": 279, "against": 363, "abstain": 6, "invalid": 0, "both": 0, "neither": 0},
    # ...
    # }
    #
    # Here, the "0" key is the index of the principle in the summaries.
    # Eventually, e.g. saved to 041_votes_per_cluster.json (by algorithm.main.py).
    combined_votes = combine_votes(list(raw_votes), summaries)

    logger.info(
        f"Post-processing complete: took {time.time() - postprocess_start_time:.4f} seconds"
    )
    logger.info("Votes complete")

    return raw_votes, combined_votes


async def run_pass_to_get_votes_for_principles(
    feedback_df: pd.DataFrame,
    summaries: dict,
    config: ExpConfig,
    model_name: str,
    cache_path: Path,
    prompt_principles: bool,
    max_concurrent_tasks: int = 10,
) -> dict:
    """
    Given a dataframe of conversations, run voting with each proposed
    principle on each pairwise comparison. Single pass over dataset.
    """
    feedback_df = feedback_df.copy()
    feedback_df["votes"] = None

    vote_cache = VoteCache(cache_path)
    initial_cached_votes = vote_cache.get_cached_votes()

    # Create semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # Function to process each row
    async def process_row(
        index, row, summaries, model_name, config, initial_cached_votes
    ):
        async with semaphore:
            preferred = get_preferred_text(row)
            rejected = get_rejected_text(row)

            principles = list(summaries.values())
            hashes = {
                principle: get_vote_hash(
                    preferred=preferred,
                    rejected=rejected,
                    principle=principle,
                    model_name=model_name,
                )
                for principle in principles
            }

            all_hashes_in_cache = True
            for hash_str in hashes.values():
                if hash_str not in initial_cached_votes:
                    all_hashes_in_cache = False
                    break

            if all_hashes_in_cache:
                await asyncio.sleep(0.1)
                return {h: initial_cached_votes[h] for h in hashes.values()}

            vote = await get_preference_vote_for_single_text(
                prompt=inverse_cai.algorithm.utils.get_prompt_from_row(row),
                preferred_sample=preferred,
                rejected_sample=rejected,
                principles=principles,
                model_name=model_name,
                config=config,
                prompt_principles=prompt_principles,
            )

            # Update cache
            hashed_vote = {
                hash_str: vote.get(principle, "invalid")
                for principle, hash_str in hashes.items()
            }
            for hash_str, vote_value in hashed_vote.items():
                vote_cache.update_cache(hash_str, vote_value)

            return hashed_vote

    # Create async tasks for parallel processing
    tasks = [
        process_row(index, row, summaries, model_name, config, initial_cached_votes)
        for index, row in feedback_df.iterrows()
    ]

    # Execute all tasks concurrently with progress bar
    votes = await tqdm.gather(*tasks)

    # Combine votes into a single dictionary
    # Each entry is a single vote for a principle on a single comparison.
    full_votes = {k: v for part in votes for k, v in part.items()}

    return full_votes


async def get_preference_vote_for_single_text(
    prompt,
    preferred_sample,
    rejected_sample,
    principles,
    config: ExpConfig,
    model_name: str,
    prompt_principles: bool = False,
):
    """
    Given a dataframe of conversations, let the model votes according to each proposed principles.

    Model output is formatted as json format, for each principle.

    Note: preference-based voting require ast-based parsing here to ensure flipped
    votes can be corrected for right away.
    """

    flipped = random.choice([True, False])

    if flipped:
        sample_a, sample_b = rejected_sample, preferred_sample
    else:
        sample_a, sample_b = preferred_sample, rejected_sample

    numbered_principles = {i: v for i, v in enumerate(principles)}

    messages = inverse_cai.algorithm.utils.parse_prompt(
        prompt_str=(config.alg_prompts.prompt_voting_prompt if prompt_principles else config.alg_prompts.voting_prompt),
        prompt_kwargs=(
                dict(prompt=prompt)
            if prompt_principles else
                dict(
                    sample_a=sample_a,
                    sample_b=sample_b,
                )
        ) | dict(summaries=numbered_principles),
        prompt_optional_kwargs={},
    )

    model = inverse_cai.models.get_model(model_name)
    vote = (await inverse_cai.algorithm.utils.run_with_http_retries(model.ainvoke, messages)).content
    vote = parse_individual_pref_vote(
        vote,
        num_principles=len(principles),
        prompt_principles=prompt_principles,
    )

    # change back to original keys
    vote = {numbered_principles[k]: v for k, v in vote.items() if k in numbered_principles}

    if flipped:
        vote = {k: "A" if v == "B" else "B" if v == "A" else v for k, v in vote.items()}

    # translate votes to correct/incorrect/invalid
    updated_vote = {}
    for key, value in vote.items():
        if prompt_principles:
            if value in (True, False):
                updated_vote[key] = value
            else:
                updated_vote[key] = "invalid"
        else:
            if value == "A":
                updated_vote[key] = True
            elif value == "B":
                updated_vote[key] = False
            elif value is None:
                updated_vote[key] = None
            elif value in ("Both", "Neither"):
                updated_vote[key] = value
            else:
                updated_vote[key] = "invalid"

    return updated_vote


def parse_individual_pref_vote(vote, num_principles, prompt_principles=False):
    """
    Parse preference-based votes.

    Using each principle to make a preference decision.
    """
    try:
        vote_json = clean_vote_json(vote, num_principles)
        vote_dict = ast.literal_eval(vote_json)
    except Exception as e:
        vote_dict = {i: "invalid" for i in range(num_principles)}
        logger.error(f"Failed to parse vote: {vote}")
        logger.error(e)

    # make sure all keys are integers
    vote_dict = {int(k): v for k, v in vote_dict.items()}

    if len(vote_dict) != num_principles:
        logger.error(
            f"Vote length {len(vote_dict)} does not match number of principles {num_principles}"
        )

    if prompt_principles:
        valid = ["True", "False", "true", "false", True, False]
    else:
        valid = ["A", "B", "Both", "Neither", "None", None]
    for key, value in vote_dict.items():
        if value not in valid:
            logger.error(f"Vote value {value} is not in {valid}")
            vote_dict[key] = "invalid"

    return vote_dict


def combine_votes(votes: pd.Series, summaries: dict):
    """
    Combine list of votes into an overall result, for each principle.
    """
    vote_dict = {
        i: {"for": 0, "against": 0, "abstain": 0, "invalid": 0, "both": 0, "neither": 0}
        for i in summaries.keys()
    }
    for vote in votes:
        for j in summaries.keys():
            if j not in vote:
                logger.error(f"Principle {j} not found in vote")
                vote_dict[j]["invalid"] += 1
            elif vote[j] is True:
                vote_dict[j]["for"] += 1
            elif vote[j] is False:
                vote_dict[j]["against"] += 1
            elif vote[j] == "Both":
                vote_dict[j]["both"] += 1
            elif vote[j] == "Neither":
                vote_dict[j]["both"] += 1
            elif vote[j] is None:
                vote_dict[j]["abstain"] += 1
            else:
                vote_dict[j]["invalid"] += 1

    return vote_dict


def clean_vote_json(vote_json, summaries_len):
    """
    Clean the vote json.
    """
    vote_json = (
        vote_json.replace("\n", "")
        .replace(" ", "")
        .replace("true", "True")
        .replace("false", "False")
        .replace("both", "Both")
        .replace("neither", "Neither")
        .replace("null", "None")
    )
    # replace string keys with int keys
    for i in list(range(summaries_len + 10)) + ["True", "False", "None"]:
        vote_json = vote_json.replace(f'"{i}"', f"{i}")
        vote_json = vote_json.replace(f"'{i}'", f"{i}")

    for letter in ["A", "B", "Both", "Neither"]:
        vote_json = vote_json.replace(f"'{letter}'", f'"{letter}"')

    return vote_json
