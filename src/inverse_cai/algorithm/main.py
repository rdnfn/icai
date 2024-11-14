from loguru import logger
import pandas as pd
import random

from inverse_cai.algorithm.clustering import (
    get_cluster_summaries,
    print_clusters,
    cluster_principles,
)
from inverse_cai.algorithm.proposal import generate_principles_from_feedback
from inverse_cai.algorithm.voting import get_votes_for_principles
from inverse_cai.algorithm.filter import filter_according_to_votes
from inverse_cai.utils import save_to_json
from inverse_cai.experiment.config import ExpConfig
import inverse_cai.visualisation
import inverse_cai.experiment


def run(
    feedback: pd.DataFrame,
    save_path: str,
    num_principles_generated_per_ranking: int,
    num_rankings_per_sampling_step: int,
    num_clusters: int,
    random_clusters: bool,
    skip_voting: bool,
    require_majority_true: bool,
    require_majority_relevant: bool,
    require_majority_valid: bool,
    require_minimum_relevance: float,
    order_by: str,
    max_principles: int,
    ratio_of_max_principles_to_cluster_again: float,
    model_name: str,
    config: ExpConfig,
    load_path: str = None,
) -> dict:
    """
    Run the inverse CAI algorithm.

    Args:
        feedback (pd.DataFrame): The feedback data as a pandas DataFrame,
            follows standard format of ICAI.
        save_path (str): The path to save the results.
        num_principles_generated_per_ranking (int): The number of
            principles to generate per ranking.
        num_clusters (int): The number of clusters to generate.
        model_name (str): The model to use for generating principles.
        load_path (str): The path to load intermediary results from.

    Returns:
        dict: The results of the algorithm.
    """

    logger.info("Running the Inverse Constitutional AI algorithm")

    ### checking inputs
    # make sure data without ties
    assert not feedback["preferred_text"].str.contains("tie").any()

    ### STAGE 1: Generate principles from feedback
    logger.info("Stage 1: Generate principles from feedback")
    feedback = generate_principles_from_feedback(
        feedback,
        num_principles_generated_per_ranking,
        model_name=model_name,
        config=config,
        num_rankings_per_sampling_step=num_rankings_per_sampling_step,
    )
    feedback["principles"].to_csv(
        save_path / "010_principles_per_comparison.csv", index=True, index_label="index"
    )

    # flatten list of lists of principles into single list
    principles = [item for sublist in list(feedback["principles"]) for item in sublist]
    print("\n".join(principles))
    save_to_json(principles, save_path / "011_principles_list.json")

    ### STAGE 2: Cluster principles
    logger.info("Stage 2: Cluster principles")
    clusters = cluster_principles(
        principles,
        num_clusters=num_clusters,
        random_clusters=random_clusters,
    )
    save_to_json(clusters, save_path / "020_principle_clusters.json")

    summaries = get_cluster_summaries(
        clusters,
        model_name=model_name,
        sample_instead_of_rewrite=True,
        config=config,
    )
    print_clusters(clusters, summaries)
    save_to_json(summaries, save_path / "030_distilled_principles_per_cluster.json")

    ### STAGE 3: Get votes for principles
    logger.info("Stage 3: Get votes for principles")

    if not skip_voting:
        raw_votes, combined_votes = get_votes_for_principles(
            feedback_df=feedback,
            summaries=summaries,
            max_votes_in_single_prompt=config.s3_filter_max_votes_in_single_prompt,
            model_name=model_name,
            config=config,
        )

        raw_votes.to_csv(
            save_path / "040_votes_per_comparison.csv", index=True, index_label="index"
        )
        save_to_json(combined_votes, save_path / "041_votes_per_cluster.json")

        # visualise
        inverse_cai.visualisation.plot_approval_bars(
            categories=list(summaries.values()),
            votes=list(combined_votes.values()),
            path=save_path / "042_principle_approval_votes.png",
        )

        filtered_plinciple_keys = filter_according_to_votes(
            combined_votes=combined_votes,
            require_majority_true=require_majority_true,
            require_majority_relevant=require_majority_relevant,
            require_majority_valid=require_majority_valid,
            require_minimum_relevance=require_minimum_relevance,
            order_by=order_by,
            max_principles=(
                int(max_principles * ratio_of_max_principles_to_cluster_again)
                if max_principles is not None
                else None
            ),
        )

        filtered_principles = [summaries[key] for key in filtered_plinciple_keys]

        save_to_json(filtered_principles, save_path / "050_filtered_principles.json")

        if not max_principles or len(filtered_principles) <= max_principles:
            logger.warning(
                "Number of filtered principles is less or equal to max principles, "
                "or max principles is not set. "
                "Using all filtered principles. "
                "Skipping final clustering and subsampling step."
            )
            final_principles = filtered_principles
        else:
            logger.info(
                f"Final clustering and subsampling step. Going from {len(filtered_principles)} to {max_principles} principles."
            )
            filtered_clusters = cluster_principles(
                filtered_principles,
                num_clusters=max_principles,
                random_clusters=False,
            )

            # filtered summaries are first occuring principles in filter_principles for each cluster
            def find_first_in_second_list(list1, list2):
                # Create a set of indices from list2 based on elements in list1
                index_map = {element: list2.index(element) for element in list1}
                # Return the element in list1 with the minimum index in list2
                selection = min(list1, key=lambda x: index_map[x])
                logger.info(f"Selected {selection} out of {list1}.")
                return selection

            filtered_summaries = {
                key: find_first_in_second_list(
                    filtered_clusters[key], filtered_principles
                )
                for key in filtered_clusters
            }

            print_clusters(filtered_clusters, filtered_summaries)
            combined_clusters = {
                filtered_summaries[key]: filtered_clusters[key]
                for key in filtered_clusters
            }
            save_to_json(combined_clusters, save_path / "51_final_clusters.json")

            # ensure we retain original order set during filtering
            final_principles = [
                value
                for value in filtered_principles
                if value in filtered_summaries.values()
            ]

    else:
        logger.warning("Skipping voting stage")
        combined_votes = None
        filtered_principles = None

        # randomly sample from all principles instead of voting
        available_principles = list(summaries.values())
        final_principles = random.choices(available_principles, k=max_principles)

    # Generate constitution text from principles
    constitution = "\n".join(
        [f"{i+1}. " + principle for i, principle in enumerate(final_principles)]
    )

    logger.info(f"Constitution generated:\n\n{constitution}\n")
    save_to_json(constitution, save_path / "060_constitution.json")

    return_val = {
        "feedback": feedback,
        "clusters": clusters,
        "summaries": summaries,
        "combined_votes": combined_votes,
        "filtered_plinciples": filtered_principles,
        "final_principles": final_principles,
        "constitution": constitution,
    }

    logger.info("Inverse constitutional AI algorithm completed successfully.")

    return return_val
