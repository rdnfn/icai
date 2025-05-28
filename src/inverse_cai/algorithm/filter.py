"""Tools to filter principles according to votes."""

from typing import Optional


def filter_according_to_votes(
    combined_votes: dict,
    require_majority_true: bool,
    require_majority_relevant: bool,
    require_majority_valid: bool,
    require_minimum_relevance: Optional[float],
    order_by: str,
    max_principles: int | None,
) -> list:
    """
    Filter principles according to votes.

    Args:
        combined_votes (dict): The votes for each principle.
        require_majority_true (bool): Require a majority of votes
            to be true.
        require_majority_relevant (bool): Require a majority of
            votes to be relevant.
        require_majority_valid (bool): Require a majority of votes
            to be valid.
        require_minimum_relevance (Optional[float]): Require a
            minimum relevance. Minimum amount of votes "for" the principle,
            so that the principle is kept.

    Returns:
        list: The principles to keep.
    """
    principle_keys_to_keep = []
    for principle, votes in combined_votes.items():
        keep_principle = True
        (
            votes_for,
            votes_against,
            votes_abstain,
            votes_invalid,
            votes_both,
            votes_neither,
        ) = (
            votes["for"],
            votes["against"],
            votes["abstain"],
            votes["invalid"],
            votes["both"],
            votes["neither"],
        )
        total_votes = (
            votes_for
            + votes_against
            + votes_abstain
            + votes_invalid
            + votes_both
            + votes_neither
        )

        if require_majority_true:
            if votes_for <= votes_against:
                keep_principle = False

        if require_majority_relevant:
            if votes_for <= votes_abstain:
                keep_principle = False

        if require_minimum_relevance:
            if votes_for < total_votes * require_minimum_relevance:
                keep_principle = False

        if require_majority_valid:
            if votes_invalid >= votes_for + votes_against + votes_abstain:
                keep_principle = False

        if keep_principle:
            principle_keys_to_keep.append(principle)

    if order_by == "for":
        principle_keys_to_keep = sorted(
            principle_keys_to_keep,
            key=lambda x: combined_votes[x]["for"],
            reverse=True,
        )
    elif order_by == "for_minus_against":
        principle_keys_to_keep = sorted(
            principle_keys_to_keep,
            key=lambda x: combined_votes[x]["for"] - combined_votes[x]["against"],
            reverse=True,
        )
    else:
        raise ValueError(f"Invalid order_by parameter: {order_by}")

    if max_principles:
        principle_keys_to_keep = principle_keys_to_keep[:max_principles]

    return principle_keys_to_keep
