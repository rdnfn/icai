"""Compute metrics """

import pandas as pd


def compute_metrics(votes_df: pd.DataFrame) -> dict:

    principles = votes_df["principle"].unique()
    num_pairs = len(votes_df["comparison_id"].unique())

    def get_agreement(principle: str) -> float:
        principle_votes = votes_df[votes_df["principle"] == principle]
        value_counts = principle_votes["vote"].value_counts()
        return value_counts.get("Agree", 0) / len(principle_votes)

    agreement_by_principle = {
        principle: get_agreement(principle) for principle in principles
    }

    principles_by_agreement = sorted(
        principles,
        key=lambda x: agreement_by_principle[x],
    )

    def get_acc(principle: str) -> int:
        value_counts = votes_df[votes_df["principle"] == principle][
            "vote"
        ].value_counts()
        try:
            acc = value_counts.get("Agree", 0) / (
                value_counts.get("Disagree", 0) + value_counts.get("Agree", 0)
            )
            # main sort by accuracy, then by agreement, then by disagreement (reversed)
            return acc, value_counts.get("Agree", 0), -value_counts.get("Disagree", 0)
        except ZeroDivisionError:
            return 0, value_counts.get("Agree", 0), -value_counts.get("Disagree", 0)

    acc_by_principle = {principle: get_acc(principle) for principle in principles}

    principles_by_acc = sorted(
        principles,
        key=lambda x: acc_by_principle[x],
    )

    def get_relevance(principle: str) -> float:
        principle_votes = votes_df[votes_df["principle"] == principle]
        value_counts = principle_votes["vote"].value_counts()
        return (value_counts.get("Agree", 0) + value_counts.get("Disagree", 0)) / len(
            principle_votes
        )

    relevance_by_principle = {
        principle: get_relevance(principle) for principle in principles
    }

    principles_by_relevance = sorted(
        principles,
        key=lambda x: relevance_by_principle[x],
    )

    def get_perf(principle: str) -> float:
        acc = acc_by_principle[principle][0]
        relevance = relevance_by_principle[principle]
        return (acc - 0.5) * relevance * 2

    perf_by_principle = {principle: get_perf(principle) for principle in principles}

    principles_by_perf = sorted(
        principles,
        key=lambda x: perf_by_principle[x],
    )

    return {
        "principles": principles,
        "num_pairs": num_pairs,
        "metrics": {
            "agreement": {
                "by_principle": agreement_by_principle,
                "principle_order": principles_by_agreement,
            },
            "acc": {
                "by_principle": {
                    principle: acc_by_principle[principle][0]
                    for principle in principles
                },  # because multiple secondary sort values
                "principle_order": principles_by_acc,
            },
            "relevance": {
                "by_principle": relevance_by_principle,
                "principle_order": principles_by_relevance,
            },
            "perf": {
                "by_principle": perf_by_principle,
                "principle_order": principles_by_perf,
            },
        },
    }


def get_metric_cols_by_principle(
    principle: str,
    metrics: dict,
    metric_names: str,
    metrics_cols_start_y: float,
    metrics_cols_width: float,
) -> dict:
    num_cols = len(metric_names)
    metric_col_width = metrics_cols_width / num_cols

    metric_col_options = {
        "agreement": {
            "short": "Agr.",
            "descr": "Agreement: proportion of all votes that agree with original preferences",
        },
        "acc": {
            "short": "Acc.",
            "descr": "Accuracy: proportion of non-irrelevant votes ('agree' or 'disagree')<br>that agree with original preferences",
        },
        "relevance": {
            "short": "Rel.",
            "descr": "Relevance: proportion of all votes that are not 'not applicable'",
        },
        "perf": {
            "short": "Perf.",
            "descr": "Performance: relevance * (accuracy - 0.5) * 2",
        },
    }

    return [
        [
            metrics_cols_start_y + (i + 1) * metric_col_width,
            metrics["metrics"][metric_name]["by_principle"][principle],
            metric_col_options[metric_name]["short"],
            metric_col_options[metric_name]["descr"],
        ]
        for i, metric_name in enumerate(metric_names)
    ]


def get_ordering_options(
    metrics,
    shown_metric_names: list,
    initial: str,
) -> list:
    order_options = {
        "agreement": [
            "Agreement (desc.)",
            metrics["metrics"]["agreement"]["principle_order"],
        ],
        "acc": ["Accuracy (desc.)", metrics["metrics"]["acc"]["principle_order"]],
        "relevance": [
            "Relevance (desc.)",
            metrics["metrics"]["relevance"]["principle_order"],
        ],
        "perf": ["Performance (desc.)", metrics["metrics"]["perf"]["principle_order"]],
    }

    if initial not in order_options.keys():
        raise ValueError(f"Initial ordering metric '{initial}' not found.")

    ordering = [
        value for key, value in order_options.items() if key in shown_metric_names
    ]

    # make sure initial is first
    ordering.insert(0, ordering.pop(ordering.index(order_options[initial])))

    return ordering
