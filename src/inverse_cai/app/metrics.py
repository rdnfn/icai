"""Compute metrics """

import pandas as pd


def get_agreement(value_counts: pd.Series) -> float:
    return value_counts.get("Agree", 0) / value_counts.sum()


def get_acc(value_counts: pd.Series) -> float:
    try:
        acc = value_counts.get("Agree", 0) / (
            value_counts.get("Disagree", 0) + value_counts.get("Agree", 0)
        )
        return acc
    except ZeroDivisionError:
        return 0


def get_relevance(value_counts: pd.Series) -> float:
    return (
        value_counts.get("Agree", 0) + value_counts.get("Disagree", 0)
    ) / value_counts.sum()


def get_perf(value_counts: pd.Series) -> float:
    acc = get_acc(value_counts)
    relevance = get_relevance(value_counts)
    return (acc - 0.5) * relevance * 2


def compute_metrics(votes_df: pd.DataFrame) -> dict:

    # votes_df is a pd.DataFrame with one row
    # per vote, and columns "comparison_id", "principle", "vote"

    metric_fn = {
        "agreement": get_agreement,
        "acc": get_acc,
        "relevance": get_relevance,
        "perf": get_perf,
    }

    principles = votes_df["principle"].unique()
    num_pairs = len(votes_df["comparison_id"].unique())

    metrics = {}

    for principle in principles:
        principle_votes = votes_df[votes_df["principle"] == principle]
        value_counts = principle_votes["vote"].value_counts()

        for metric in metric_fn.keys():
            if metric not in metrics:
                metrics[metric] = {}
            if "by_principle" not in metrics[metric]:
                metrics[metric]["by_principle"] = {}
            metrics[metric]["by_principle"][principle] = metric_fn[metric](value_counts)

    for metric in metrics.keys():
        metrics[metric]["principle_order"] = sorted(
            principles,
            key=lambda x: (
                metrics[metric]["by_principle"][x],
                metrics["relevance"]["by_principle"][x],
            ),
        )

    return {
        "principles": principles,
        "num_pairs": num_pairs,
        "metrics": metrics,
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
