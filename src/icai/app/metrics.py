"""Compute metrics """

import pandas as pd


def get_agreement(value_counts: pd.Series) -> float:
    return value_counts.get("Agree", 0) / value_counts.sum()


def get_acc(value_counts: pd.Series) -> float:
    """
    Accuracy: proportion of non-irrelevant votes ('agree' or 'disagree')
    that agree with original preferences.

    If there are no non-irrelevant votes, return 0.
    """
    num_agreed = value_counts.get("Agree", 0)
    num_disagreed = value_counts.get("Disagree", 0)

    denominator = num_agreed + num_disagreed
    if denominator == 0:
        return 0
    else:
        return num_agreed / denominator


def get_relevance(value_counts: pd.Series) -> float:
    return (
        value_counts.get("Agree", 0) + value_counts.get("Disagree", 0)
    ) / value_counts.sum()


def get_perf(value_counts: pd.Series) -> float:
    acc = get_acc(value_counts)
    relevance = get_relevance(value_counts)
    return (acc - 0.5) * relevance * 2


def get_num_votes(value_counts: pd.Series) -> int:
    return value_counts.sum()


def get_agreed(value_counts: pd.Series) -> int:
    return value_counts.get("Agree", 0)


def get_disagreed(value_counts: pd.Series) -> int:
    return value_counts.get("Disagree", 0)


def get_not_applicable(value_counts: pd.Series) -> int:
    return value_counts.get("Not applicable", 0)


def compute_metrics(votes_df: pd.DataFrame, baseline_metrics: dict = None) -> dict:

    # votes_df is a pd.DataFrame with one row
    # per vote, and columns "comparison_id", "principle", "vote"

    metric_fn = {
        "agreement": get_agreement,
        "acc": get_acc,
        "relevance": get_relevance,
        "perf": get_perf,
        "num_votes": get_num_votes,
        "agreed": get_agreed,
        "disagreed": get_disagreed,
        "not_applicable": get_not_applicable,
    }

    principles = votes_df["principle"].unique()
    num_pairs = len(votes_df["comparison_id"].unique())

    metrics = {}

    # slightly faster to make data types categorical
    votes_df = votes_df.assign(
        principle=votes_df["principle"].astype("category"),
        vote=votes_df["vote"].astype("category"),
    )

    # more efficient than doing operation for each principle group separately
    value_counts_all = (
        votes_df.groupby(["principle", "vote"], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    # this is equivalent to:
    # grouped = votes_df.groupby("principle", observed=False)
    # for principle in principles:
    #     value_counts = grouped.get_group(principle)["vote"].value_counts(
    #         sort=False, dropna=False
    #     )

    for principle in principles:
        value_counts: pd.Series = value_counts_all.loc[principle]
        value_counts = value_counts.fillna(0)

        for metric in metric_fn.keys():
            if metric not in metrics:
                metrics[metric] = {}
            if "by_principle" not in metrics[metric]:
                metrics[metric]["by_principle"] = {}
            metrics[metric]["by_principle"][principle] = metric_fn[metric](value_counts)

        if baseline_metrics is not None:
            for metric in metric_fn.keys():
                if metric + "_diff" not in metrics:
                    metrics[metric + "_diff"] = {"by_principle": {}}
                if metric + "_base" not in metrics:
                    metrics[metric + "_base"] = {"by_principle": {}}

                metrics[metric + "_diff"]["by_principle"][principle] = (
                    metrics[metric]["by_principle"][principle]
                    - baseline_metrics["metrics"][metric]["by_principle"][principle]
                )

                metrics[metric + "_base"]["by_principle"][principle] = baseline_metrics[
                    "metrics"
                ][metric]["by_principle"][principle]

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


METRIC_COL_OPTIONS = {
    "agreement": {
        "name": "Agreement",
        "short": "Agr.",
        "descr": "Agreement: proportion of all votes that agree with original preferences",
    },
    "acc": {
        "name": "Accuracy",
        "short": "Acc.",
        "descr": "Accuracy: proportion of non-irrelevant votes ('agree' or 'disagree')<br>that agree with original preferences",
    },
    "relevance": {
        "name": "Relevance",
        "short": "Rel.",
        "descr": "Relevance: proportion of all votes that are not 'not applicable'",
    },
    "perf": {
        "name": "Performance",
        "short": "Perf.",
        "descr": "Performance: relevance * (accuracy - 0.5) * 2",
    },
    "perf_base": {
        "name": "Performance on full dataset",
        "short": "(all)",
        "descr": "Performance on all datapoints (not just selected subset)",
    },
    "perf_diff": {
        "name": "Performance difference (full vs subset)",
        "short": "(diff)",
        "descr": "Absolute performance difference to votes on entire dataset",
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

    return [
        [
            metrics_cols_start_y + (i + 1) * metric_col_width,
            metrics["metrics"][metric_name]["by_principle"][principle],
            METRIC_COL_OPTIONS[metric_name]["short"],
            METRIC_COL_OPTIONS[metric_name]["descr"],
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
        "perf_base": [
            "Performance on full dataset (desc.)",
            metrics["metrics"]["perf_base"]["principle_order"],
        ],
        "perf_diff": [
            "Performance difference (desc.)",
            metrics["metrics"]["perf_diff"]["principle_order"],
        ],
    }

    if initial not in order_options.keys():
        raise ValueError(f"Initial ordering metric '{initial}' not found.")

    ordering = [
        value for key, value in order_options.items() if key in shown_metric_names
    ]

    if initial in shown_metric_names:
        # make sure initial is first
        ordering.insert(0, ordering.pop(ordering.index(order_options[initial])))

    return ordering
