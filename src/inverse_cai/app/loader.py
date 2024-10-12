import pathlib
import json
import ast
import pandas as pd

import gradio as gr

import plotly.express as px

df = px.data.tips()
fig = px.bar(df, x="total_bill", y="day", orientation="h")


def load_json_file(path: str):
    with open(path, "r") as f:
        content = json.load(f)

    return content


def get_comparison_data(id: int):
    return {
        "prompt": "What color is the sky?",
        "response_1": "blue",
        "response_2": "red",
    }


def create_votes_df(results_dir: pathlib.Path) -> list[dict]:
    votes_per_comparison = pd.read_csv(results_dir / "040_votes_per_comparison.csv")
    # rename column Unnamed: 0 to comparison_id
    votes_per_comparison = votes_per_comparison.rename(
        columns={"Unnamed: 0": "comparison_id"}
    )
    # set comparison_id as index
    votes_per_comparison = votes_per_comparison.set_index("comparison_id")

    print("votes_per_comparison", votes_per_comparison)
    principles_by_id: dict = load_json_file(
        results_dir / "030_distilled_principles_per_cluster.json",
    )

    print("principles_by_id", principles_by_id)

    # list of dicts with votes per comparison, per principle
    # including the prompt and responses
    votes_by_comparison_and_principle = []

    for comparison_id, values in votes_per_comparison.iterrows():

        # get dict from pd.Series with string value of dict
        vote_dict = ast.literal_eval(values["votes"])

        comparison_data = get_comparison_data(comparison_id)

        principles_by_id

        # for each principle, add an entry including comparison data
        for principle_id, vote in vote_dict.items():
            principle = principles_by_id[str(principle_id)]
            votes_by_comparison_and_principle.append(
                {
                    "comparison_id": comparison_id,
                    "principle_id": principle_id,
                    "principle": principle,
                    "vote": vote,
                    "weight": 1,
                    **comparison_data,
                }
            )

    print("votes", votes_by_comparison_and_principle)

    return pd.DataFrame(votes_by_comparison_and_principle)


def load_data(path: str):
    # check results dir inside the path
    results_dir = pathlib.Path(path) / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found in path '{path}'")

    # check if the results dir is empty
    if not any(results_dir.iterdir()):
        raise FileNotFoundError(f"Results directory is empty in path '{path}'")

    gr.Info(f"Loading result files from path '{path}'")

    votes_df = create_votes_df(results_dir)

    table = gr.DataFrame(votes_df)
    fig = px.bar(
        votes_df,
        x="weight",
        y="principle",
        color="vote",
        orientation="h",
        hover_data=["prompt", "response_1", "response_2"],
    )
    plot = gr.Plot(fig)

    return table, plot
