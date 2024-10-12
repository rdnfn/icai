import pathlib
import json
import ast
import pandas as pd

import gradio as gr

import plotly.express as px

import inverse_cai.app.plotting as plotting

df = px.data.tips()
fig = px.bar(df, x="total_bill", y="day", orientation="h")


def load_json_file(path: str):
    with open(path, "r") as f:
        content = json.load(f)

    return content


def convert_vote_to_string(vote: int) -> str:
    if vote == True:
        return "Agree"
    elif vote == False:
        return "Disagree"
    elif vote is None:
        return "Not applicable"
    else:
        raise ValueError(f"Completely invalid vote value: {vote}")


def create_votes_df(results_dir: pathlib.Path) -> list[dict]:
    votes_per_comparison = pd.read_csv(
        results_dir / "040_votes_per_comparison.csv", index_col="index"
    )
    principles_by_id: dict = load_json_file(
        results_dir / "030_distilled_principles_per_cluster.json",
    )

    # list of dicts with votes per comparison, per principle
    # including the prompt and responses
    votes_by_comparison_and_principle = []

    comparison_df = pd.read_csv(results_dir / "000_train_data.csv", index_col="index")

    def get_comparison_data(id: int):
        return {
            # "prompt": comparison_data.loc[id, "prompt"],
            "text_a": comparison_df.loc[id, "text_a"],
            "text_b": comparison_df.loc[id, "text_b"],
            "preferred_text": comparison_df.loc[id, "preferred_text"],
        }

    for comparison_idx, values in votes_per_comparison.iterrows():

        # get dict from pd.Series with string value of dict
        vote_dict = ast.literal_eval(values["votes"])

        comparison_data = get_comparison_data(comparison_idx)

        # for each principle, add an entry including comparison data
        for principle_id, vote in vote_dict.items():
            principle = principles_by_id[str(principle_id)]
            votes_by_comparison_and_principle.append(
                {
                    "comparison_id": comparison_idx,
                    "principle_id": principle_id,
                    "principle": principle,
                    "vote": convert_vote_to_string(vote),
                    "weight": 1,
                    **comparison_data,
                }
            )

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

    fig = plotting.generate_hbar_chart(votes_df)

    plot = gr.Plot(fig)

    return plot
