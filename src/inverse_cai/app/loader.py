import pathlib
import json

import gradio as gr


def load_data(path: str):

    print("test")

    gr.Info(f"Loading data from path '{path}'")

    # check results dir inside the path
    results_dir = pathlib.Path(path) / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found in path '{path}'")

    # check if the results dir is empty
    if not any(results_dir.iterdir()):
        raise FileNotFoundError(f"Results directory is empty in path '{path}'")

    result_files = list(results_dir.iterdir())

    gr.Info(f"Loaded {len(result_files)} result files from path '{path}'")

    votes_file = results_dir / "041_votes_per_cluster.json"

    with open(votes_file, "r") as f:
        votes = json.load(f)

    votes_df = create_votes_df(votes)

    gr.Info(f"Loaded votes: {votes}")

    return gr.Markdown(votes), gr.BarPlot(votes_df)


def create_votes_df(dict_votes: dict):

    import pandas as pd

    votes_df = pd.DataFrame(dict_votes)
    return votes_df
