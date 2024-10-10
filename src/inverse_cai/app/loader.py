import pathlib
import json
import pandas as pd

import gradio as gr


def load_json_file(path: str):
    with open(path, "r") as f:
        content = json.load(f)

    return content


def load_data(path: str):
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
    votes = load_json_file(votes_file)

    principle_file = results_dir / "030_distilled_principles_per_cluster.json"
    principles = load_json_file(principle_file)

    print("Votes:", votes)

    votes_df = pd.DataFrame(
        [
            {"principle": principle, **votes[str(id)]}
            for id, principle in principles.items()
        ]
    )

    return gr.BarPlot(
        votes_df,
        x="principle",
        y="for",
        caption="agreement",
    )


def create_votes_df(dict_votes: dict):

    import pandas as pd

    votes_df = pd.DataFrame(dict_votes)
    return votes_df
