"""Call backs to be used in the app."""

import gradio as gr
import pandas as pd
import pathlib

from inverse_cai.app.loader import create_votes_df
import inverse_cai.app.plotting as plotting
from inverse_cai.app.constants import NONE_SELECTED_VALUE


def generate_callbacks(inp: dict, state: dict, out: dict) -> dict:

    def load_data(
        path: str,
        prior_state_datapath: str,
        filter_col: str,
        filter_val: str,
    ):
        new_path = True if path != prior_state_datapath else False

        if new_path:
            filter_col = NONE_SELECTED_VALUE
            filter_val = NONE_SELECTED_VALUE

        # check results dir inside the path
        results_dir = pathlib.Path(path) / "results"
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found in path '{path}'")

        # check if the results dir is empty
        if not any(results_dir.iterdir()):
            raise FileNotFoundError(f"Results directory is empty in path '{path}'")

        gr.Info(f"Loading result files from path '{path}'")

        votes_df: pd.DataFrame = create_votes_df(results_dir)

        if filter_col != NONE_SELECTED_VALUE:
            if filter_val == NONE_SELECTED_VALUE:
                gr.Warning(
                    f"Filter value is not selected, but filter column '{filter_col}' is."
                )
            else:
                votes_df = votes_df[votes_df[filter_col] == filter_val]

        fig = plotting.generate_hbar_chart(votes_df)

        plot = gr.Plot(fig)

        return (
            gr.Accordion(visible=True),
            gr.Dropdown(
                choices=[filter_col] + votes_df.columns.to_list(),
                value=filter_col,
                interactive=True,
            ),
            plot,
            votes_df,
            path,
        )

    def set_filter_val_dropdown(column: str, votes_df: pd.DataFrame):
        if NONE_SELECTED_VALUE in votes_df.columns:
            raise gr.Error(
                f"Column '{NONE_SELECTED_VALUE}' is in the dataframe. This is currently not supported."
            )
        if column == NONE_SELECTED_VALUE:
            return gr.Dropdown(
                choices=[NONE_SELECTED_VALUE],
                value=NONE_SELECTED_VALUE,
                interactive=True,
            )
        else:
            avail_values = votes_df[column].unique().tolist()
            return gr.Dropdown(
                choices=[NONE_SELECTED_VALUE] + avail_values,
                value=NONE_SELECTED_VALUE,
                interactive=True,
            )

    return {
        "load_data": load_data,
        "set_filter_val_dropdown": set_filter_val_dropdown,
    }


def attach_callbacks(inp: dict, state: dict, out: dict, callbacks: dict) -> None:

    inp["load_btn"].click(
        callbacks["load_data"],
        inputs=[
            inp["datapath"],
            state["datapath"],
            inp["filter_col_dropdown"],
            inp["filter_value_dropdown"],
        ],
        outputs=[
            inp["filter_accordion"],
            inp["filter_col_dropdown"],
            out["plot"],
            state["df"],
            state["datapath"],
        ],
    )

    inp["filter_col_dropdown"].change(
        callbacks["set_filter_val_dropdown"],
        inputs=[inp["filter_col_dropdown"], state["df"]],
        outputs=[inp["filter_value_dropdown"]],
    )

    inp["filter_value_dropdown"].input(
        callbacks["load_data"],
        inputs=[
            inp["datapath"],
            state["datapath"],
            inp["filter_col_dropdown"],
            inp["filter_value_dropdown"],
        ],
        outputs=[
            inp["filter_accordion"],
            inp["filter_col_dropdown"],
            out["plot"],
            state["df"],
            state["datapath"],
        ],
    )
