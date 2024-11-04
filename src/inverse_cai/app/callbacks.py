"""Call backs to be used in the app."""

import pathlib

import gradio as gr
import pandas as pd
from loguru import logger

from inverse_cai.app.loader import create_votes_df
import inverse_cai.app.plotting as plotting
from inverse_cai.app.constants import NONE_SELECTED_VALUE
from inverse_cai.app.builtin_datasets import (
    get_config_from_name,
    get_dataset_from_name,
    BuiltinDataset,
    Config,
)


def generate_callbacks(inp: dict, state: dict, out: dict) -> dict:
    """Generate callbacks for the ICAI app."""

    def load_data(
        path: str,
        prior_state_datapath: str,
        show_individual_prefs: bool,
        pref_order: str,
        filter_col: str,
        filter_val: str,
        filter_col_2: str,
        filter_val_2: str,
        metrics: list[str],
        reset_filters_if_new: bool = True,
        used_from_button: bool = False,
        filterable_columns: list[str] | None = None,
    ):

        if not used_from_button:
            button_updates = update_dataset_buttons("")
        else:
            button_updates = {}

        new_path = True if path != prior_state_datapath else False

        if new_path and reset_filters_if_new:
            filter_col = NONE_SELECTED_VALUE
            filter_val = NONE_SELECTED_VALUE
            filter_col_2 = NONE_SELECTED_VALUE
            filter_val_2 = NONE_SELECTED_VALUE

        # check results dir inside the path
        results_dir = pathlib.Path(path) / "results"
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found in path '{path}'")

        # check if the results dir is empty
        if not any(results_dir.iterdir()):
            raise FileNotFoundError(f"Results directory is empty in path '{path}'")

        gr.Info(f"Updating results (from path '{path}')", duration=3)

        votes_df: pd.DataFrame = create_votes_df(results_dir)

        unfiltered_df = votes_df.copy(deep=True)

        if filterable_columns is not None:
            available_columns = [NONE_SELECTED_VALUE] + [
                filterable_column
                for filterable_column in filterable_columns
                if filterable_column in votes_df.columns.to_list()
            ]
        else:
            available_columns = [NONE_SELECTED_VALUE] + votes_df.columns.to_list()

        for col, val in [(filter_col, filter_val), (filter_col_2, filter_val_2)]:
            if col != NONE_SELECTED_VALUE:
                if val == NONE_SELECTED_VALUE:
                    gr.Warning(
                        f"Filter value is not selected, but filter column '{col}' is."
                    )
                else:
                    logger.debug(f"Filter: only showing data where '{col}' = '{val}'")
                    votes_df = votes_df[votes_df[col] == val]

        if len(votes_df) == 0:
            error_msg = (
                f"No data to display after filtering "
                f"({filter_col} = {filter_val}, "
                f"{filter_col_2} = {filter_val_2}). "
                "Please try other filters"
            )
            raise gr.Error(error_msg)

        fig = plotting.generate_hbar_chart(
            votes_df,
            unfiltered_df=unfiltered_df,
            show_examples=show_individual_prefs,
            sort_examples_by_agreement=(
                True if pref_order == "By reconstruction success" else False
            ),
            shown_metric_names=metrics,
        )

        plot = gr.Plot(fig)

        return {
            inp["filter_col_dropdown"]: gr.Dropdown(
                choices=available_columns,
                value=filter_col,
                interactive=True,
            ),
            inp["filter_col_dropdown_2"]: gr.Dropdown(
                choices=available_columns,
                value=filter_col_2,
                interactive=True,
            ),
            out["plot"]: plot,
            state["df"]: votes_df,
            state["unfiltered_df"]: unfiltered_df,
            state["datapath"]: path,
            **button_updates,
        }

    def update_dataset_buttons(active_dataset: str) -> dict:
        """Update dataset button variants based on active dataset."""
        updates = {}
        for name, btn in inp["dataset_btns"].items():
            updates[btn] = gr.Button(
                variant="primary" if name == active_dataset else "secondary"
            )
        return updates

    def update_advanced_config_and_load_data(
        prior_state_datapath: str,
        selected_adv_config: str,
        dataset_name: str,
    ):
        # load dataset specific setup
        dataset_config: BuiltinDataset = get_dataset_from_name(dataset_name)

        new_path = True if dataset_config.path != prior_state_datapath else False

        if not dataset_config.options:
            simple_config_avail = False
        else:
            simple_config_avail = True

        # load selected advanced config
        if new_path:
            if dataset_config.options:
                selected_adv_config = (
                    dataset_config.options[0].name
                    if dataset_config.options
                    else NONE_SELECTED_VALUE
                )
            else:
                selected_adv_config = NONE_SELECTED_VALUE

        adv_config: Config = get_config_from_name(
            selected_adv_config, dataset_config.options
        )

        # Update button variants
        button_updates = update_dataset_buttons(dataset_name)

        return {
            **button_updates,
            inp["simple_config_dropdown_placeholder"]: gr.Text(
                visible=not simple_config_avail
            ),
            inp["simple_config_dropdown"]: gr.Dropdown(
                choices=(
                    [config.name for config in dataset_config.options]
                    + [NONE_SELECTED_VALUE]
                    if dataset_config.options
                    else [NONE_SELECTED_VALUE]
                ),
                value=selected_adv_config,
                interactive=True,
                visible=simple_config_avail,
            ),
            state["active_dataset"]: dataset_name,  # Update active dataset state
            inp["datapath"]: dataset_config.path,
            state["datapath"]: dataset_config.path,
            state["dataset_name"]: dataset_name,
            **load_data(
                dataset_config.path,
                prior_state_datapath,
                adv_config.show_individual_prefs,
                adv_config.pref_order,
                adv_config.filter_col,
                adv_config.filter_value,
                adv_config.filter_col_2,
                adv_config.filter_value_2,
                metrics=adv_config.metrics,
                reset_filters_if_new=False,
                used_from_button=True,
                filterable_columns=dataset_config.filterable_columns,
            ),
            inp["filter_value_dropdown"]: gr.Dropdown(
                choices=[adv_config.filter_value],
                value=adv_config.filter_value,
                interactive=True,
            ),
            inp["filter_value_dropdown_2"]: gr.Dropdown(
                choices=[adv_config.filter_value_2],
                value=adv_config.filter_value_2,
                interactive=True,
            ),
            inp["show_individual_prefs_dropdown"]: gr.Dropdown(
                value=adv_config.show_individual_prefs,
                interactive=True,
            ),
            inp["pref_order_dropdown"]: gr.Dropdown(
                value=adv_config.pref_order,
                interactive=True,
            ),
            inp["metrics_dropdown"]: gr.Dropdown(
                value=adv_config.metrics,
                interactive=True,
            ),
        }

    def set_filter_val_dropdown(column: str, votes_df: pd.DataFrame):
        if NONE_SELECTED_VALUE in votes_df.columns:
            raise gr.Error(
                f"Column '{NONE_SELECTED_VALUE}' is in the "
                "dataframe. This is currently not "
                "supported."
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
        "update_advanced_config_and_load_data": update_advanced_config_and_load_data,
    }


def attach_callbacks(inp: dict, state: dict, out: dict, callbacks: dict) -> None:

    load_data_inputs = [
        inp["datapath"],
        state["datapath"],
        inp["show_individual_prefs_dropdown"],
        inp["pref_order_dropdown"],
        inp["filter_col_dropdown"],
        inp["filter_value_dropdown"],
        inp["filter_col_dropdown_2"],
        inp["filter_value_dropdown_2"],
        inp["metrics_dropdown"],
    ]

    load_data_outputs = [
        inp["filter_col_dropdown"],
        inp["filter_col_dropdown_2"],
        out["plot"],
        state["df"],
        state["unfiltered_df"],
        state["datapath"],
        inp["datapath"],
    ] + list(
        inp["dataset_btns"].values()
    )  # Add dataset buttons to outputs

    # reload data when load button is clicked or view config is changed
    inp["load_btn"].click(
        callbacks["load_data"],
        inputs=load_data_inputs,
        outputs=load_data_outputs,
    )

    for config_value_dropdown in [
        inp["pref_order_dropdown"],
        inp["show_individual_prefs_dropdown"],
        inp["filter_value_dropdown"],
        inp["filter_value_dropdown_2"],
        inp["metrics_dropdown"],
    ]:
        config_value_dropdown.input(
            callbacks["load_data"],
            inputs=load_data_inputs,
            outputs=load_data_outputs,
        )

    # load dataset when one of dataset buttons is clicked
    update_load_data_inputs = [
        state["datapath"],
        inp["simple_config_dropdown"],
    ]

    update_load_data_outputs = (
        load_data_outputs
        + [
            inp["simple_config_dropdown"],
            inp["simple_config_dropdown_placeholder"],
            state["dataset_name"],
            inp["filter_value_dropdown"],
            inp["filter_value_dropdown_2"],
            inp["show_individual_prefs_dropdown"],
            inp["pref_order_dropdown"],
            inp["metrics_dropdown"],
            state["active_dataset"],  # Add active dataset state
        ]
        + list(inp["dataset_btns"].values())
    )  # Add all dataset buttons as outputs

    for dataset_button in inp["dataset_btns"].values():
        dataset_button.click(
            callbacks["update_advanced_config_and_load_data"],
            inputs=update_load_data_inputs + [dataset_button],
            outputs=update_load_data_outputs,
        )

    inp["simple_config_dropdown"].input(
        callbacks["update_advanced_config_and_load_data"],
        inputs=update_load_data_inputs + [state["dataset_name"]],
        outputs=update_load_data_outputs,
    )

    # update filter value dropdowns when
    # corresponding filter column dropdown is changed
    inp["filter_col_dropdown"].input(
        callbacks["set_filter_val_dropdown"],
        inputs=[inp["filter_col_dropdown"], state["unfiltered_df"]],
        outputs=[inp["filter_value_dropdown"]],
    )
    inp["filter_col_dropdown_2"].input(
        callbacks["set_filter_val_dropdown"],
        inputs=[inp["filter_col_dropdown_2"], state["unfiltered_df"]],
        outputs=[inp["filter_value_dropdown_2"]],
    )
