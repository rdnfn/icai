"""Call backs to be used in the app."""

import pathlib

import gradio as gr
import pandas as pd
from loguru import logger

from inverse_cai.app.loader import get_votes_df
import inverse_cai.app.plotting
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
        data: dict,
        *,
        reset_filters_if_new: bool = True,
        used_from_button: bool = False,
        filterable_columns: list[str] | None = None,
        dataset_name: str = None,
        dataset_description: str = None,
    ):
        """Load data with dictionary inputs instead of individual arguments."""
        path = data[inp["datapath"]]
        prior_state_datapath = data[state["datapath"]]
        show_individual_prefs = data[inp["show_individual_prefs_dropdown"]]
        pref_order = data[inp["pref_order_dropdown"]]
        plot_col_name = data[inp["plot_col_name_dropdown"]]
        plot_col_values = data[inp["plot_col_value_dropdown"]]
        filter_col = data[inp["filter_col_dropdown"]]
        filter_val = data[inp["filter_value_dropdown"]]
        filter_col_2 = data[inp["filter_col_dropdown_2"]]
        filter_val_2 = data[inp["filter_value_dropdown_2"]]
        metrics = data[inp["metrics_dropdown"]]
        cache = data[state["cache"]]

        if not used_from_button:
            button_updates = update_dataset_buttons("")
        else:
            button_updates = {}

        new_path = True if str(path) != str(prior_state_datapath) else False

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

        votes_df: pd.DataFrame = get_votes_df(results_dir, cache=cache)

        # TODO: check if deep copy is necessary
        unfiltered_df = votes_df.copy(deep=False)

        if filterable_columns is not None:
            available_columns = [NONE_SELECTED_VALUE] + [
                filterable_column
                for filterable_column in filterable_columns
                if filterable_column in votes_df.columns.to_list()
            ]
        else:
            available_columns = [NONE_SELECTED_VALUE] + votes_df.columns.to_list()

        # only update filter value dropdown choices if new path
        # to avoid resetting the filter values (if using built-in dataset)
        # even if using advanced config
        if new_path:
            additional_col_selection_kwargs = {
                "choices": available_columns,
            }
        else:
            additional_col_selection_kwargs = {}

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

        fig = inverse_cai.app.plotting.generate_plot(
            votes_df,
            unfiltered_df=unfiltered_df,
            show_examples=show_individual_prefs,
            sort_examples_by_agreement=(
                True if pref_order == "By reconstruction success" else False
            ),
            shown_metric_names=metrics,
            plot_col_name=plot_col_name,
        )

        plot = gr.Plot(fig)

        return {
            inp["filter_col_dropdown"]: gr.Dropdown(
                value=filter_col,
                interactive=True,
                **additional_col_selection_kwargs,
            ),
            inp["filter_col_dropdown_2"]: gr.Dropdown(
                value=filter_col_2,
                interactive=True,
                **additional_col_selection_kwargs,
            ),
            inp["plot_col_name_dropdown"]: gr.Dropdown(
                value=plot_col_name,
                interactive=True,
                **additional_col_selection_kwargs,
            ),
            out["plot"]: plot,
            state["df"]: votes_df,
            state["unfiltered_df"]: unfiltered_df,
            state["datapath"]: path,
            state["cache"]: cache,
            inp["dataset_info"]: create_dataset_info(
                unfiltered_df=unfiltered_df,
                filtered_df=votes_df,
                dataset_path=path,
                dataset_name=dataset_name,
                dataset_description=dataset_description,
            ),
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

    def update_advanced_config_and_load_data(data: dict):
        """Update config with dictionary inputs instead of individual arguments."""
        prior_state_datapath = data[state["datapath"]]
        selected_adv_config = data[inp["simple_config_dropdown"]]
        cache = data[state["cache"]]

        # get dataset name from button clicked
        # other buttons are not in data dict
        dataset_name = None
        for button in inp["dataset_btns"].values():
            if button in data:
                dataset_name = data[button]

        if dataset_name is None:
            dataset_name = data[state["active_dataset"]]

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
                {
                    inp["datapath"]: dataset_config.path,
                    state["datapath"]: prior_state_datapath,
                    inp[
                        "show_individual_prefs_dropdown"
                    ]: adv_config.show_individual_prefs,
                    inp["pref_order_dropdown"]: adv_config.pref_order,
                    inp["plot_col_name_dropdown"]: adv_config.plot_col_name,
                    inp["plot_col_value_dropdown"]: adv_config.plot_col_values,
                    inp["filter_col_dropdown"]: adv_config.filter_col,
                    inp["filter_value_dropdown"]: adv_config.filter_value,
                    inp["filter_col_dropdown_2"]: adv_config.filter_col_2,
                    inp["filter_value_dropdown_2"]: adv_config.filter_value_2,
                    inp["metrics_dropdown"]: adv_config.metrics,
                    state["cache"]: cache,
                },
                reset_filters_if_new=False,
                used_from_button=True,
                filterable_columns=dataset_config.filterable_columns,
                dataset_name=dataset_config.name,
                dataset_description=dataset_config.description,
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

    def set_filter_val_dropdown(data: dict):
        """Set filter values with dictionary inputs."""
        votes_df = data.pop(state["unfiltered_df"])
        column = data.popitem()[1]

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


def create_dataset_info(
    unfiltered_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
    dataset_description: str | None = None,
) -> str:
    """Create dataset info markdown string.

    Args:
        df: DataFrame containing the dataset
        dataset_name: Name of the dataset
        dataset_description: Description of the dataset

    Returns:
        str: Markdown formatted dataset info
    """
    if unfiltered_df.empty:
        return "*No dataset loaded*"

    if dataset_name is None:
        dataset_name = "N/A"
    if dataset_description is None:
        dataset_description = "N/A"
    if dataset_path is None:
        dataset_path = "N/A"

    metrics = {}

    for name, df in [("Unfiltered", unfiltered_df), ("Filtered", filtered_df)]:
        metrics[name] = {}
        metrics[name]["num_comparisons"] = df["comparison_id"].nunique()
        metrics[name]["num_principles"] = df["principle"].nunique()
        metrics[name]["num_total_votes"] = len(df)

    info = f"""
- **Name**: {dataset_name}
- **Path**: {dataset_path}
- **Description**: {dataset_description}
- **Metrics:**
    - *Total pairwise comparisons*: {metrics["Unfiltered"]["num_comparisons"]:,} (shown: {metrics["Filtered"]["num_comparisons"]:,})
    - *Total tested principles*: {metrics["Unfiltered"]["num_principles"]:,} (shown: {metrics["Filtered"]["num_principles"]:,})
    - *Total votes (comparisons x principles)*: {metrics["Unfiltered"]["num_total_votes"]:,} (shown: {metrics["Filtered"]["num_total_votes"]:,})



"""
    return info


def attach_callbacks(inp: dict, state: dict, out: dict, callbacks: dict) -> None:
    """Attach callbacks using dictionary inputs."""

    all_inputs = {
        inp["datapath"],
        state["datapath"],
        state["dataset_name"],
        state["active_dataset"],
        inp["show_individual_prefs_dropdown"],
        inp["pref_order_dropdown"],
        inp["plot_col_name_dropdown"],
        inp["plot_col_value_dropdown"],
        inp["filter_col_dropdown"],
        inp["filter_value_dropdown"],
        inp["filter_col_dropdown_2"],
        inp["filter_value_dropdown_2"],
        inp["metrics_dropdown"],
        inp["simple_config_dropdown"],
        state["cache"],
    }

    load_data_outputs = [
        inp["plot_col_name_dropdown"],
        inp["plot_col_value_dropdown"],
        inp["filter_col_dropdown"],
        inp["filter_col_dropdown_2"],
        out["plot"],
        state["df"],
        state["unfiltered_df"],
        state["datapath"],
        state["active_dataset"],
        state["dataset_name"],
        state["cache"],
        inp["datapath"],
        inp["dataset_info"],
    ] + list(inp["dataset_btns"].values())

    # reload data when load button is clicked or view config is changed
    inp["load_btn"].click(
        callbacks["load_data"],
        inputs=all_inputs,
        outputs=load_data_outputs,
    )

    for config_value_dropdown in [
        inp["pref_order_dropdown"],
        inp["show_individual_prefs_dropdown"],
        inp["plot_col_value_dropdown"],
        inp["filter_value_dropdown"],
        inp["filter_value_dropdown_2"],
        inp["metrics_dropdown"],
    ]:
        config_value_dropdown.input(
            callbacks["load_data"],
            inputs=all_inputs,
            outputs=load_data_outputs,
        )

    update_load_data_outputs = (
        load_data_outputs
        + [
            inp["simple_config_dropdown"],
            inp["simple_config_dropdown_placeholder"],
            inp["plot_col_value_dropdown"],
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
            inputs=all_inputs.union({dataset_button}),
            outputs=update_load_data_outputs,
        )

    inp["simple_config_dropdown"].input(
        callbacks["update_advanced_config_and_load_data"],
        inputs=all_inputs,
        outputs=update_load_data_outputs,
    )

    # update filter value dropdowns when
    # corresponding filter column dropdown is changed
    for dropdown, output in [
        (inp["plot_col_name_dropdown"], inp["plot_col_value_dropdown"]),
        (inp["filter_col_dropdown"], inp["filter_value_dropdown"]),
        (inp["filter_col_dropdown_2"], inp["filter_value_dropdown_2"]),
    ]:
        dropdown.input(
            callbacks["set_filter_val_dropdown"],
            inputs={state["unfiltered_df"], dropdown},
            outputs=[output],
        )
