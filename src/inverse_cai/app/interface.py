import gradio as gr
import pandas as pd

from inverse_cai.app.callbacks import generate_callbacks, attach_callbacks
from inverse_cai.app.constants import (
    NONE_SELECTED_VALUE,
    VERSION,
    ALLOW_LOCAL_RESULTS,
    DEFAULT_DATASET_PATH,
)
from inverse_cai.app.builtin_datasets import BUILTIN_DATASETS
from inverse_cai.app.info_texts import METHOD_INFO_TEXT, METHOD_INFO_HEADING, TLDR_TEXT
from inverse_cai.app.metrics import METRIC_COL_OPTIONS
from inverse_cai.app.styling import CUSTOM_CSS, THEME


def add_title_row(title: str):
    """Add a title row to the interface.

    Args:
        title (str): Title text to display
    """
    with gr.Row(elem_classes="title-row"):
        gr.Markdown(f"## {title}")


def create_data_loader(inp: dict, state: dict):
    state["datapath"] = gr.State(value="")
    state["df"] = gr.State(value=pd.DataFrame())
    state["unfiltered_df"] = gr.State(value=pd.DataFrame())
    state["dataset_name"] = gr.State(value="")
    state["active_dataset"] = gr.State(value="")
    state["cache"] = gr.State(value={})

    with gr.Row(variant="default"):
        with gr.Column(scale=4, variant="default", min_width="300px"):
            gr.HTML(
                '<img src="https://github.com/rdnfn/icai/blob/34065605749f42a33ab2fc0be3305e96840e9412/docs/img/00_logo_v0_wide.png?raw=true" alt="Logo" width="320">'
            )
        link_button_variant = "secondary"
        link_button_size = "lg"
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown(
                    f"**Research Preview** – ICAI App v{VERSION}",
                    container=True,
                )
                gr.Button(
                    "📖 Paper",
                    link="https://arxiv.org/abs/2406.06560",
                    variant=link_button_variant,
                    size=link_button_size,
                )
                gr.Button(
                    "📦 GitHub",
                    link="https://github.com/rdnfn/icai",
                    variant=link_button_variant,
                    size=link_button_size,
                )

    add_title_row("Background")

    with gr.Row(visible=True, variant="panel"):
        with gr.Column(scale=1):
            gr.Markdown(TLDR_TEXT, container=True)
            with gr.Accordion(METHOD_INFO_HEADING, open=False):
                gr.Markdown(METHOD_INFO_TEXT, container=True)
        with gr.Column(scale=1):
            # https://github.com/rdnfn/icai/blob/e639b4d2d1c11dc0b52bc6a22700363a13b49b91/docs/img/04_app_overview_v1.png
            gr.Markdown(
                '**Overview**\n\n<img src="https://github.com/rdnfn/icai/blob/c356c708b949c0bc7f5cc0f78432bcaba0f371db/docs/img/04_app_overview_v2.png?raw=true" alt="App Overview" style="min-width: 400px; max-width: 500px; display: block; margin-left: auto; margin-right: auto;">',
                container=True,
                sanitize_html=False,
            )

    add_title_row("Data selection")
    with gr.Row(variant="panel"):
        with gr.Column(scale=3):
            with gr.Accordion("Select dataset to analyze"):
                inp["dataset_btns"] = {}
                for dataset in BUILTIN_DATASETS:
                    inp["dataset_btns"][dataset.name] = gr.Button(
                        dataset.name, variant="secondary"
                    )
            with gr.Accordion(
                "Alternatively load local results", visible=ALLOW_LOCAL_RESULTS
            ):
                with gr.Group():
                    inp["datapath"] = gr.Textbox(
                        label="💾 Path",
                        value=DEFAULT_DATASET_PATH,
                    )
                    inp["load_btn"] = gr.Button("Load")

        with gr.Column(scale=3):
            with gr.Accordion("ℹ️ Dataset description", open=True):
                inp["dataset_info"] = gr.Markdown("*No dataset loaded*", container=True)

    add_title_row("View configuration")
    inp["config"] = gr.Row(visible=True, variant="panel")
    with inp["config"]:
        with gr.Column(
            scale=3,
        ):
            inp["simple_config_dropdown_placeholder"] = gr.Markdown(
                "*No simple dataset configuration available. Load different dataset or use advanced config.*",
                container=True,
            )
            inp["simple_config_dropdown"] = gr.Dropdown(
                label="🔧 Feedback subset to show",
                info='Show principles\' performance reconstructing ("explaining") the selected feedback subset. *Example interpretation: If the principle "Select the more concise response" reconstructs GPT-4 wins well, GPT-4 may be more concise than other models in this dataset.*',
                visible=False,
            )
        with gr.Column(
            scale=3,
        ):
            with gr.Accordion(label="⚙️ Advanced config", open=False, visible=True):
                gr.Markdown(
                    "Advanced configuration options that enable filtering the dataset and changing other visibility settings. If available, settings to the left of this menu will automatically set these advanced options. Set advanced options here manually to override."
                )
                with gr.Group():
                    # button to disable efficient
                    inp["show_individual_prefs_dropdown"] = gr.Dropdown(
                        label="🗂️ Show individual preferences (slow)",
                        info="Whether to show individual preference examples. May slow down the app.",
                        choices=[False, True],
                        value=False,
                        interactive=True,
                    )

                    inp["plot_col_name_dropdown"] = gr.Dropdown(
                        label="Show plot across values of column",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    )
                    inp["plot_col_value_dropdown"] = gr.Dropdown(
                        label="Values to show (if none selected, all values are shown)",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=True,
                        multiselect=True,
                    )

                    inp["pref_order_dropdown"] = gr.Dropdown(
                        label="📊 Order of reconstructed preferences",
                        choices=[
                            "By reconstruction success",
                            "Original (random) order",
                        ],
                        value="By reconstruction success",
                        interactive=True,
                    )
                    metric_choices = [
                        (
                            f"{metric['name']}",
                            key,
                        )
                        for key, metric in METRIC_COL_OPTIONS.items()
                    ]
                    inp["metrics_dropdown"] = gr.Dropdown(
                        multiselect=True,
                        label="📈 Metrics to show",
                        choices=metric_choices,
                        value=["perf", "relevance", "acc"],
                        interactive=True,
                    )

                inp["filter_accordion"] = gr.Accordion(
                    label="🎚️ Filter 1", open=False, visible=True
                )
                with inp["filter_accordion"]:
                    inp["filter_col_dropdown"] = gr.Dropdown(
                        label="Filter by column",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    )
                    # add equal sign between filter_dropdown and filter_text

                    inp["filter_value_dropdown"] = gr.Dropdown(
                        label="equal to",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    )
                inp["filter_accordion_2"] = gr.Accordion(
                    label="🎚️ Filter 2", open=False, visible=True
                )
                with inp["filter_accordion_2"]:
                    inp["filter_col_dropdown_2"] = gr.Dropdown(
                        label="Filter by column",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    )
                    # add equal sign between filter_dropdown and filter_text

                    inp["filter_value_dropdown_2"] = gr.Dropdown(
                        label="equal to",
                        choices=[NONE_SELECTED_VALUE],
                        value=NONE_SELECTED_VALUE,
                        interactive=False,
                    )


def create_principle_view(out: dict):
    out["plot"] = gr.Plot()


def generate():

    inp = {}
    state = {}
    out = {}

    with gr.Blocks(theme=THEME, css=CUSTOM_CSS) as demo:
        create_data_loader(inp, state)

        add_title_row("Results")
        with gr.Row():
            with gr.Column(scale=2, variant="panel") as right_col:
                create_principle_view(out)

        with gr.Row():
            gr.HTML(f"<center>Inverse Constitutional AI (ICAI) App v{VERSION}</center>")

        callbacks = generate_callbacks(inp, state, out)
        attach_callbacks(inp, state, out, callbacks)

    return demo
