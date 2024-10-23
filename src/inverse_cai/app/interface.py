import gradio as gr
import pandas as pd

from inverse_cai.app.callbacks import generate_callbacks, attach_callbacks
from inverse_cai.app.constants import NONE_SELECTED_VALUE, VERSION
from inverse_cai.app.builtin_datasets import BUILTIN_DATASETS


def create_data_loader(inp: dict, state: dict):
    state["datapath"] = gr.State(value="")
    state["df"] = gr.State(value=pd.DataFrame())
    state["unfiltered_df"] = gr.State(value=pd.DataFrame())
    state["dataset_name"] = gr.State(value="")
    with gr.Row(variant="panel"):
        with gr.Column(scale=2, variant="compact", min_width="300px"):
            gr.HTML(
                '<img src="https://github.com/rdnfn/icai/blob/34065605749f42a33ab2fc0be3305e96840e9412/docs/img/00_logo_v0_wide.png?raw=true" alt="Logo" width="300">'
            )
        link_button_variant = "secondary"
        link_button_size = "lg"
        with gr.Column(scale=4):
            pass
        with gr.Column(scale=1):
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
    with gr.Row(variant="panel"):
        with gr.Column(scale=3, variant="panel"):
            with gr.Accordion("Load built-in results"):
                inp["dataset_btns"] = {}
                for dataset in BUILTIN_DATASETS:
                    inp["dataset_btns"][dataset.name] = gr.Button(dataset.name)
        with gr.Column(scale=3, variant="panel"):
            with gr.Accordion("Load custom results"):
                with gr.Group():
                    inp["datapath"] = gr.Textbox(
                        label="💾 Path",
                        value="exp/outputs/2024-10-12_15-58-26",
                    )
                    inp["load_btn"] = gr.Button("Load")

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
                    inp["efficient_mode_dropdown"] = gr.Dropdown(
                        label="🏃 Efficient mode",
                        info="Efficient mode makes the interface faster, but hides individual preference information",
                        choices=[True, False],
                        value=True,
                        interactive=True,
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
    # note: barplot without any args can lead to
    # silent error
    # out["barplot"] = gr.BarPlot(pd.DataFrame())
    out["plot"] = gr.Plot()


def generate():

    inp = {}
    state = {}
    out = {}

    theme = gr.themes.Ocean().set(
        button_secondary_background_fill="*neutral_200",
        button_secondary_background_fill_dark="*primary_700",
        button_secondary_background_fill_hover="*neutral_100",
        button_secondary_background_fill_hover_dark="*neutral_600",
        block_info_text_color="*neutral_600",
    )
    with gr.Blocks(theme=theme) as demo:
        create_data_loader(inp, state)

        with gr.Row() as main_row:
            with gr.Column(scale=2, variant="panel") as right_col:
                create_principle_view(out)

        with gr.Row():
            gr.HTML(f"<center>Inverse Constitutional AI (ICAI) App v{VERSION}</center>")

        callbacks = generate_callbacks(inp, state, out)
        attach_callbacks(inp, state, out, callbacks)

    return demo
