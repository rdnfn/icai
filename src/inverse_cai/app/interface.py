import gradio as gr
import pandas as pd

from inverse_cai.app.callbacks import generate_callbacks, attach_callbacks
from inverse_cai.app.constants import NONE_SELECTED_VALUE


def create_data_loader(inp: dict, state: dict):
    state["datapath"] = gr.State(value="")
    state["df"] = gr.State(value=pd.DataFrame())
    state["unfiltered_df"] = gr.State(value=pd.DataFrame())
    with gr.Row(variant="panel"):
        with gr.Column(scale=3, variant="compact", min_width="100px"):
            gr.HTML(
                '<img src="https://github.com/rdnfn/icai/blob/34065605749f42a33ab2fc0be3305e96840e9412/docs/img/00_logo_v0_wide.png?raw=true" alt="Logo" width="350">'
            )
        with gr.Column(scale=3, variant="panel"):
            with gr.Accordion("Load built-in results"):
                # title
                gr.Button("🏟️ Chatbot Arena")
                gr.Button("💎 PRISM")

        with gr.Column(scale=3, variant="panel"):
            with gr.Accordion("Load custom results"):
                with gr.Group():
                    inp["datapath"] = gr.Textbox(
                        label="💾 Path to custom experimental results",
                        value="exp/outputs/2024-10-12_15-58-26",
                    )
                    inp["load_btn"] = gr.Button("Load")

    inp["config"] = gr.Row(visible=True, variant="panel")
    with inp["config"]:
        with gr.Column(
            scale=3,
        ):
            gr.Markdown("To be implemented")
        with gr.Column(
            scale=3,
        ):
            with gr.Accordion(label="⚙️ Advanced config", open=True, visible=True):
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
    with gr.Blocks() as demo:
        create_data_loader(inp, state)

        with gr.Row() as main_row:
            with gr.Column(scale=2, variant="panel") as right_col:
                create_principle_view(out)

        callbacks = generate_callbacks(inp, state, out)
        attach_callbacks(inp, state, out, callbacks)

    return demo
