import gradio as gr
import pandas as pd

from inverse_cai.app.loader import load_data


def create_data_loader(inp: dict):
    with gr.Group():
        inp["datapath"] = gr.Textbox(
            label="Data Path", value="exp/outputs/2024-10-12_15-58-26"
        )
        inp["load_btn"] = gr.Button("Load")


def create_principle_view(out: dict):
    # note: barplot without any args can lead to
    # silent error
    # out["barplot"] = gr.BarPlot(pd.DataFrame())
    out["plot"] = gr.Plot()


def generate():

    inp = {}
    out = {}
    with gr.Blocks() as demo:
        with gr.Row() as main_row:
            with gr.Column(scale=1, variant="panel") as left_col:
                with gr.Group():
                    create_data_loader(inp)
        with gr.Row() as main_row:
            with gr.Column(scale=2, variant="panel") as right_col:
                create_principle_view(out)

        inp["load_btn"].click(
            load_data,
            inputs=[inp["datapath"]],
            outputs=[out["plot"]],
        )

    return demo
