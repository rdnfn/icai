import gradio as gr

from inverse_cai.app.loader import load_data


def greet(name):
    return "Hello " + name + "!"


def create_data_loader(inp: dict):
    with gr.Group():
        inp["datapath"] = gr.Textbox(
            label="Data Path", value="exp/outputs/2024-10-08_18-24-23"
        )
        inp["load_btn"] = gr.Button("Load")


def create_principle_view(out: dict):
    # placeholder gradio text
    out["title"] = gr.Textbox(label="Title")
    out["principle_box"] = gr.Markdown("*Principle View*")


def generate():

    inp = {}
    out = {}
    with gr.Blocks() as demo:
        with gr.Row() as main_row:
            with gr.Column(scale=1) as left_col:
                with gr.Group():
                    create_data_loader(inp)

            with gr.Column(scale=2) as right_col:
                with gr.Group() as principle_group:
                    create_principle_view(out)

        inp["load_btn"].click(load_data, inputs=[inp["datapath"]])

    return demo
