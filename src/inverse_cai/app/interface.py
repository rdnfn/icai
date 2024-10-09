import gradio as gr

from inverse_cai.app.loader import load_data


def greet(name):
    return "Hello " + name + "!"


def create_data_loader():
    with gr.Group():
        datapath = gr.Textbox(
            label="Data Path", value="exp/outputs/2024-10-08_18-24-23"
        )
        load_btn = gr.Button("Load")

        load_btn.click(load_data, inputs=[datapath])


def create_principle_view():
    # placeholder gradio text
    title = gr.Textbox(label="Title")
    principle_box = gr.Markdown("*Principle View*")


def generate():
    with gr.Blocks() as demo:
        with gr.Row() as main_row:
            with gr.Column(scale=1) as left_col:
                with gr.Group():
                    create_data_loader()

            with gr.Column(scale=2) as right_col:
                with gr.Group() as principle_group:
                    create_principle_view()

    return demo
