import gradio as gr

import inverse_cai.app.interface as interface

demo = interface.generate()


def run():
    demo.launch()


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
