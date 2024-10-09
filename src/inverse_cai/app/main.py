import gradio as gr
import gradio.themes.utils.fonts

import inverse_cai.app.interface as interface

# make gradio work offline
gradio.themes.utils.fonts.GoogleFont.stylesheet = lambda self: None


demo = interface.generate()


def run():

    demo.launch()


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
