import gradio.themes.utils.fonts

import inverse_cai.app.interface as interface
from inverse_cai.app.constants import USERNAME, PASSWORD

# make gradio work offline
gradio.themes.utils.fonts.GoogleFont.stylesheet = lambda self: None


demo = interface.generate()


def run():
    if USERNAME and PASSWORD:
        auth = (USERNAME, PASSWORD)
    else:
        auth = None

    demo.launch(auth=auth)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
