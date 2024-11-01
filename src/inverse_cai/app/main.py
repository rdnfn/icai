import gradio.themes.utils.fonts

import inverse_cai.app.interface as interface
from inverse_cai.app.constants import USERNAME, PASSWORD

# make gradio work offline
gradio.themes.utils.fonts.GoogleFont.stylesheet = lambda self: None


demo = interface.generate()


def run():
    if USERNAME and PASSWORD:
        auth = (USERNAME, PASSWORD)
        auth_message = "Welcome to the ICAI App demo!"
    else:
        auth = None
        auth_message = None

    demo.launch(auth=auth, auth_message=auth_message)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
