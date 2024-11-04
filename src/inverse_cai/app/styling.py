"""Module for styling the ICAI app in Gradio."""

import gradio as gr


CUSTOM_CSS = """
.title-row {
    margin-top: 0.5rem !important;
    margin-bottom: -1rem !important;
    padding: 0.5rem !important;
    border-radius: 0.5rem !important;
}
.title-row h2 {
    opacity: 0.6 !important;
}
"""

THEME = gr.themes.Base(primary_hue="neutral", secondary_hue="neutral").set(
    button_secondary_background_fill="*neutral_200",
    button_secondary_background_fill_dark="*primary_700",
    button_secondary_background_fill_hover="*neutral_100",
    button_secondary_background_fill_hover_dark="*neutral_600",
    block_info_text_color="*neutral_600",
)
