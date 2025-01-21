"""Constants for the app."""

import os

# App/package version
import importlib.metadata

VERSION = importlib.metadata.version("inverse_cai")

DEFAULT_DATASET_PATH = "exp/outputs/prism_v2"

# Constants from environment vars
# get env var with github token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# App username and password
# Will block app behind login if env vars are set
USERNAME = os.getenv("ICAI_APP_USER")
PASSWORD = os.getenv("ICAI_APP_PW")
ALLOW_LOCAL_RESULTS = os.getenv("ICAI_ALLOW_LOCAL_RESULTS", "true").lower() == "true"

### Layout and dimensions
PRINCIPLE_SHORT_LENGTH = 70  # length of principle shown before cutting off
# this sets where the actual plot starts and ends (individual datapoints)
FIG_PROPORTIONS_X = [0.60, 0.99]

FIG_HEIGHT_PER_PRINCIPLE = 10  # height of each principle in px
FIG_HEIGHT_HEADER = 50
FIG_HEIGHT_BOTTOM = 10


def get_fig_proportions_y(num_principles):
    total_height_px = (
        num_principles * FIG_HEIGHT_PER_PRINCIPLE
        + FIG_HEIGHT_HEADER
        + FIG_HEIGHT_BOTTOM
    )

    return [
        FIG_HEIGHT_BOTTOM / total_height_px,
        (FIG_HEIGHT_BOTTOM + FIG_HEIGHT_PER_PRINCIPLE * num_principles)
        / total_height_px,
    ]


NON_FIG_LEN = FIG_PROPORTIONS_X[0]
PRINCIPLE_END_X = NON_FIG_LEN * 0.70
METRICS_START_X = PRINCIPLE_END_X + 0.01
MENU_X = 0.3
MENU_Y = 0.97


# Text style
FONT_FAMILY = '"Open Sans", verdana, arial, sans-serif'


### Colors
LIGHT_GREEN = "#d9ead3"
DARK_GREEN = "#38761d"
LIGHT_RED = "#f4cacb"
DARK_RED = "#a61d00"
LIGHTER_GREY = "#fafafa"
LIGHT_GREY = "#e4e4e7"
DARK_GREY = "rgba(192, 192, 192, 0.8)"
VERY_DARK_GREY = "rgba(48, 48, 48, 0.8)"
# used for the background of reconstruction votes
COLORS_DICT = {
    "Agree": "#93c37d",
    "Disagree": "#c17c92",
    "Not applicable": DARK_GREY,
    "Invalid": "black",
}
DARK_COLORS_DICT = {
    "Agree": DARK_GREEN,
    "Disagree": DARK_RED,
    "Not applicable": VERY_DARK_GREY,
}
PAPER_BACKGROUND_COLOR = "white"  # LIGHT_GREY
PLOT_BACKGROUND_COLOR = "white"  # LIGHT_GREY

NONE_SELECTED_VALUE = "(None selected)"


# Plotly config
# values to be removed from the modebar]
# (tool bar that has no use for us, but is difficult to
# remove when working with Gradio and Plotly combined)
PLOTLY_MODEBAR_POSSIBLE_VALUES = [
    "autoScale2d",
    "autoscale",
    "editInChartStudio",
    "editinchartstudio",
    "hoverCompareCartesian",
    "hovercompare",
    "lasso",
    "lasso2d",
    "orbitRotation",
    "orbitrotation",
    "pan",
    "pan2d",
    "pan3d",
    "reset",
    "resetCameraDefault3d",
    "resetCameraLastSave3d",
    "resetGeo",
    "resetSankeyGroup",
    "resetScale2d",
    "resetViewMap",
    "resetViewMapbox",
    "resetViews",
    "resetcameradefault",
    "resetcameralastsave",
    "resetsankeygroup",
    "resetscale",
    "resetview",
    "resetviews",
    "select",
    "select2d",
    "sendDataToCloud",
    "senddatatocloud",
    "tableRotation",
    "tablerotation",
    "toImage",
    "toggleHover",
    "toggleSpikelines",
    "togglehover",
    "togglespikelines",
    "toimage",
    "zoom",
    "zoom2d",
    "zoom3d",
    "zoomIn2d",
    "zoomInGeo",
    "zoomInMap",
    "zoomInMapbox",
    "zoomOut2d",
    "zoomOutGeo",
    "zoomOutMap",
    "zoomOutMapbox",
    "zoomin",
    "zoomout",
]
