"""Constants for the app."""

### Layout and dimensions
PRINCIPLE_SHORT_LENGTH = 55  # length of principle shown before cutting off
# this sets where the actual plot starts and ends (individual datapoints)
FIG_PROPORTIONS_X = [0.40, 0.99]
FIG_PROPORTIONS_Y = [0.01, 0.91]
SPACE_PER_NUM_COL = 0.04
PRINCIPLE_END_Y = FIG_PROPORTIONS_X[0] - 0.01 - 2 * SPACE_PER_NUM_COL
AGREEMENT_END_Y = FIG_PROPORTIONS_X[0] - 0.01 - SPACE_PER_NUM_COL
ACC_END_Y = FIG_PROPORTIONS_X[0] - 0.01
HEADING_HEIGHT_Y = FIG_PROPORTIONS_Y[1]
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
