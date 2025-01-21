"""
Profile the load_data function

Usage:
python profile_load_data_fn.py
"""

from line_profiler import LineProfiler
from inverse_cai.app.callbacks import generate_callbacks
from inverse_cai.app.plotting import main
from inverse_cai.app import metrics
from inverse_cai.app.loader import create_votes_df

import gradio as gr


def setup_test_environment():
    # Create minimal input, state, and output dictionaries required by generate_callbacks
    inp = {
        "filter_col_dropdown": gr.Dropdown(),
        "filter_col_dropdown_2": gr.Dropdown(),
        "filter_value_dropdown": gr.Dropdown(),
        "filter_value_dropdown_2": gr.Dropdown(),
        "show_individual_prefs_dropdown": gr.Dropdown(),
        "pref_order_dropdown": gr.Dropdown(),
        "metrics_dropdown": gr.Dropdown(),
        "dataset_info": gr.Markdown(),
        "dataset_btns": {},  # Add any required dataset buttons
        "datapath": gr.Textbox(),
    }

    state = {
        "df": None,
        "unfiltered_df": None,
        "datapath": None,
        "dataset_name": None,
        "active_dataset": None,
    }

    out = {
        "plot": gr.Plot(),
    }

    return inp, state, out


def profile_load_data():
    # Setup test environment
    inp, state, out = setup_test_environment()

    # Get callbacks including load_data
    callbacks = generate_callbacks(inp, state, out)
    load_data = callbacks["load_data"]

    # Setup test parameters
    test_params = {
        "path": "exp/outputs/prism_v2",
        "prior_state_datapath": "",
        "show_individual_prefs": False,
        "pref_order": "By reconstruction success",
        "filter_col": "None selected",
        "filter_val": "None selected",
        "filter_col_2": "None selected",
        "filter_val_2": "None selected",
        "metrics": ["perf", "relevance", "acc"],
        "reset_filters_if_new": True,
        "used_from_button": False,
        "filterable_columns": None,
        "dataset_name": None,
        "dataset_description": None,
    }

    # Create profiler
    profiler = LineProfiler()

    # profiler.add_function(metrics.compute_metrics)  # Add the metrics function

    # profiler.add_function(plotting.generate_hbar_chart)  # Add the plotting function

    profiler.add_function(create_votes_df)  # Add the loader function

    # Add function to profile
    profiler_wrapper = profiler(load_data)
    # Run the function
    profiler_wrapper(**test_params)
    # Print the results
    profiler.print_stats(sort=True, stripzeros=True)


if __name__ == "__main__":
    with gr.Blocks():  # Create Gradio context
        profile_load_data()
