import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import pandas as pd
import re
from loguru import logger
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def get_results_from_paths(results_path, dataset):
    results_paths = [p for p in results_path.iterdir() if p.is_dir()]
    results_dfs = {}

    logger.info(f"Loading {len(results_paths)} results files fom path: {results_path}")
    for seed, path in enumerate(results_paths):
        complete_results_path = pathlib.Path(path) / "results"

        # check if old path
        if os.path.exists(complete_results_path / "092_results.csv"):
            path = complete_results_path / "092_results.csv"
        elif dataset == "train":
            path = complete_results_path / "092_results_training.csv"
        elif dataset == "test":
            path = complete_results_path / "093_results_testset.csv"
        elif dataset.startswith("testset"):
            path = complete_results_path / f"093_results_{dataset}.csv"
        else:
            logger.error(f"Could not find results files for dataset {dataset}.")

        constitution_path = complete_results_path / "060_constitution.json"
        if os.path.exists(constitution_path):
            with open(constitution_path, "r") as f:
                constitution = json.load(f)
        else:
            constitution = None

        df = pd.read_csv(path)
        df["annotator"] = df["Unnamed: 0"]
        df = df.drop(columns=["Unnamed: 0"])

        # for annotators with "constitution" in their name, add the constitution
        if constitution:
            df.loc[
                df["annotator"].str.contains("constitution"),
                "constitution",
            ] = constitution
        else:
            df["constitution"] = None
        results_dfs[f"seed{seed}"] = df
    return results_dfs


def parse_avg_df_to_dict(avg_df):
    out_dict = {}
    for i, row in avg_df.iterrows():
        out_dict[row["annotator"]] = {
            "mean": row["human_agreement_mean"],
            "std": row["std"],
            "max": row["max"],
            "min": row["min"],
            "error": row["std"],
            "values": row["values"],
            "constitutions": row.get("constitutions"),
            "max_constitution": row.get("max_constitution"),
            "min_constitution": row.get("min_constitution"),
            "median_constitution": row.get("median_constitution"),
        }
    return out_dict


def get_metrics_dict(results_path, dataset: str = "test"):
    """Given a dictionary of results paths, return the average values for each annotator.

    Args:
        results_paths: dict
            Dictionary of results paths.
            Each key is a separate experiment and plot.
            Each value is the path to the results file.
            All results paths are grouped by annotator and averaged.

    Returns:
        average_results: dict
            Dictionary containing the average values for each annotator.
            Each key is the annotator name.
            Each value is a dictionary containing the following keys:
                "mean": float
                    Mean value for the annotator.
                "std": float
                    Standard deviation of the values for the annotator.
                "max": float
                    Maximum value for the annotator.
                "min": float
                    Minimum value for the annotator.
                "error": float
                    Standard deviation of the values for the annotator.
                "values": list
                    List of all values for the annotator.
                "constitutions": list
                    List of constitutions for the annotator.
                "max_constitution": str
                    Constitution with the highest agreement for the annotator.
                "min_constitution": str
                    Constitution with the lowest agreement for the annotator.
                "median_constitution": str
                    Constitution closest to the median agreement for the annotator.
    """
    results_dfs = get_results_from_paths(results_path, dataset)
    combined_df = pd.concat([df for df in results_dfs.values()])
    average_results = (
        combined_df.groupby("annotator")["Human agreement"].mean().reset_index()
    )
    # rename human agreement column to human_agreement_mean
    average_results.rename(
        columns={"Human agreement": "human_agreement_mean"}, inplace=True
    )
    # max agreement per annotator
    average_results["max"] = (
        combined_df.groupby("annotator")["Human agreement"]
        .max()
        .reset_index()["Human agreement"]
    )
    average_results["min"] = (
        combined_df.groupby("annotator")["Human agreement"]
        .min()
        .reset_index()["Human agreement"]
    )
    average_results["std"] = (
        combined_df.groupby("annotator")["Human agreement"]
        .std()
        .reset_index()["Human agreement"]
    )
    average_results["values"] = (
        combined_df.groupby("annotator")["Human agreement"]
        .apply(lambda x: list(x))
        .reset_index()["Human agreement"]
    )

    # add constitutions in list
    average_results["constitutions"] = (
        combined_df.groupby("annotator")["constitution"]
        .apply(lambda x: list(x))
        .reset_index()["constitution"]
    )

    # Avoid all constitutions having the same index (not sure why this happens, possibly due to the previous reset index?)
    combined_df.reset_index(drop=True, inplace=True)
    # Add max constitution per annotator
    max_constitutions = combined_df.loc[
        combined_df.groupby("annotator")["Human agreement"].idxmax(),
        ["annotator", "constitution"],
    ].reset_index(drop=True)
    max_constitutions.rename(columns={"constitution": "max_constitution"}, inplace=True)

    # Add min constitution per annotator
    min_constitutions = combined_df.loc[
        combined_df.groupby("annotator")["Human agreement"].idxmin(),
        ["annotator", "constitution"],
    ].reset_index(drop=True)
    min_constitutions.rename(columns={"constitution": "min_constitution"}, inplace=True)

    # Add median constitution per annotator
    def choose_worse_median(x):
        # The index of the median in the sorted array. If there is no unqiue
        # median element (in the case of an even number of elements), choose the
        # one to the left (worse agreement).
        median_idx_sorted = len(x) // 2 - 1 if len(x) % 2 == 0 else len(x) // 2
        sorted_agreement = x["Human agreement"].sort_values(ascending=True)
        median_idx_orig = sorted_agreement.iloc[[median_idx_sorted]].index
        return x.loc[median_idx_orig]

    median_constitutions = (
        combined_df.groupby("annotator")
        .apply(choose_worse_median)
        .reset_index(drop=True)
    )

    median_constitutions.rename(
        columns={"constitution": "median_constitution"}, inplace=True
    )

    # Merge max/min/median constitution into the average results DataFrame
    for summary in [max_constitutions, min_constitutions, median_constitutions]:
        average_results = average_results.merge(summary, on="annotator")

    average_results = parse_avg_df_to_dict(average_results)
    return average_results


def get_results_by_model(results_df, model_name):
    mean_agreement = results_df[results_df["annotator"] == model_name][
        "human_agreement_mean"
    ].iloc[0]
    try:
        max_agreement = results_df[results_df["annotator"] == model_name]["max"].iloc[0]
        min_agreement = results_df[results_df["annotator"] == model_name]["min"].iloc[0]
    except KeyError:
        print(f"Could not find max/min for {model_name}")
        max_agreement = None
        min_agreement = None
    return mean_agreement, max_agreement, min_agreement


def wrap_text(text, width, max_lines=6):
    text = text.replace("\n", " ")
    words = text.split(" ")
    wrapped_text = ""
    line = ""
    line_num = 0
    incomplete = False
    for word in words:
        # check if start of list (e.g. "1.", "2.", etc.)
        if word[0].isdigit() and word[1] == ".":
            line_num += 1
            if line_num <= max_lines - 1:
                wrapped_text += line + "\n"
                line = word + " "
            else:
                incomplete = True
                break
        elif len(line) + len(word) + 1 <= width:
            line += word + " "
        else:
            line_num += 1
            if line_num <= max_lines:
                wrapped_text += line + "\n"
                line = "    " + word + " "
            else:
                incomplete = True
                break

    wrapped_text += line
    if incomplete:
        wrapped_text += "[...]"
    return wrapped_text


# Function to execute a notebook
def run_notebook(path_to_notebook, path_to_output=None, save_output=False):
    with open(path_to_notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

        try:
            out = ep.preprocess(nb, {"metadata": {"path": "./"}})
        except CellExecutionError as e:
            out = None
            print(f"Error executing the notebook: {e}")
            raise
        finally:
            if save_output:
                if path_to_output:
                    with open(path_to_output, "w", encoding="utf-8") as f:
                        nbformat.write(nb, f)
                else:
                    nbformat.write(nb, path_to_notebook)


COLORS = [
    # "#fff3cc",
    "#d9ead3",
    "#fff3cc",
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

HATCHES = ["//", "", "|", "-", "+", "x", "o", "O", "/"]

STRENGTH = 0.2
EDGE_COLOR = (STRENGTH, STRENGTH, STRENGTH)

BASELINE_NAMES = ["chatgpt_fn_noinstruction", "alpaca_eval_gpt4_turbo_fn_noinstruction"]

CONSTITUTIONAL_NAMES = [
    "chatgpt_fn_constitutional_base_neutral_v1",
    "gpt4_fn_constitutional_base_neutral_v1",
]

PADDING_TOP = 0.15

FIG_WIDTH = 7
AGREEMENT_HEIGHT = 1.4
CONSTITUTION_HEIGHT = 1.1


def plot_data(
    data: dict,
    save_path: str,
    fig_width=FIG_WIDTH,
    agreement_height=AGREEMENT_HEIGHT,
    constitution_height=CONSTITUTION_HEIGHT,
    model_name_map=None,
    x_axis_margin=0.6,
    legend_remove=False,
    legend_loc="best",
    legend_parent_name=None,
    legend_bbox_to_anchor=None,
    y_label="Agreement (%)",
    colors=None,
    hatches=None,
    add_random_baseline=True,
    add_constitutions=False,
    constitutions: list[str] = None,
    constitution_colors=None,
    constitution_text_width=50,
):
    """Plot the data in the given dictionary.

    Args:
        data (dict): Dictionary containing the data to plot.
        The dictionary should have the following structure:
            {
                "plot_name_1": {
                    "cluster_name_1": {
                        "model_name_1": {
                            "value": 0.5,
                            "error": 0.1,
                        },
                    },
                },
                "plot_name_2": {
                    ...
                },
            }
        save_path (str): Path to save the plot.
        fig_size (tuple, optional): Size of the figure. Defaults to FIG_SIZE.
    """

    if colors is None:
        colors = COLORS
    if hatches is None:
        hatches = HATCHES

    num_plots = len(data)

    if add_constitutions:
        num_rows = 2
        gridspec_kw = {
            "height_ratios": [
                agreement_height,
                constitution_height,
            ]
        }
    else:
        num_rows = 1
        gridspec_kw = None
        constitution_height = 0

    fig, axes = plt.subplots(
        num_rows,
        num_plots,
        figsize=(fig_width, agreement_height + constitution_height),
        sharey=True,
        gridspec_kw=gridspec_kw,
    )

    if num_plots == 1:
        axes = [axes]  # Make it iterable if only one plot
    if num_rows == 1:
        axes = [axes]

    # Create a dictionary to store legend handles and labels to ensure uniqueness
    legend_handles = {}
    model_formatting = {}

    for ax_index, (plot_name, clusters) in enumerate(data.items()):
        ax = axes[0][ax_index]
        num_clusters = len(clusters)
        cluster_width = 0.8  # total width for all bars in one cluster
        bar_width = cluster_width / len(
            next(iter(clusters.values()))
        )  # width of each bar

        if add_random_baseline:
            ax.axhline(y=50, color="grey", linestyle="--", linewidth=1, zorder=0)

        for cluster_index, (cluster_name, models) in enumerate(clusters.items()):
            offsets = np.arange(len(models))
            offsets = offsets - offsets.mean()  # centering bars within each cluster

            if model_name_map:
                model_order_list = list(model_name_map.keys()) + list(
                    set(models.keys()) - set(model_name_map.keys())
                )
            else:
                model_order_list = list(models.keys())

            for model_index, model_name in enumerate(model_order_list):
                if model_name not in models:
                    # skip if model not present in this cluster
                    continue

                model_data = models[model_name]

                # Add model formatting if not already present
                if model_name not in model_formatting:
                    model_formatting[model_name] = {
                        "color": colors[len(model_formatting) % len(colors)],
                        "hatch": hatches[len(model_formatting) % len(hatches)],
                    }

                position = cluster_index + offsets[model_index] * bar_width
                bar = ax.bar(
                    position,
                    model_data["mean"],
                    width=bar_width,
                    label=model_name,
                    yerr=model_data["error"],
                    capsize=5,
                    color=model_formatting[model_name]["color"],
                    hatch=model_formatting[model_name]["hatch"],
                    edgecolor=EDGE_COLOR,
                )
                model_shown_name = (
                    model_name_map[model_name] if model_name_map else model_name
                )
                if model_data["mean"] == 0.0:
                    ax.text(
                        position,
                        0 + 4,
                        f"{model_data['mean']:.0%}\n({model_shown_name})",
                        ha="center",
                        va="bottom",
                        color="grey",
                        fontsize=8,
                    )

                # Only add to legend handles if not already present
                if model_name not in legend_handles:
                    legend_handles[model_shown_name] = bar[0]

            if add_constitutions:
                const_ax = axes[1][ax_index]

                # reduce vertical size of ax plot

                if constitutions is None:
                    # get the first non nan constitution
                    for cluster_name, models in clusters.items():
                        for model_name, model_data in models.items():
                            if model_data["mean"] is not None:
                                constitution = model_data.get("max_constitution", None)
                                break
                        if constitution:
                            break
                else:
                    # get the constitution from the list
                    constitution = constitutions[ax_index]

                # Wrap text based on character width
                wrapped_text = wrap_text(
                    constitution, constitution_text_width
                )  # Adjust the width as needed

                const_ax.text(
                    0.5,
                    0.5 + 0.05,
                    wrapped_text,
                    transform=const_ax.transAxes,
                    fontsize=8,
                    verticalalignment="center",
                    horizontalalignment="center",
                    multialignment="left",
                )

                if ax_index == 0:
                    const_ax.set_ylabel("Example\nconstitutions")
                    # make y label italics
                    const_ax.yaxis.label.set_fontstyle("italic")
                    const_ax.yaxis.label.set_fontsize(9)

                # remove ticks and box
                const_ax.set_xticks([])
                # const_ax.set_yticks([])

                # add background color
                if constitution_colors is not None:
                    const_ax.set_facecolor(constitution_colors[ax_index])
                else:
                    const_ax.set_facecolor("#fff3cc")
                # const_ax.spines["top"].set_visible(False)

                # make sure the text box is the same size as the plot above

        ax.set_title(plot_name)
        ax.set_xticks(range(num_clusters))
        ax.set_xticklabels(list(clusters.keys()))

        # add y label only to first plot
        if ax_index == 0:
            ax.set_ylabel(y_label)

        # check if all cluster keys are empty strings (i.e. no cluster names)
        if all([not cluster_name for cluster_name in clusters.keys()]):
            ax.set_xticklabels([])
            # remove ticks
            ax.tick_params(axis="x", which="both", bottom=False)

        # add some space around edges of plot on x-axis
        ax.set_xlim(-x_axis_margin, num_clusters - 1 + x_axis_margin)

    if len(axes) > 1:
        for bottom_ax in axes[-1]:
            bottom_ax.tick_params(axis="y", left=False, labelleft=False)

    if not legend_remove:
        # Use the accumulated handles for the legend
        # legend_parent = axes[0][-1]
        if not legend_parent_name:
            legend_parent = axes[0][-1]
        elif legend_parent_name == "fig":
            legend_parent = fig
        else:
            raise ValueError(f"Invalid legend parent name: {legend_parent_name}")

        legend = legend_parent.legend(
            title="Annotators",
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=8,
        )
        plt.setp(legend.get_title(), fontsize=8, fontstyle="italic")

    plt.tight_layout(
        pad=0.3,
    )
    plt.savefig(save_path, dpi=300)
    plt.show()


def generate_latex_table(
    combined_data,
    caption,
    label,
    save_path=None,
    create_save_path=True,
    model_name_map=None,
):
    # Collecting data into a list of dictionaries
    data = []
    for dataset in combined_data.keys():
        for model in combined_data[dataset].keys():
            for annotator in combined_data[dataset][model].keys():
                # print(annotator)
                stats = combined_data[dataset][model][annotator]
                annotator = (
                    model_name_map[annotator]
                    if model_name_map is not None
                    else annotator
                )
                data.append(
                    {
                        "Dataset": dataset,
                        "Model": model,
                        "Annotator": annotator,
                        "Mean": stats["mean"],
                        "Std": stats["std"],
                        "Min": stats["min"],
                        "Max": stats["max"],
                    }
                )

    # Create DataFrame
    numerical_results = pd.DataFrame(data)

    # Drop unneeded index columns
    remaining_index_columns = []
    for index_column in ["Dataset", "Model", "Annotator"]:
        if not numerical_results[index_column].eq("").all():
            remaining_index_columns.append(index_column)
        else:
            numerical_results.drop(columns=[index_column], inplace=True)

    num_data_columns = 4
    num_index_columns = len(remaining_index_columns)

    original_columns = numerical_results.columns
    # Use multi index to collapse repeating values with multirow
    numerical_results.set_index(remaining_index_columns, inplace=True)

    # Create a Styler object
    styler = numerical_results.style.set_caption("Numerical Results")
    styler = styler.format(precision=2)
    styler = styler.format("{:.2f}\\%", subset=["Mean", "Min", "Max"])

    # Highlight max and min, but only withing a shared dataset (first element of multi index)
    def bold_max(data):
        props = "textbf:--rwrap;"
        is_max_within_dataset = data == data.groupby(level=0).transform("max")
        return np.where(is_max_within_dataset, props, "")

    def bold_min(data):
        props = "textbf:--rwrap;"
        is_max_within_dataset = data == data.groupby(level=0).transform("min")
        return np.where(is_max_within_dataset, props, "")

    styler.apply(bold_max, subset=["Mean", "Min", "Max"])
    styler.apply(bold_min, subset=["Std"])

    # Export the Styler to LaTeX
    latex_table = styler.to_latex(
        column_format="l" * num_index_columns + "r" * num_data_columns,
        caption=caption,
        label=label,
        hrules=True,
        multirow_align="t",
        position="H",
        position_float="centering",
    )

    # Now manually rewrite the column header, since pandas generates two separate lines for the multiindex and regular column headers [1], which we do not want.
    # Replace everything between "\toprule" and "\midrule"
    # [1] https://stackoverflow.com/questions/73612241/how-to-use-styler-to-latex-to-output-the-column-names-and-multi-index-names-in-t
    original_columns_bold = [r"\\textbf{" + col + "}" for col in original_columns]
    new_header = " & ".join(original_columns_bold) + r" \\\\"
    latex_table = re.sub(
        r"\\toprule.*?\\midrule",
        r"\\toprule" + "\n" + new_header + "\n" + r"\\midrule",
        latex_table,
        flags=re.DOTALL,
    )

    # Repalce tabular with tabularx, e.g. \begin{tabular} -> \begin{tabularx}{\textwidth}, \end{tabular} -> \end{tabularx}
    latex_table = re.sub(
        r"\\begin{tabular}", r"\\begin{tabularx}{\\textwidth}", latex_table
    )
    latex_table = re.sub(r"\\end{tabular}", r"\\end{tabularx}", latex_table)

    if save_path is not None:
        if create_save_path:
            save_path = pathlib.Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex_table)

    return latex_table


def write_constitutions_to_file(
    annotator_metrics, save_path_base, create_save_path=True
):
    # Write min,max,mean to file
    save_path_base = pathlib.Path(save_path_base)
    if create_save_path:
        save_path_base.mkdir(parents=True, exist_ok=True)
    else:
        assert save_path_base.exists(), f"Path {save_path_base} does not exist."
    for constitution_key in [
        "max_constitution",
        "min_constitution",
        "median_constitution",
    ]:
        with open(save_path_base / f"{constitution_key}.txt", "w") as f:
            f.write(annotator_metrics[constitution_key])

    for i, constitution in enumerate(annotator_metrics["constitutions"]):
        with open(save_path_base / f"constitution_{i+1}.txt", "w") as f:
            f.write(constitution)
