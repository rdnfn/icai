"""Convert a dataset to the format pairrm expects."""

import json
import argparse

from inverse_cai.experiment.core import setup_data


# Unfortunately our datasets merged the instruction and output into one field,
# while the pairrm format expects them to be separate. We need to split them
# using a heuristic.


def split_instruction_output(text, output_delim, instruction_prefix):
    assert output_delim in text, f"Output delimiter not found. Text: {text}"
    split_results = text.split(output_delim)
    assert (
        len(split_results) == 2
    ), "Output delimiter found more than once, no robust parsing possible. Text: {text}"
    instruction, output = split_results
    assert instruction.startswith(instruction_prefix)
    return instruction[len(instruction_prefix) :].strip(), output.strip()


def split_instruction_output_synthetic(text):
    return split_instruction_output(
        text, output_delim="\n\nOutput: ", instruction_prefix="Instruction: "
    )


def split_instruction_output_alpacaeval(text):
    return split_instruction_output(
        text, output_delim="\n\nAssistant:\n", instruction_prefix="Instruction:\n"
    )


def split_instruction_output_heuristic(text):
    try:
        return split_instruction_output_synthetic(text)
    except AssertionError:
        return split_instruction_output_alpacaeval(text)


# See this for reference on the format:
# https://github.com/timokau/LLM-Blender/blob/82ef7c6067bb85dfceeff7dd31678c4546e2a2a4/data/mini_preferences/all_train.json
def df_data_to_pairrm_json(data, output_path):
    prefs = []
    for i in data.index:
        text_a = data["text_a"][i]
        text_b = data["text_b"][i]

        instr, candidate_text_a = split_instruction_output_heuristic(text_a)
        instr_b, candidate_text_b = split_instruction_output_heuristic(text_b)
        assert instr_b == instr

        prefs.append(
            {
                "id": f"pref_{i}",
                # This dataset differentiates between instruction and input. The
                # intention is likely that instruction is something like "Summarize
                # the following text", and input is the text to summarize. We do not
                # have this distinction in our data, so we leave input empty.
                "instruction": instr,
                "input": "",
                "candidates": [
                    {
                        "text": candidate_text_a,
                        "model": "human",
                        "decoding_method": "preference",
                        "scores": {
                            "human_preference": (
                                1.0 if data["preferred_text"][i] == "text_a" else 0.0
                            )
                        },
                    },
                    {
                        "text": candidate_text_b,
                        "model": "human",
                        "decoding_method": "preference",
                        "scores": {
                            "human_preference": (
                                1.0 if data["preferred_text"][i] == "text_b" else 0.0
                            )
                        },
                    },
                ],
            }
        )

    with open(output_path, "w") as f:
        json.dump(prefs, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for fine-tuning a pair_rm model."
    )
    parser.add_argument(
        "input_icai_csv", type=str, help="Data to prepare in ICAI's csv format."
    )
    parser.add_argument(
        "output_pairrm_json",
        type=str,
        help="Path to save the output json file in the format expected by LLm Blender.",
    )
    parser.add_argument(
        "--invert_labels", action="store_true", help="Invert the labels."
    )
    parser.add_argument(
        "--data_len", type=int, default=None, help="Length of the data to use."
    )
    parser.add_argument(
        "--data_start_index",
        type=int,
        default=0,
        help="Start index of the data to use.",
    )

    args = parser.parse_args()

    data = setup_data(
        data_path=args.input_icai_csv,
        invert_labels=args.invert_labels,
        data_len=args.data_len,
        data_start_index=args.data_start_index,
    )

    df_data_to_pairrm_json(data, args.output_pairrm_json)


if __name__ == "__main__":
    main()
