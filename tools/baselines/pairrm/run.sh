#! /bin/bash

# Reproduce the pairrm fine-tuning results. This runs the entire pipeline,
# including data preparation, base-model fetching, fine-tuning, and evaluation.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 SOURCE_DATA_DIR [PAIRRM_DATA_DIR] [LOG_DIR] [MODEL_DIR]"
    echo "SOURCE_DATA_DIR: Directory containing the processed data"
    echo "PAIRRM_DATA_DIR: Directory to store the PairRM data (default: a temporary directory)"
    echo "LOG_DIR: Directory to store the logs (default: a temporary directory)"
    echo "MODEL_DIR: Directory to store the fine-tuned models (default: a temporary directory)"
    exit 1
fi

temp_dir=$(mktemp -d)
SOURCE_DATA_DIR="$1"
PAIRRM_DATA_DIR="${2:-$temp_dir/data}"
LOG_DIR="${3:-$temp_dir/logs}"
MODEL_DIR="${4:-$temp_dir/models}"
mkdir -p "$LOG_DIR"

EPOCHS=5    

script_dir=$(dirname "$0")

if ! python3 -c "import llm_blender.train_ranker"; then
    echo "Could not import the `train_ranker` module." >&2
    echo "Please install the LLM-Blender fork:" >&2
    # If https://github.com/yuchenlin/LLM-Blender/pull/32 gets merged we can use the official repo.
    echo 'pip install "llm_blender[train,eval] @ git+https://github.com/timokau/LLM-Blender.git@icai-pairrm-finetune"' >&2
    exit 1
fi

# Check if data already exists
if [ -d "$PAIRRM_DATA_DIR" ]; then
    echo "Data already exists in $PAIRRM_DATA_DIR"
    echo "Delete it (d) or skip generation (s)? [d/s]"
    read -r delete_data
    if [ "$delete_data" = "d" ]; then
        rm -r "$PAIRRM_DATA_DIR"
    elif [ "$delete_data" = "s" ]; then
        echo "Skipping data generation"
    else
        echo "Invalid input"
        exit 1
    fi
fi
if [ ! -d "$PAIRRM_DATA_DIR" ]; then
    echo "Preparing PairRM data in $PAIRRM_DATA_DIR"
    log_file="$LOG_DIR/prepare_data.log"
    echo "Logging to $log_file"
    time bash "$script_dir/prepare_data.sh" "$SOURCE_DATA_DIR" "$PAIRRM_DATA_DIR" >"$log_file"
    echo "Done generating PairRM data"
fi

echo "Evaluating untuned PairRM. Logging to $LOG_DIR"
for ds in $(ls "$PAIRRM_DATA_DIR"); do
    log_file="$LOG_DIR/$ds-no-tune.log"
    echo "Evaluating on $ds. Logging to $log_file."
    time bash "$script_dir/fine_tune_pairrm.sh" "$MODEL_DIR" "$PAIRRM_DATA_DIR" "$ds" 0 >"$log_file" 2>&1
done

echo "Fine-tuning PairRM. Logging to $LOG_DIR. Saving models to $MODEL_DIR"
mkdir -p "$MODEL_DIR"
for ds in $(ls "$PAIRRM_DATA_DIR"); do
    log_file="$LOG_DIR/$ds-fine-tune.log"
    echo "Fine-tuning on $ds. Logging to $log_file."
    time bash "$script_dir/fine_tune_pairrm.sh" "$MODEL_DIR" "$PAIRRM_DATA_DIR" "$ds" "$EPOCHS" >"$log_file" 2>&1
done

for ds in $(ls "$PAIRRM_DATA_DIR"); do
    tune_log_file="$LOG_DIR/$ds-fine-tune.log"
    eval_log_file="$LOG_DIR/$ds-no-tune.log"
    # Extract "metric_1" of "test_sel" ("selection", i.e., preference choice,
    # on test set). metric_1 is accuracy, see compute_metrics_for_pairranker in
    # llm_blender/pair_ranker/trainer.py, which computes the mean "score" (1 if
    # correct choice, 0 otherwise) of the chosen output, which equals the
    # accuracy.
    tune_accuracy=$(sed -n "s/.*'test_sel': {'metric_1': \([0-9.]\+\)}.*/\1/p" "$tune_log_file")
    no_tune_accuracy=$(sed -n "s/.*'test_sel': {'metric_1': \([0-9.]\+\)}.*/\1/p" "$eval_log_file")
    echo "Fine-tuned $ds: $tune_accuracy. No tuning: $no_tune_accuracy"
done


echo "Done fine-tuning PairRM. Logs are in $LOG_DIR. Data is in $PAIRRM_DATA_DIR."

echo "NOTE: This script left data behind in the following locations. You may want to delete it if you do not need it anymore."
echo "Data: $PAIRRM_DATA_DIR"
echo "Logs: $LOG_DIR"
echo "Models: $MODEL_DIR"
echo "Delete all? [y/N]"
read -r delete_temp
if [ "$delete_temp" = "y" ]; then
    rm -r "$PAIRRM_DATA_DIR" "$LOG_DIR" "$MODEL_DIR"
	rmdir "$temp_dir"
fi
