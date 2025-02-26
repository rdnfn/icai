#!/usr/bin/env bash

# Convert the datasets needed for the experiments in the paper to the format
# used by PairRM.

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_basedir> <target_dir>"
    exit 1
fi

data_basedir="$1" # Should point to the processed data, e.g., ./data/processed
tgt_dir="$2"
script_dir=$(dirname "$0")
prepdata="python3 $script_dir/convert_data.py"

# Generate synthetic-aligned (same data for val/train/test)
target_base="$tgt_dir/icai_synthetic_aligned"
mkdir -p "$target_base"
src="$data_basedir/synthetic/synthetic_data_aligned_vqvtb.csv"
$prepdata "$src" "${target_base}/val.json"
$prepdata "$src" "${target_base}/train.json"
$prepdata "$src" "${target_base}/test.json"

# Generate synthetic-unaligned (same data for val/train/test)
target_base="$tgt_dir/icai_synthetic_unaligned"
mkdir -p "$target_base"
src="$data_basedir/synthetic/synthetic_data_aligned_vqvtb.csv"
$prepdata --invert_labels "$src" "${target_base}/val.json"
$prepdata --invert_labels "$src" "${target_base}/train.json"
$prepdata --invert_labels "$src" "${target_base}/test.json"

# Generate synthetic-orthogonal
target_base="$tgt_dir/icai_synthetic_orthogonal"
mkdir -p "$target_base"
src="$data_basedir/synthetic/synthetic_data_orthogonal_adcbg.csv"
$prepdata "$src" "${target_base}/val.json"
$prepdata "$src" "${target_base}/train.json"
$prepdata "$src" "${target_base}/test.json"

# Generate alpacaeval-aligned full (val/train/test: 32/292/324)
target_base="$tgt_dir/icai_alpacaeval_full_aligned"
mkdir -p "$target_base"
src="$data_basedir/tatsu_lab/alpacaeval_goldcrossannotations_rand.csv"
$prepdata --data_start_index=0 --data_len=32 "$src" "${target_base}/val.json"
$prepdata --data_start_index=32 --data_len=292 "$src" "${target_base}/train.json"
$prepdata --data_start_index=324 --data_len=324 "$src" "${target_base}/test.json"

# Generate alpacaeval-unaligned full (val/train/test: 32/292/324)
target_base="$tgt_dir/icai_alpacaeval_full_unaligned"
mkdir -p "$target_base"
src="$data_basedir/tatsu_lab/alpacaeval_goldcrossannotations_rand.csv"
$prepdata --invert_labels --data_start_index=0 --data_len=32 "$src" "${target_base}/val.json"
$prepdata --invert_labels --data_start_index=32 --data_len=292 "$src" "${target_base}/train.json"
$prepdata --invert_labels --data_start_index=324 --data_len=324 "$src" "${target_base}/test.json"

# Generate alpacaeval-aligned small (val/train/test: 10/55/65)
target_base="$tgt_dir/icai_alpacaeval_small_aligned"
mkdir -p "$target_base"
src="$data_basedir/tatsu_lab/alpacaeval_goldcrossannotations_rand.csv"
$prepdata --data_start_index=0 --data_len=10 "$src" "${target_base}/val.json"
$prepdata --data_start_index=10 --data_len=55 "$src" "${target_base}/train.json"
$prepdata --data_start_index=200 --data_len=65  "$src" "${target_base}/test.json"

# Generate alpacaeval-unaligned small (val/train/test: 10/55/65)
target_base="$tgt_dir/icai_alpacaeval_small_unaligned"
mkdir -p "$target_base"
src="$data_basedir/tatsu_lab/alpacaeval_goldcrossannotations_rand.csv"
$prepdata --invert_labels --data_start_index=0 --data_len=10 "$src" "${target_base}/val.json"
$prepdata --invert_labels --data_start_index=10 --data_len=55 "$src" "${target_base}/train.json"
$prepdata --invert_labels --data_start_index=200 --data_len=65  "$src" "${target_base}/test.json"
