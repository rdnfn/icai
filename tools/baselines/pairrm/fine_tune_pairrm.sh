 #!/bin/bash

 # Fine-tune PairRanker on a preference dataset.

 # This was adapted from train_ranker.sh in LLM-Blender.
 # https://github.com/yuchenlin/LLM-Blender/blob/33204d2712944b6b17996f7c079e74cd963ccc7c/train_ranker.sh

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <target_dir> <data_base> <data_name> <num_epochs> [cache_dir]"
    exit 1
fi

target_dir=$(realpath "$1")
data_base=$(realpath "$2")

data_name="$3"
num_train_epochs="$4" # PairRanker uses 5
cache_dir="${5:-pairrm_cache}"
mkdir -p "$cache_dir"

base_checkpoint="hf:llm-blender/PairRM"

backbone_type="deberta"
backbone_name="microsoft/deberta-v3-large"
n_gpu=1

learning_rate=5e-6 # PairRanker uses 1e-5

per_device_eval_batch_size=1
per_device_train_batch_size=1
gradient_accumulation_steps=1
source_maxlength=1224
candidate_maxlength=412
max_grad_norm=10e10 # set a large value to disable gradient clipping
fp16=True # whether to use fp16

do_train=True
if [ ${num_train_epochs} -eq 0 ]; then
    do_train=False
fi

LAUNCH_CMD="python3 -m llm_blender.train_ranker"

train_data_path="${data_base}/${data_name}/train.json"
dev_data_path="${data_base}/${data_name}/val.json"
test_data_path="${data_base}/${data_name}/test.json"
run_name="${data_name}-${num_train_epochs}epochs"


pushd "$cache_dir" # Fine-tuning from a huggingface model leaves a hf_models dir in PWD.
# Run training
${LAUNCH_CMD} \
    --ranker_type "pairranker" \
    --model_type ${backbone_type} \
    --model_name ${backbone_name} \
    --run_name ${run_name} \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates -1 \
    --candidate_model "" \
    --candidate_decoding_method "" \
    --using_metrics "human_preference" \
    --learning_rate ${learning_rate} \
    --source_maxlength ${source_maxlength} \
    --candidate_maxlength ${candidate_maxlength} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${num_train_epochs} \
    --do_train "$do_train" \
    --do_eval True \
    --do_predict True \
    --inference_mode "bubble" \
    --load_checkpoint "$base_checkpoint" \
    --max_grad_norm ${max_grad_norm} \
    --max_train_data_size -1\
    --max_eval_data_size -1 \
    --max_predict_data_size -1 \
    --max_grad_norm ${max_grad_norm} \
    --fp16 ${fp16} \
    --num_pos 5 \
    --num_neg 5 \
    --loss_type "instructgpt" \
    --sub_sampling_mode "all_pair" \
    --output_dir "$target_dir/${run_name}" \
    --overwrite_output_dir True
popd
