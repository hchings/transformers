export TRAINING_DIR="/fsx/erincho/examples/flax/language-modeling"
export ARTIFACTS_DIR="/fsx/erincho/examples/flax/language-modeling/roberta-base"
#export TRAINING_DIR="./"

python ${TRAINING_DIR}/run_mlm_flax.py \
    --output_dir=$ARTIFACTS_DIR \
    --model_type="roberta" \
    --config_name=$ARTIFACTS_DIR \
    --dataset_name="oscar" \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --max_seq_length="128" \
    --weight_decay="0.01" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --learning_rate="3e-4" \
    --warmup_steps="1000" \
    --overwrite_output_dir \
    --num_train_epochs="18" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --logging_steps="10" \
    --save_steps="2500" \
    --eval_steps="2500"