export TRAINING_DIR="/fsx/erincho/examples/pytorch/language-modeling"
export ARTIFACTS_DIR="/fsx/erincho/examples/pytorch/language-modeling/roberta-base"
#export TRAINING_DIR="./"

export NUM_GPUS=1
export TOKENIZERS_PARALLELISM=0
export MODEL_DIR="./roberta-base"
export MASTER_ADDR="compute-st-worker-60"
#mkdir -p ${MODEL_DIR}

python3 -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} \
   --rdzv_endpoint=$MASTER_ADDR:29400 \
	 --rdzv_id=100 \
	 --rdzv_backend=c10d ${TRAINING_DIR}/run_mlm_no_trainer.py \
    --output_dir=$ARTIFACTS_DIR \
    --model_type="roberta" \
    --config_name=$ARTIFACTS_DIR \
    --tokenizer_name="${MODEL_DIR}" \
    --dataset_name="oscar" \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --max_seq_length="128" \
    --weight_decay="0.01" \
    --per_device_train_batch_size="160" \
    --per_device_eval_batch_size="160" \
    --gradient_accumulation="4" \
    --learning_rate="3e-4" \
    --num_warmup_steps="1000" \
    --num_train_epochs="18" \
    --logging_steps="10" \
    --seed=42
