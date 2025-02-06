#!/bin/bash -l

source /path/to/venv/bin/activate

echo Using Python `which python`

wandb login

NNODES=1
NRANKS_PER_NODE=4
NPROCESSES=$(( NNODES * NRANKS_PER_NODE ))
MODEL_NAME="model_name"

MAIN_IP_ADDR=$(hostname -i)

WANDB_PROJECT="project_name"
RUN_NAME="run_name"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_RUN_GROUP="grooup_name"

OUTDIR=`pwd`/"$WANDB_PROJECT"/debug/"$RUN_NAME"

HF_HOME=/path/to/HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
# NCCL_COLLNET_ENABLE=1
# NCCL_NET_GDR_LEVEL=PHB

echo Training from: $MODEL_NAME  Project: $WANDB_PROJECT  Run name: $RUN_NAME
echo Output Directory: $OUTDIR

# deepspeed setup, update the following to suit your system

# HOSTFILE="`pwd`"/hostfile
# rm ${HOSTFILE}
# touch ${HOSTFILE}
# NGPUS=$(nvidia-smi -L | wc -l)
# cat $PBS_NODEFILE > ${HOSTFILE}
# sed -e 's/$/ slots=4/' -i ${HOSTFILE}

# DEEPSPEED="`pwd`"/.deepspeed_env
# rm ${DEEPSPEED}
# touch ${DEEPSPEED}
# echo "PATH=${PATH}" >> ${DEEPSPEED}
# echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ${DEEPSPEED}
# echo "http_proxy=${http_proxy}" >> ${DEEPSPEED}
# echo "https_proxy=${https_proxy}" >> ${DEEPSPEED}
# echo "CFLAGS=${CFLAGS}" >> ${DEEPSPEED}
# echo "PYTHONUSERBASE=$PYTHONUSERBASE" >> ${DEEPSPEED}
# echo "HF_HOME=${HF_HOME}" >> ${DEEPSPEED}
# echo "WANDB_PROJECT=${WANDB_PROJECT}" >> ${DEEPSPEED}

DATASET_PATH=/path/to/datasets


deepspeed \
    --hostfile=${HOSTFILE} --no_local_rank run_mlm.py \
    --deepspeed ds_config_zero2.json \
    --model_name_or_path "$MODEL_NAME" \
    --train_file "$DATASET_PATH"train.json \
    --validation_file "$DATASET_PATH"test.json \
    --cache_dir "$HF_HOME" \
    --report_to wandb \
    --do_train \
    --do_eval \
    --bf16 \
    --max_seq_length 512 \
    --pad_to_max_length \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 3e-4 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --max_steps 100 \
    --warmup_steps 1800 \
    --save_strategy steps \
    --save_steps 3000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --logging_strategy steps \
    --logging_steps 50 \
    --output_dir "$OUTDIR" \
    --run_name $RUN_NAME \
    --seed 50
