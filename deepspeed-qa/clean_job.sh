#!/bin/bash -l

source /path/to/venv/bin/activate

echo Using Python `which python`

wandb login

NNODES=1
NRANKS_PER_NODE=4
NPROCESSES=$(( NNODES * NRANKS_PER_NODE ))
MODEL_NAME="model_name"


WANDB_PROJECT="project_name"
WANDB_GROUP="group_name"
RUN_NAME="$(date -I)-1"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_RUN_GROUP="$WANDB_GROUP"

OUTDIR="`pwd`"/"$WANDB_GROUP"/"$RUN_NAME"

HF_HOME=/path/to/HF_HOME
export HF_HOME=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
# NCCL_COLLNET_ENABLE=1
# NCCL_NET_GDR_LEVEL=PHB

echo Training from: $MODEL_NAME  Project: $WANDB_PROJECT  Run name: $RUN_NAME
echo Output Directory: $OUTDIR

# deepspeed setup, depends on your system.

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

deepspeed \
    --hostfile=${HOSTFILE} --no_local_rank run_qa.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name squad \
    --report_to wandb tensorboard \
    --do_train \
    --do_eval \
    --learning_rate 4e-5 \
    --warmup_ratio 0.2 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --doc_stride 128 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --eval_accumulation_steps 10 \
    --max_eval_samples 4096 \
    --logging_strategy steps \
    --logging_steps 50 \
    --output_dir "$OUTDIR" \
    --run_name $RUN_NAME \
    --deepspeed ds_config_zero2.json \
    --seed 50 \
