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

OUTDIR="`pwd`"/eval-results/"$RUN_NAME"

HF_HOME=/path/to/HF_HOME
export HF_HOME=$HF_HOME
# NCCL_COLLNET_ENABLE=1
# NCCL_NET_GDR_LEVEL=PHB

#  export CUDA_VISIBLE_DEVICES=0

# export MPICH_GPU_SUPPORT_ENABLED=1

echo Training from: $MODEL_NAME  Project: $WANDB_PROJECT  Run name: $RUN_NAME
echo Output Directory: $OUTDIR

# deepspeed setup, depends on your system

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

python run_qa.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name squad \
    --report_to wandb \
    --do_eval\
    --per_device_eval_batch_size 16 \
    --eval_accumulation_steps 5 \
    --doc_stride 128 \
    --max_seq_length 384 \
    --output_dir "$OUTDIR" \
    --overwrite_output_dir \
    --run_name $RUN_NAME \
    --overwrite_cache \
    --dataloader_num_workers 1 \
    --eval_do_concat_batches false \

