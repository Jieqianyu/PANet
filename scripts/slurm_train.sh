#!/usr/bin/env bash
set -x

PARTITION=$1
JOB_NAME=$2
BATCH_SIZE=$3
CONFIG=$4
DIR=$5
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    -t 7-00:00:00 \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u cfg_train.py \
    --tcp_port 12345 \
    --batch_size ${BATCH_SIZE} \
    --config ${CONFIG} \
    --tag ${JOB_NAME} \
    --log_dir ${DIR} \
    --launcher slurm