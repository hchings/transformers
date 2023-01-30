#!/bin/bash

num_nodes=${1:-2}

SMP_USER=${2:-"erincho"}
CONTAINER_NAME=${3:-"smp"}
SOURCE_CODE_USER=${4:-"$SMP_USER"}


set -ex
# build mpi command
smprun $SMP_USER -n $num_nodes -v --mpi-path /opt/amazon/openmpi/bin/mpirun --notify-exit \
        -c $CONTAINER_NAME \
        -d /fsx/${SOURCE_CODE_USER}/examples/flax/playground/multi-node/ \
        -x NCCL_DEBUG=INFO -x NCCL_PROTO=simple \
        /opt/conda/bin/python run.py \
