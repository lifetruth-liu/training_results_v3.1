#!/bin/bash

num_of_run=5

source config_PG_10gpu_epoch35.sh
logdir=$(realpath ../logs-resnet-10gpu)

export MLPERF_SUBMISSION_ORG=Fujitsu
export MLPERF_SUBMISSION_PLATFORM=PRIMERGY-CDI
for idx in $(seq 1 $num_of_run); do
    NEXP=1 CONT=nvcr.io/nvdlfwea/mlperfv31/resnet:20230913.mxnet DATADIR=/mnt/data4/work/forMXNet_no_resize  \
        LOGDIR=$logdir PULL=0 ./run_with_docker.sh
done


