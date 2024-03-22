#!/bin/bash


# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

#python train_imagenet.py --gpus 0,1,2,3,4,5,6,7 --batch-size 1664 --kv-store device --lr 0.6 \
#    --mom 0.9 --lr-step-epochs 30,60,80 --lars-eta 0.001 --label-smoothing 0.0 --wd 0.0001 \
#    --warmup-epochs 5 --eval-period 4 --eval-offset 2 --optimizer sgd --network resnet-v1b-fl \
#    --num-layers 50 --num-epochs 90 --accuracy-threshold 0.759 --seed 1 --dtype float16 --disp-batches 20 \
#    --image-shape 4,224,224 --fuse-bn-relu 1 --fuse-bn-add-relu 1 --bn-group 1 --min-random-area 0.05 \
#    --max-random-area 1.0 --conv-algo 1 --force-tensor-core 1 --input-layout NHWC --conv-layout NHWC \
#    --batchnorm-layout NHWC --pooling-layout NHWC --batchnorm-mom 0.9 --batchnorm-eps 1e-5 \
#    --data-train /data/train.rec --data-train-idx /data/train.idx --data-val /data/val.rec --data-val-idx /data/val.idx \
#    --dali-dont-use-mmap 0 --dali-hw-decoder-load 0.0 --dali-prefetch-queue 2 --dali-nvjpeg-memory-padding 64 \
#    --input-batch-multiplier 1 --dali-threads 3 --dali-cache-size 0 --dali-roi-decode 0 --dali-preallocate-width 0 \
#    --dali-preallocate-height 0 --dali-tmp-buffer-hint 25273239 --dali-decoder-buffer-hint 1315942 \
#    --dali-crop-buffer-hint 165581 --dali-normalize-buffer-hint 441549 --e2e-cuda-graphs 0 \
#    --use-nvshmem 0 --sustained_training_time 0

SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
OPTIMIZER=${OPTIMIZER:-"sgd"}
BATCHSIZE=${BATCHSIZE:-1664}
INPUT_BATCH_MULTIPLIER=${INPUT_BATCH_MULTIPLIER:-1}
KVSTORE=${KVSTORE:-"device"}
LR=${LR:-"0.6"}
MOM=${MOM:-"0.9"}
LRSCHED=${LRSCHED:-"30,60,80"}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
LARSETA=${LARSETA:-'0.001'}
DALI_HW_DECODER_LOAD=${DALI_HW_DECODER_LOAD:-'0.0'}
WD=${WD:-'0.0001'}
LABELSMOOTHING=${LABELSMOOTHING:-'0.0'}
SEED=${SEED:-1}
EVAL_OFFSET=${EVAL_OFFSET:-2}
EVAL_PERIOD=${EVAL_PERIOD:-4}
DALI_PREFETCH_QUEUE=${DALI_PREFETCH_QUEUE:-2}
DALI_NVJPEG_MEMPADDING=${DALI_NVJPEG_MEMPADDING:-64}
DALI_THREADS=${DALI_THREADS:-3}
DALI_CACHE_SIZE=${DALI_CACHE_SIZE:-0}
DALI_ROI_DECODE=${DALI_ROI_DECODE:-0}
DALI_PREALLOCATE_WIDTH=${DALI_PREALLOCATE_WIDTH:-0}
DALI_PREALLOCATE_HEIGHT=${DALI_PREALLOCATE_HEIGHT:-0}
DALI_TMP_BUFFER_HINT=${DALI_TMP_BUFFER_HINT:-25273239}
DALI_DECODER_BUFFER_HINT=${DALI_DECODER_BUFFER_HINT:-1315942}
DALI_CROP_BUFFER_HINT=${DALI_CROP_BUFFER_HINT:-165581}
DALI_NORMALIZE_BUFFER_HINT=${DALI_NORMALIZE_BUFFER_HINT:-441549}
DALI_DONT_USE_MMAP=${DALI_DONT_USE_MMAP:-0}
NUMEPOCHS=${NUMEPOCHS:-90}
NETWORK=${NETWORK:-"resnet-v1b-fl"}
BN_GROUP=${BN_GROUP:-1}
NUMEXAMPLES=${NUMEXAMPLES:-}
NUMVALEXAMPLES=${NUMVALEXAMPLES:-}
THR="0.759"
E2E_CUDA_GRAPHS=${E2E_CUDA_GRAPHS:-0}
USE_NVSHMEM=${USE_NVSHMEM:-0}
SUSTAINED_TRAINING_TIME=${SUSTAINED_TRAINING_TIME:-0}

DATAROOT="/data"
TIME_TAGS=${TIME_TAGS:-0}
NVTX_FLAG=${NVTX_FLAG:-0}
NCCL_TEST=${NCCL_TEST:-0}
EPOCH_PROF=${EPOCH_PROF:-0}
SYNTH_DATA=${SYNTH_DATA:-0}
DISABLE_CG=${DISABLE_CG:-0}
ENABLE_IB_BINDING=${ENABLE_IB_BINDING:-1}

GPUS=$(seq 0 $(($NGPUS - 1)) | tr "\n" "," | sed 's/,$//')
PARAMS=(
      --gpus               "${GPUS}"
      --batch-size         "${BATCHSIZE}"
      --kv-store           "${KVSTORE}"
      --lr                 "${LR}"
      --mom                "${MOM}"
      --lr-step-epochs     "${LRSCHED}"
      --lars-eta           "${LARSETA}"
      --label-smoothing    "${LABELSMOOTHING}"
      --wd                 "${WD}"
      --warmup-epochs      "${WARMUP_EPOCHS}"
      --eval-period        "${EVAL_PERIOD}"
      --eval-offset        "${EVAL_OFFSET}"
      --optimizer          "${OPTIMIZER}"
      --network            "${NETWORK}"
      --num-layers         "50"
      --num-epochs         "${NUMEPOCHS}"
      --accuracy-threshold "${THR}"
      --seed               "${SEED}"
      --dtype              "float16"
      --disp-batches       "20"
      --image-shape        "4,224,224"
      --fuse-bn-relu       "1"
      --fuse-bn-add-relu   "1"
      --bn-group           "${BN_GROUP}"
      --min-random-area    "0.05"
      --max-random-area    "1.0"
      --conv-algo          "1"
      --force-tensor-core  "1"
      --input-layout       "NHWC"
      --conv-layout        "NHWC"
      --batchnorm-layout   "NHWC"
      --pooling-layout     "NHWC"
      --batchnorm-mom      "0.9"
      --batchnorm-eps      "1e-5"
      --data-train         "${DATAROOT}/train.rec"
      --data-train-idx     "${DATAROOT}/train.idx"
      --data-val           "${DATAROOT}/val.rec"
      --data-val-idx       "${DATAROOT}/val.idx"
      --dali-dont-use-mmap "${DALI_DONT_USE_MMAP}"
      --dali-hw-decoder-load "${DALI_HW_DECODER_LOAD}"
      --dali-prefetch-queue        "${DALI_PREFETCH_QUEUE}"
      --dali-nvjpeg-memory-padding "${DALI_NVJPEG_MEMPADDING}"
      --input-batch-multiplier     "${INPUT_BATCH_MULTIPLIER}"
      --dali-threads       "${DALI_THREADS}"
      --dali-cache-size    "${DALI_CACHE_SIZE}"
      --dali-roi-decode    "${DALI_ROI_DECODE}"
      --dali-preallocate-width "${DALI_PREALLOCATE_WIDTH}"
      --dali-preallocate-height "${DALI_PREALLOCATE_HEIGHT}"
      --dali-tmp-buffer-hint "${DALI_TMP_BUFFER_HINT}"
      --dali-decoder-buffer-hint "${DALI_DECODER_BUFFER_HINT}"
      --dali-crop-buffer-hint "${DALI_CROP_BUFFER_HINT}"
      --dali-normalize-buffer-hint "${DALI_NORMALIZE_BUFFER_HINT}"
      --e2e-cuda-graphs    "${E2E_CUDA_GRAPHS}"
      --use-nvshmem    "${USE_NVSHMEM}"
      --sustained_training_time "${SUSTAINED_TRAINING_TIME}"
)

echo "RN50 PARAMS:"
echo "${PARAMS[@]}"
echo "\n"

python train_imagenet.py "${PARAMS[@]}"; ret_code=$?

sleep 3

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="IMAGE_CLASSIFICATION"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"