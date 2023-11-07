export CUDA_VISIBLE_DEVICES=0,4,2,6,1,5,3,7
export TRAIN_ONLY=0
export USE_DIST_OPTIMIZER=True
export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=8
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_MIN_NCHANNELS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
: "${CHECKPOINT_NAME:=""}"
export LOAD_CHECKPOINT="/load_checkpoints/"$CHECKPOINT_NAME
export MICRO_BATCH_SIZE=2
: "${LOAD_MINIMAL_NUM_SAMPLES:=0}"
if [[ "${LOAD_MINIMAL_NUM_SAMPLES}" -eq 1 ]]; then
  export MAX_STEPS=500
  export OVERRIDE_ZERO_CONSUMED_SAMPLES=0
  export INIT_GLOBAL_STEP=0
fi
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NVTE_UB_SPLIT_RS=0
export NVTE_UB_ATOMIC_GEMM_RS=1
export NVTE_RS_STRIDED_ATOMIC=1
export UB_SKIPMC=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NCCL_P2P_NET_CHUNKSIZE=2097152
export MEGATRON_CORE_P2P_CGA_GROUPS=2
export MEGATRON_CORE_P2P_MAX_CTAS=2
export MEGATRON_CORE_EMBD_CGA_GROUPS=2
export MEGATRON_CORE_EMBD_MAX_CTAS=2
export MEGATRON_CORE_DP_CGA_GROUPS=4
export MEGATRON_CORE_DP_MAX_CTAS=8
export MEGATRON_CORE_AMAX_CGA_GROUPS=1
export MEGATRON_CORE_AMAX_MAX_CTAS=1
export MEGATRON_CORE_MP_CGA_GROUPS=2
export MEGATRON_CORE_MP_MAX_CTAS=2
export MEGATRON_CORE_TP_CGA_GROUPS=4
export MEGATRON_CORE_TP_MAX_CTAS=8
export NEMO_MANUAL_GC_IN_VALIDATION=0
export EXTRA_ARGS="exp_manager.create_tensorboard_logger=False ${EXTRA_ARGS:-}"
export NEMO_LOG_TRAIN_LOSS=0
