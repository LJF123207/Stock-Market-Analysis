#!/bin/bash

MLRUNS_ROOT=/z5s/morph/home/ljf/qlib/mlruns
TMP_YAML_DIR=tmp_yaml
LOG_ROOT=logs

mkdir -p ${TMP_YAML_DIR}
mkdir -p ${LOG_ROOT}

YAMLS=(
  examples/benchmarks/Localformer/workflow_config_localformer_Times_News.yaml
  examples/benchmarks/ALSTM/workflow_config_alstm_Times_News.yaml
  examples/benchmarks/LSTM/workflow_config_lstm_Times_News.yaml
  examples/benchmarks/TCN/workflow_config_tcn_Times_News.yaml
  examples/benchmarks/GRU/workflow_config_gru_Times_News.yaml
  examples/benchmarks/GATs/workflow_config_gats_Times_News.yaml
  examples/benchmarks/Transformer/workflow_config_transformer_Times_News.yaml
)

FUSION_TYPE=decoder
LAYER_NUM=6
LR=0.0005
EARLY_STOP=10

GPUS=(0 1 2 3 4 5 6 7)


get_free_gpu () {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -k2 -n \
    | head -n1 \
    | cut -d',' -f1
}


for YAML in "${YAMLS[@]}"; do
    MODEL_NAME=$(basename $(dirname ${YAML}))  
    BASENAME=$(basename ${YAML})

    # 等待空闲 GPU
    while true; do
        GPU=$(get_free_gpu)
        if [ -n "${GPU}" ]; then
            break
        fi
        sleep 10
    done

    echo "Using GPU ${GPU} for ${MODEL_NAME}"

    
    if [ "$FUSION_TYPE" = "crossattn" ] || [ "$FUSION_TYPE" = "decoder" ]; then
        NAME_SUFFIX=${MODEL_NAME}_${FUSION_TYPE}_layer${LAYER_NUM}
    else
        NAME_SUFFIX=${MODEL_NAME}_${FUSION_TYPE}
    fi
    LOG_FILE=${LOG_ROOT}/${NAME_SUFFIX}.log
    TMP_YAML=${TMP_YAML_DIR}/${NAME_SUFFIX}.yaml
    cp ${YAML} ${TMP_YAML}

    # 动态修改 yaml
    sed -i "s/fusion_type:.*/fusion_type: ${FUSION_TYPE}/" ${TMP_YAML}
    sed -i "s/layer_num:.*/layer_num: ${LAYER_NUM}/" ${TMP_YAML}
    # sed -i "s/early_stop:.*/early_stop: ${EARLY_STOP}/" ${TMP_YAML}
    # sed -i "s/lr:.*/lr: ${LR}/" ${TMP_YAML}
    
    (
        CUDA_VISIBLE_DEVICES=${GPU} \
        python qlib/cli/run.py ${TMP_YAML} \
        2>&1 | tee ${LOG_FILE}

        EXP_ID=$(grep -oP 'Experiment \K[0-9]+' ${LOG_FILE} | head -n1)
        REC_ID=$(grep -oP 'Recorder \K[a-f0-9]+' ${LOG_FILE} | head -n1)

        if [ -z "${EXP_ID}" ] || [ -z "${REC_ID}" ]; then
            echo "Failed to parse EXP_ID or REC_ID from log"
            exit 1
        fi

        EXP_DIR=${MLRUNS_ROOT}/${EXP_ID}
        if [ "$FUSION_TYPE" = "crossattn" ] || [ "$FUSION_TYPE" = "decoder" ]; then
            NAME_SUFFIX=${MODEL_NAME}_${FUSION_TYPE}_layer${LAYER_NUM}_${REC_ID}
        else
            NAME_SUFFIX=${MODEL_NAME}_${FUSION_TYPE}_${REC_ID}
        fi

        DST_DIR=${EXP_DIR}/${NAME_SUFFIX}

        mkdir -p ${DST_DIR}

        SRC_REC_DIR=${EXP_DIR}/${REC_ID}
        if [ -d "${SRC_REC_DIR}" ]; then
            mv ${SRC_REC_DIR}/* ${DST_DIR}/
            rmdir ${SRC_REC_DIR}
        fi

        # =========================
        # 保存本次 yaml 和 log
        # =========================

        cp ${TMP_YAML} ${DST_DIR}/workflow.yaml
        cp ${LOG_FILE} ${DST_DIR}/run.log

        echo "Saved all artifacts to ${DST_DIR}"

    ) &


    sleep 5
done

wait
echo "All experiments finished"
