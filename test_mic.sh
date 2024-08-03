#!/bin/bash

source activate common

export REPO_DIR=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning

export MODEL=mic
export FOLDER_NAME=MIC
export HF_PATH=BleachNick/MMICL-Instructblip-T5-xxl
export LANG_ENCODER_PATH=Salesforce/instructblip-flan-t5-xxl
export THIRD_FOLDER_NAME=MMICL-InstructBLIP-T5-XXL


export IMAGE_DIR=${REPO_DIR}/dataset/imgs/all/
export EXP_MODE=RANDOM
export EXP_NAME=exp1
export DEVICE=cuda
export SCORING_TYPE=generated_text
export PROMPT_TYPE=ITM
export SEED=42
export ZERO_COT_ACTIVE=is_zero_cot_active
export FEW_COT_ACTIVE=no-is_few_cot_active
export SIMILARITY_MODEL=ClipModel
export COT_MODEL=llava
export TOP_K=100
export SECOND_FOLDER_NAME=TOP_K_${TOP_K}
export SC_EXP_CNT=1

echo "Settings:"
echo MODEL=$MODEL
echo HF_PATH=$HF_PATH
echo LANG_ENCODER_PATH=$LANG_ENCODER_PATH

echo IMAGE_DIR=$IMAGE_DIR
echo EXP_MODE=$EXP_MODE
echo EXP_NAME=$EXP_NAME
echo DEVICE=$DEVICE
echo SCORING_TYPE=$SCORING_TYPE
echo PROMPT_TYPE=$PROMPT_TYPE
echo SEED=$SEED
echo ZERO_COT_ACTIVE=$ZERO_COT_ACTIVE
echo FEW_COT_ACTIVE=$FEW_COT_ACTIVE
echo TOP_K=$TOP_K
echo FOLDER_NAME=$FOLDER_NAME
echo SECOND_FOLDER_NAME=$SECOND_FOLDER_NAME
echo THIRD_FOLDER_NAME=$THIRD_FOLDER_NAME
echo SIMILARITY_MODEL=$SIMILARITY_MODEL
echo COT_MODEL=$COT_MODEL
echo SC_EXP_CNT=$SC_EXP_CNT


tasks=(
    "existence"   
    "plurals"
    "counting-hard"
    "counting-small-quant"
    "counting-adversarial"
    "relations"
    "action-replacement"
    "actant-swap"
    "coreference-standard"
    "coreference-hard"
    "foil-it"
)

shot_counts=( 0 4 8 )

for task in "${tasks[@]}"; do
    for exp_count in "${shot_counts[@]}"; do
        export TASK_NAME=$task
        export EXP_COUNT=$exp_count
        export TOP_N=$exp_count
        export OUTPUT_FILE=${REPO_DIR}/additional_results/${EXP_NAME}/${FOLDER_NAME}/${SECOND_FOLDER_NAME}/${THIRD_FOLDER_NAME}/${EXP_COUNT}_shot/${TASK_NAME}_${MODEL}.json
        export ANNO_FILE=${REPO_DIR}/dataset/annotations/$TASK_NAME.json
        export SIMILARITY_ANNO_FILE=${REPO_DIR}/example_info/similarities/${SIMILARITY_MODEL}/$TASK_NAME.json
        export COT_ANNO_FILE=${REPO_DIR}/example_info/CoT/${COT_MODEL}/$TASK_NAME.json
        
        echo TASK_NAME=$TASK_NAME
        echo EXP_COUNT=$EXP_COUNT
        echo TOP_N=$TOP_N
        echo OUTPUT_FILE=$OUTPUT_FILE
        echo ANNO_FILE=$ANNO_FILE
        echo SIMILARITY_ANNO_FILE=$SIMILARITY_ANNO_FILE
        echo COT_ANNO_FILE=$COT_ANNO_FILE

        python test_mic.py --annotation_file $ANNO_FILE --support_example_count $EXP_COUNT --image_dir $IMAGE_DIR --output_file $OUTPUT_FILE --sup_exp_mode $EXP_MODE --device $DEVICE --scoring_type $SCORING_TYPE --prompt_type $PROMPT_TYPE --seed $SEED --lang_encoder_path $LANG_ENCODER_PATH --hf_path $HF_PATH --similarity_data_path $SIMILARITY_ANNO_FILE --top_k $TOP_K --top_n $TOP_N --$ZERO_COT_ACTIVE --$FEW_COT_ACTIVE --sc_exp_cnt $SC_EXP_CNT --cot_desc_data_path $COT_ANNO_FILE
    done
done
