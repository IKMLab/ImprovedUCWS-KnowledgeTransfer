export CUDA_VISIBLE_DEVICES=$2
DATA=$1

MAX_SEG_LEN='4'
hug_name='bert-base-chinese'
SLM_DIR=exp/first_stage/slm_${DATA}_${MAX_SEG_LEN}
CLS_DIR=exp/second_stage
SEED='000'

echo "Training in the second stage."

mkdir -p $CLS_DIR

python -m codes.train_second_stage \
    --data_name ${DATA} \
    --early_stop_threshold 30 \
    --exp $SLM_DIR \
    --file_name prediction.txt \
    --save_dir $CLS_DIR \
    --hug_name $hug_name \
    --seed $SEED \
    --save_every_steps 1
