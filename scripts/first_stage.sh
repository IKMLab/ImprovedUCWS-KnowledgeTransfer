export CUDA_VISIBLE_DEVICES=$2
DATA=$1

MAX_SEG_LEN='4'
CONFIG_FILE=configs/slm_${DATA}_${MAX_SEG_LEN}_config.json
SLM_DIR=exp/first_stage/slm_${DATA}_${MAX_SEG_LEN}
SEED='000'

echo "Training in the second stage."

mkdir -p $SLM_DIR

python -u -m codes.train_first_stage \
    --do_train \
    --do_valid \
    --do_eval \
    --data_name $DATA \
    --config_file $CONFIG_FILE \
    --save_dir $SLM_DIR \
    --sgd_lr_rate 16.0 \
    --adam_lr_rate 0.005 \
    --train_steps 6000 \
    --warm_up_steps 800 \
    --train_batch_size 16000 \
    --valid_batch_size 500 \
    --eval_batch_size 500 \
    --segment_token "  " \
    --seed $SEED
