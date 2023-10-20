export CUDA_VISIBLE_DEVICES=$1

python -m codes.perturb_tokenization \
--data msr \
--data_type test \
--pertur_bz 128 \
--bound 11
