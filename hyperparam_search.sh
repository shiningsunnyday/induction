#!/usr/bin/env bash

# chmod +x run_hparam_search.sh

LATENT_DIM_VALUES=256

NUM_SAMPLES=3

ENCODER_LAYERS=(3 5 7)

DECODER_LAYERS=4

ENCODER="TOKEN"

DATAPKL="TOKEN"

BATCH_SIZE=256

EPOCHS=500

CUDA="cuda"


for EL in "${ENCODER_LAYERS[@]}"
do
    echo "Running training with encoder layers=$EL"
    python train.py \
        --num-samples $NUM_SAMPLES \
        --encoder-layers $EL \
        --decoder-layers $DECODER_LAYERS \
        --encoder $ENCODER \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --cuda $CUDA \
        --latent-dim $LATENT_DIM_VALUES \
        --datapkl $DATAPKL
done
echo "All runs finished!!!"
