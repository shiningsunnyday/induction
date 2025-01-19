#!/usr/bin/env bash

# chmod +x run_hparam_search.sh

LATENT_DIM_VALUES=256

EMBED_DIM_VALUES=(64 128 256 512 1024)

NUM_SAMPLES=3

ENCODER_LAYERS=8

DECODER_LAYERS=4

ENCODER="TOKEN"

DATAPKL="TOKEN"

BATCH_SIZE=1024

EPOCHS=500

CUDA="cuda"

KL_DIV=0.6


for EB in "${EMBED_DIM_VALUES[@]}"
do
    echo "Running training with embed dim=$EB"
    python train.py \
        --num-samples $NUM_SAMPLES \
        --encoder-layers $ENCODER_LAYERS \
        --decoder-layers $DECODER_LAYERS \
        --encoder $ENCODER \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --cuda $CUDA \
        --latent-dim $LATENT_DIM_VALUES \
        --embed-dim $EB \
        --datapkl $DATAPKL \
        --klcoeff $KL_DIV
done
echo "All runs finished!!!"
