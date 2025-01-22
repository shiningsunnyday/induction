#!/usr/bin/env bash

# chmod +x run_hparam_search.sh

# LATENT_DIM_VALUES=(64 128 256 512 1024)
LATENT_DIM_VALUES=256

EMBED_DIM_VALUES=256

NUM_SAMPLES=3

ENCODER_LAYERS=4

DECODER_LAYERS=(1 2 3 4 5 6 7 8)

ENCODER="TOKEN"

DATAPKL="TOKEN"

BATCH_SIZE=256

EPOCHS=500

CUDA="cuda:1"

KL_DIV=0.5

SEED=1234

DATASET="bn" # choices=["ckt", "bn", "enas"]

for DL in "${DECODER_LAYERS[@]}"
do
    echo "Running training with decoder layers=$DL"
    python train.py \
        --num-samples $NUM_SAMPLES \
        --encoder-layers $ENCODER_LAYERS \
        --decoder-layers $DL \
        --encoder $ENCODER \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --cuda $CUDA \
        --latent-dim $LATENT_DIM_VALUES \
        --embed-dim $EMBED_DIM_VALUES \
        --datapkl $DATAPKL \
        --klcoeff $KL_DIV \
        --dataset $DATASET \
        --seed $SEED \ 
done
echo "All runs finished!!!"
