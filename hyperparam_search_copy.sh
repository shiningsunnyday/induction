#!/usr/bin/env bash

# chmod +x run_hparam_search.sh

# LATENT_DIM_VALUES=(64 128 256 512 1024)
LATENT_DIM_VALUES=256

EMBED_DIM_VALUES=256

NUM_SAMPLES=3

ENCODER_LAYERS=(5 6 7 8)

DECODER_LAYERS=6

ENCODER="GNN"

DATAPKL="GNN"

BATCH_SIZE=256

EPOCHS=500

CUDA="cuda"

KL_DIV=0.5

SEED=1234


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
        --embed-dim $EMBED_DIM_VALUES \
        --datapkl $DATAPKL \
        --klcoeff $KL_DIV \
        --seed $SEED

done
echo "All runs finished!!!"
