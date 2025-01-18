#!/usr/bin/env bash

# chmod +x run_hparam_search.sh

LATENT_DIM_VALUES=(256 512)
NUM_SAMPLES=3
ENCODER_LAYERS=4
DECODER_LAYERS=4
ENCODER="TOKEN"
BATCH_SIZE=256
EPOCHS=500
CUDA="cuda"
DATAPKL=true

for LD in "${LATENT_DIM_VALUES[@]}"
do
    echo "Running training with latent-dim=$LD"
    python train.py \
        --num-samples $NUM_SAMPLES \
        --encoder-layers $ENCODER_LAYERS \
        --decoder-layers $DECODER_LAYERS \
        --encoder $ENCODER \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --cuda $CUDA \
        --latent-dim $LD \
        --datapkl $DATAPKL
done
echo "All runs finished!!!"
