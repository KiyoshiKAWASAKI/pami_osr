#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -N 0321_debug_loss

# Required modules
module load conda
conda init bash
source activate new_msd_net

CUDA_VISIBLE_DEVICES=3 python demo.py --arch msdnet \
                                       --batch-size 4 \
                                       --nb_training_classes 336 \
                                       --optimizer sgd \
                                       --learning-rate 0.1 \
                                       --gpu 1 \
                                       --epochs 5 \
                                       --print-freq 1 \
                                       --test_with_novel True