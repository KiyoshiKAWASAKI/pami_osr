#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -N 0221_debug_msd

# Required modules
module load conda
conda init bash
source activate msd_net

CUDA_VISIBLE_DEVICES=3 python demo.py --arch msdnet \
                                       --batch-size 16 \
                                       --nb_training_classes 336 \
                                       --optimizer sgd \
                                       --learning-rate 0.1 \
                                       --gpu 1 \
                                       --epochs 5 \
                                       --print-freq 1 \
                                       --tf_board_path /afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/0220_debug/tf