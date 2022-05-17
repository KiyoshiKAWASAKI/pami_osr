#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -N alex


# Required modules
module load conda
conda init bash
source activate new_msd_net

python pipeline_alexnet.py --batch-size 16 \
                        --nb_training_classes 293 \
                        --optimizer sgd \
                        --learning-rate 0.1 \
                        --epochs 200 \
                        --print-freq 1