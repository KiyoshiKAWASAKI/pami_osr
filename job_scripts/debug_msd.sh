#!/bin/bash

# Required modules
module load conda
conda init bash
source activate MSDNet

# Application to execute
python -W ignore main.py --data-root  /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1/known_classes/images \
                           --data ImageNet \
                           --save /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_413_no_novelty_07232020/90_epochs \
                           --arch msdnet \
                           --batch-size 128 \
                           --epochs 90 \
                           --nBlocks 5 \
                           --stepmode even \
                           --step 4 \
                           --base 4 \
                           --nChannels 32 \
                           --growthRate 16 \
                           --grFactor 1-2-4-4 \
                           --bnFactor 1-2-4-4 \
                           --use-valid \
                           --gpu 3 \
                           -j 1
