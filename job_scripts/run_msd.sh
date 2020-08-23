#!/bin/bash

#$ -M jhuang24@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -N msd_413_50_epochs         # Specify job name
#$ -q gpu@@cvrl-titanxp	        # Specify job queue
#$ -l gpu_card=1
#$ -pe smp 4


# Required modules
module load conda
conda init bash
source activate MSDNet

# Train 413 classes without novelty
python -W ignore main.py --data-root /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1/known_classes/images \
                         --data ImageNet \
                         --test_folder_name val \
                         --save /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_413_no_novelty_07232020/50_epochs \
                         --arch msdnet \
                         --nb_training_classes 413 \
                         --batch-size 128 \
                         --epochs 50 \
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

