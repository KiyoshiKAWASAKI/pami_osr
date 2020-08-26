#!/bin/bash

# Required modules
module load conda
conda init bash
source activate MSDNet

# Application to execute
python -W ignore main.py --data-root /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_for_threshold \
                           --save /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_329_42_42/debug \
                           --log_file_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_329_42_42/debug/test_logger.log \
                           --train_folder_name small_valid_413 \
                           --test_folder_name small_valid_413 \
                           --train_early_exit True \
                           --arch msdnet \
                           --batch-size 128 \
                           --nb_training_classes 413 \
                           --epochs 3 \
                           --use-valid \
                           --gpu 3
