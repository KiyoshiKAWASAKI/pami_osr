#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu@@cvrl-titanxp -l gpu=1
#$ -N cross_entropy_sgd_0.0005

# Required modules
module load conda
conda init bash
source activate new_msd_net

# Application to execute
python -W ignore main.py --train_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/train_known_known.json \
                           --train_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/train_known_unknown.json \
                           --valid_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/valid_known_known.json \
                           --valid_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/valid_known_unknown.json \
                           --test_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/test_known_known.json \
                           --test_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/test_known_unknown.json \
                           --test_unknown_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/test_unknown_unknown.json \
                           --train_early_exit True \
                           --arch msdnet \
                           --batch-size 64 \
                           --use-valid \
                           --nb_training_classes 336 \
                           --gpu 3 \
                           --use_5_weights False \
                           --use_pp_loss False \
                           --optimizer sgd \
                           --save /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1004_full_set_sgd_lr/0.0005 \
                           --tf_board_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1004_full_set_sgd_lr/0.0005/tf \
                           --log_file_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1004_full_set_sgd_lr/0.0005/info.log \
                           --learning-rate 0.0005 \
                           --epochs 10


