#!/bin/bash
# Required modules
module load conda
conda init bash
source activate new_msd_net

python -W ignore main.py \
                --train_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json \
                --train_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json \
                --valid_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json \
                --valid_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json \
                --test_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json \
                --test_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json \
                --test_unknown_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json \
                --evaluate-from /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1115/5_weights_setup_1_no_pp/save_models/checkpoint_099.pth.tar \
                --save /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1115/5_weights_setup_1_no_pp/ \
                --save_probs_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1115/5_weights_setup_1_no_pp/save_models/test_results/probs.npy \
                --save_targets_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1115/5_weights_setup_1_no_pp/save_models/test_results/targets.npy \
                --save_original_label_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1115/5_weights_setup_1_no_pp/save_models/test_results/original_labels.npy \
                --save_rt_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1115/5_weights_setup_1_no_pp/save_models/test_results/rts.npy \
                --epochs 1 \
                --batch-size 1 \
                --thresh_top_1 0.90 \
                --thresh_top_3 0.05 \
                --thresh_top_5 0.03 \
                --evalmode anytime \
                --nb_training_classes 369 \
                --test_with_novel True \
                --save_probs True \
                --gpu 2 \
                --arch msdnet \
                --use-valid