# Initial version date:
# 07/30/2020: Getting the threshold for rejecting novelty
# Author: Jin Huang


import numpy as np
# import statistics
import sys
import os
import glob
from shutil import copyfile



prob_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_general/probs_0809_thresh.npy"
label_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_07232020/07_29_threshold/0730_targets.npy"

umd_data_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1/known_classes/images"
organized_data_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition"

first_round_unknown_classes = [  1,   2,  75,  82,  94,  97,  98,  99, 100, 103,
                                105, 117, 118, 130, 136, 137, 176, 178, 190, 191,
                                193, 194, 198, 224, 232, 257, 258, 261, 267, 268,
                                286, 288, 313, 319, 330, 345, 359, 384, 388, 398]

second_round_unknown_classes = [14, 15, 35, 38, 72, 77, 126, 128, 150, 151,
                                201, 203, 124, 125, 407, 408, 396, 397, 382, 385,
                                371, 373, 342, 401, 160, 326, 310, 309, 300, 301,
                                119, 255, 139, 140, 188, 196, 391, 392]

known_classes = [  4,   5,  10,  11,  55,  73,  91, 108, 114, 115,
                133, 135, 141, 142, 144, 154, 155, 156, 162, 163,
                164, 171, 172, 173, 202, 205, 206, 207, 236, 237,
                238, 273, 274, 314, 315, 343, 386, 402, 403, 410]

unknown_classes = first_round_unknown_classes + second_round_unknown_classes


first_round_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                       "dataset_v1_3_partition/npy_json_files/unknown_round_1.npy"
second_round_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                        "dataset_v1_3_partition/npy_json_files/unknown_round_2.npy"
save_unknown_rt_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                       "dataset_v1_3_partition/npy_json_files/unknown.npy"



def get_novelty_thresh(prob_file_path,
                       label_file_path,
                       top_k):

    """

    :param prob_file_path:
    :param label_file_path:
    :return:
    """

    probs = np.load(prob_file_path)[:, :, :369]
    labels = np.load(label_file_path)

    print(probs.shape) # (96237, 5, 413)
    print(labels.shape) # (96237,)

    max_probs = []
    top_k_probs = []

    for i in range(probs.shape[0]):
        one_prob = probs[i, :, :]

        # Top-1 prob
        max_p = np.max(one_prob)
        max_probs.append(max_p)

        # Top-5 prob: need to extract 5th from each block
        for j in range(one_prob.shape[0]):
            sub_prob = one_prob[j, :]

            top_k_p = np.sort(sub_prob)[-top_k]
            top_k_probs.append(top_k_p)

    # Get the prob of top-1 and top-k
    median_1 = statistics.median(max_probs)
    avg_1 = statistics.mean(max_probs)

    median_k = statistics.median(top_k_probs)
    avg_k = statistics.mean(top_k_probs)

    print(median_1)
    print(avg_1)
    print(median_k)
    print(avg_k)







def pick_unknown_unknown_classes(nb_total_classes,
                                 first_round_unknown_classes,
                                 second_round_unknown_classes,
                                 known_classes,
                                 nb_unknown_unknown_classes=40):
    """

    :param nb_total_classes:
    :param first_round_unknown_classes:
    :param second_round_unknown_classes:
    :param known_classes:

    :return:
    """

    # Generate a list of class indices
    all_classes = list(range(1, nb_total_classes+1))
    # print(all_classes)

    # List all the classes with RTs
    classes_with_rt = first_round_unknown_classes + \
                      second_round_unknown_classes + \
                      known_classes
    # print(len(classes_with_rt))

    # Get the classes that don't have RTs
    classes_without_rt = list(set(all_classes) - set(classes_with_rt))
    # print(classes_without_rt)

    # Pick 40 classes as (test) unknown_unknown
    step = round(len(classes_without_rt)/nb_unknown_unknown_classes)
    selected_class_index_list = list(range(1, len(classes_without_rt), int(step)))
    selected_class_list = [classes_without_rt[i] for i in selected_class_index_list]
    # print(selected_class_list)

    # Get the rest 255 known classes
    left_255_classes = list(set(classes_without_rt)-set(selected_class_list))

    return selected_class_list, left_255_classes




def organize_and_copy_data(original_data_path,
                           target_data_path,
                           known_known_classes_with_rt,
                           known_known_classes_without_rt,
                           unknown_unknown_classes):
    """
    02/16/2021 - Reorganize the original dataset for generating Json files

    Original dataset path: dataset_v1/known_classes/images/
        train - 413 classes
        val - 413 classes

    Target data organization:
        val =>
            * 40+38 - train_valid, known_unknown, with RT
            * 40 - train_valid, known_known, with RT
            * 255 - train_valid, known_known, without RT
            * 40 - extra classes
        train =>
            * 40+38 - test, known_unknown, without RT
            * 40+255 - test, known_known, without RT
            * 40 - test, unknown_unknown, without RT, no match in "val"

    :return:
    """

    train_valid_source = os.path.join(original_data_path, "val")
    train_valid_target = os.path.join(target_data_path, "train_valid")

    test_source = os.path.join(original_data_path, "train")
    test_target = os.path.join(target_data_path, "test")

    # Process "val" folder for train_valid
    # Part 1: known_known_with_rt
    for one_class in known_known_classes_with_rt:
        one_class = str(one_class).zfill(5)
        one_dir_source_path = os.path.join(train_valid_source, one_class)
        one_dir_target_path = os.path.join(train_valid_target, "known_known_with_rt", one_class)

        if not os.path.exists(one_dir_target_path):
            os.mkdir(one_dir_target_path)

        # This is the list of image names
        all_imgs = os.listdir(one_dir_source_path)

        for one_image in all_imgs:
            if one_image.endswith(".JPEG"):
                src = os.path.join(one_dir_source_path, one_image)
                dst = os.path.join(one_dir_target_path, one_image)
                copyfile(src, dst)

        print("Finished processing one train_valid known_known_with_rt dir: %s" % one_dir_target_path)


    # Part 2: known_known_without_rt
    for one_class in known_known_classes_without_rt:
        one_class = str(one_class).zfill(5)
        one_dir_source_path = os.path.join(train_valid_source, one_class)
        one_dir_target_path = os.path.join(train_valid_target, "known_known_without_rt", one_class)

        if not os.path.exists(one_dir_target_path):
            os.mkdir(one_dir_target_path)

        # This is the list of image names
        all_imgs = os.listdir(one_dir_source_path)

        for one_image in all_imgs:
            if one_image.endswith(".JPEG"):
                src = os.path.join(one_dir_source_path, one_image)
                dst = os.path.join(one_dir_target_path, one_image)
                copyfile(src, dst)

        print("Finished processing one train_valid known_known_without_rt dir: %s" % one_dir_target_path)


    # Part 3: known_unknown
    for one_class in unknown_classes:
        one_class = str(one_class).zfill(5)
        one_dir_source_path = os.path.join(train_valid_source, one_class)
        one_dir_target_path = os.path.join(train_valid_target, "known_unknown", one_class)

        if not os.path.exists(one_dir_target_path):
            os.mkdir(one_dir_target_path)

        # This is the list of image names
        all_imgs = os.listdir(one_dir_source_path)

        for one_image in all_imgs:
            if one_image.endswith(".JPEG"):
                src = os.path.join(one_dir_source_path, one_image)
                dst = os.path.join(one_dir_target_path, one_image)
                copyfile(src, dst)

        print("Finished processing one train_valid known_unknown dir: %s" % one_dir_target_path)


    # Process "train" folder for test
    # Part 1: known_known_with_rt
    for one_class in known_known_classes_with_rt:
        one_class = str(one_class).zfill(5)
        one_dir_source_path = os.path.join(test_source, one_class)
        one_dir_target_path = os.path.join(test_target, "known_known_with_rt", one_class)

        if not os.path.exists(one_dir_target_path):
            os.mkdir(one_dir_target_path)

        # This is the list of image names
        all_imgs = os.listdir(one_dir_source_path)

        for one_image in all_imgs:
            if one_image.endswith(".JPEG"):
                src = os.path.join(one_dir_source_path, one_image)
                dst = os.path.join(one_dir_target_path, one_image)
                copyfile(src, dst)

        print("Finished processing one test known_known_with_rt dir: %s" % one_dir_target_path)

    # Part 2: known_known_without_rt
    for one_class in known_known_classes_without_rt:
        one_class = str(one_class).zfill(5)
        one_dir_source_path = os.path.join(test_source, one_class)
        one_dir_target_path = os.path.join(test_target, "known_known_without_rt", one_class)

        if not os.path.exists(one_dir_target_path):
            os.mkdir(one_dir_target_path)

        # This is the list of image names
        all_imgs = os.listdir(one_dir_source_path)

        for one_image in all_imgs:
            if one_image.endswith(".JPEG"):
                src = os.path.join(one_dir_source_path, one_image)
                dst = os.path.join(one_dir_target_path, one_image)
                copyfile(src, dst)

        print("Finished processing one test known_known_without_rt dir: %s" % one_dir_target_path)

    # Part 3: known_unknown
    for one_class in unknown_classes:
        one_class = str(one_class).zfill(5)
        one_dir_source_path = os.path.join(test_source, one_class)
        one_dir_target_path = os.path.join(test_target, "known_unknown", one_class)

        if not os.path.exists(one_dir_target_path):
            os.mkdir(one_dir_target_path)

        # This is the list of image names
        all_imgs = os.listdir(one_dir_source_path)

        for one_image in all_imgs:
            if one_image.endswith(".JPEG"):
                src = os.path.join(one_dir_source_path, one_image)
                dst = os.path.join(one_dir_target_path, one_image)
                copyfile(src, dst)

        print("Finished processing one test known_unknown dir: %s" % one_dir_target_path)


    # Part : unknown_unknown
    for one_class in unknown_unknown_classes:
        one_class = str(one_class).zfill(5)
        one_dir_source_path = os.path.join(test_source, one_class)
        one_dir_target_path = os.path.join(test_target, "unknown_unknown", one_class)

        if not os.path.exists(one_dir_target_path):
            os.mkdir(one_dir_target_path)

        # This is the list of image names
        all_imgs = os.listdir(one_dir_source_path)

        for one_image in all_imgs:
            if one_image.endswith(".JPEG"):
                src = os.path.join(one_dir_source_path, one_image)
                dst = os.path.join(one_dir_target_path, one_image)
                copyfile(src, dst)

        print("Finished processing one test unknown_unknown dir: %s" % one_dir_target_path)




def combine_rt_files(unknown_rt_first_round_path,
                     unknown_rt_second_round_path,
                     save_unknown_rt_path):
    """
    Combine two npy files into 1 file

    :param unknown_rt_first_round:
    :param unknown_rt_second_round:
    :param save_unknown_rt_path:
    :return:
    """

    first_round_rt = np.load(unknown_rt_first_round_path)
    second_round_rt = np.load(unknown_rt_second_round_path)

    print(first_round_rt.shape)
    print(first_round_rt[0, :])
    print(second_round_rt.shape)

    combined = np.concatenate((first_round_rt, second_round_rt))
    print(combined.shape)

    np.save(save_unknown_rt_path, combined)










if __name__ == "__main__":
    # get_novelty_thresh(prob_file_path=prob_npy_path,
    #                    label_file_path=label_npy_path,
    #                    top_k=3)

    # selected_class_list, left_255_classes = \
    #         pick_unknown_unknown_classes(nb_total_classes=413,
    #                                      first_round_unknown_classes=first_round_unknown_classes,
    #                                      second_round_unknown_classes=second_round_unknown_classes,
    #                                      known_classes=known_classes)
    # selected_class_list: 40 unknown_unknown
    # left_255_classes: 255 known_known_classes without RTs


    # organize_and_copy_data(original_data_path=umd_data_path,
    #                        target_data_path=organized_data_save_path,
    #                        known_known_classes_with_rt=known_classes,
    #                        known_known_classes_without_rt=left_255_classes,
    #                        unknown_unknown_classes=selected_class_list)

    combine_rt_files(unknown_rt_first_round_path=first_round_npy_path,
                     unknown_rt_second_round_path=second_round_npy_path,
                     save_unknown_rt_path=save_unknown_rt_path)
