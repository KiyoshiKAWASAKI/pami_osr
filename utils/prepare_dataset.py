# Prepare the dataset csvs files for MS-DenseNet baseline study
# Initial version date: 07/18/2020
# 07/29/2020: Making a small subset of training set
# Author: Jin Huang


import os
import csv
from shutil import copyfile
import numpy as np

"""
07/20/2020

Training set: dataset_v1/known_classes/images/train (413 classes) - msd_train_0720.csv
Validation set: dataset_v1/known_classes/images/val (413 classes) - msd_valid_0720.csv
Test set: ObjectNet - msd_test_0720
"""

# Datasets
training_data_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
                     "object_recognition/image_net/derivatives/dataset_v1/known_classes/images/train"
validation_data_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
                       "object_recognition/image_net/derivatives/dataset_v1/known_classes/images/val"
test_data_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
                   "object_recognition/object_net/objectnet-1.0/images/"

# CSV
training_csv_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
                         "object_recognition/image_net/derivatives/sail_on_csv/msd_train_0720.csv"
valid_csv_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
                      "object_recognition/image_net/derivatives/sail_on_csv/msd_valid_0720.csv"
test_csv_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
                     "object_recognition/image_net/derivatives/sail_on_csv/msd_test_0720"

# UMD training set's subset
sub_training_set_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
                        "object_recognition/image_net/derivatives/dataset_v1_for_threshold/train"

small_test_413_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
                        "object_recognition/image_net/derivatives/dataset_v1_for_threshold/small_valid_413"

# UMD data path
original_413_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
                    "object_recognition/image_net/derivatives/dataset_v1/known_classes/images"

target_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
              "object_recognition/image_net/derivatives/dataset_v1_3_partition"

debug_set_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/" \
                 "object_recognition/image_net/derivatives/dataset_v1_3_partition/debug"

# Class list from data collection
first_round_40_classes = [1,   2,  75,  82,  94,  97,  98,  99, 100, 103,
                        105, 117, 118, 130, 136, 137, 176, 178, 190, 191,
                        193, 194, 198, 224, 232, 257, 258, 261, 267, 268,
                        286, 288, 313, 319, 330, 345, 359, 384, 388, 398]

second_round_38_classes = [14, 15, 35, 38, 72, 77, 126, 128, 150, 151,
                            201, 203, 124, 125, 407, 408, 396, 397, 382, 385,
                            371, 373, 342, 401, 160, 326, 310, 309, 300, 301,
                            119, 255, 139, 140, 188, 196, 391, 392]


def gen_label_match(first_round,
                    second_round,
                    nb_total_classes):
    """

    :param first_round:
    :param second_round:
    :param nb_total_classes:
    :return:
    """

    all_labels = list(range(1, nb_total_classes+1))
    print(all_labels)

    all_unknown_labels = first_round + second_round

    known_labels = [x for x in all_labels if x not in all_unknown_labels]
    print(known_labels)

    for i in range(1, len(known_labels)+1):
        print(known_labels[i-1], i)



def generate_csv(data_dir, csv_path):
    """

    :return:
    """

    csv_list = []

    for path, subdirs, files in os.walk(data_dir):
        for name in files:
            csv_list.append(os.path.join(path, name))
            print(os.path.join(path, name))

    with open(csv_path, mode='w') as data_csv:
        csv_writer = csv.writer(data_csv, quoting=csv.QUOTE_ALL)
        csv_writer.writerow(csv_list)




def generate_sub_training_set(source_data_path,
                              target_data_path,
                              extract_ratio):
    """
    Extract a portion of training set for getting novelty threshold or to create a small set for debugging.

    :param source_data_path:
    :param target_data_path:
    :param extra_ratio:
    :return:
    """

    for dir, _, _ in os.walk(source_data_path):
        if (os.path.basename(os.path.normpath(dir)) == "train") or \
                (os.path.basename(os.path.normpath(dir)) == "val"):
            continue

        else:
            folder_name = os.path.basename(os.path.normpath(dir))

            if not os.path.exists(os.path.join(target_data_path, folder_name)):
                os.mkdir(os.path.join(target_data_path, folder_name))

            all_images_list = os.listdir(dir)
            nb_taking = int(len(all_images_list) * extract_ratio)

            image_list = all_images_list[:nb_taking]

            for image_path in image_list:
                image_source_path = os.path.join(dir, image_path)
                image_target_path = os.path.join(os.path.join(target_data_path, folder_name),
                                                 image_path)

                print("copying from %s to %s" % (image_source_path, image_target_path))
                copyfile(image_source_path, image_target_path)




def generate_three_partitions(source_dir,
                              target_dir,
                              known_unknown_classes,
                              unknown_unknown_classes,
                              cp_known_known=False,
                              cp_known_unknown=False,
                              cp_unknown_unknown=False,
                              make_debug_set=False,
                              extract_ratio=0.01,
                              nb_total_classes=413):
    """
    Make copy of data and partition into 3 parts as follows:

    Training and validation:
        Known_known: image/train/333 classes (413-80)
        Known_unknown: image/train/ 40 classes (40 out of 80) -- will be treated as 1 class "other"

    Test:
        Known_known: image/val/333 classes (413-80)
        Known_unknown: image/val/ 40 classes (40 out of 80)
        unknown_unknown: image/val/40 classes (the left 40 out of 80)

    :param source_dir:
    :param target_dir:
    :param known_unknown_classes:
    :param unknown_unknown_classes:

    :return:
    """

    # Get the class indices for 3 partitions
    all_classes = list(range(1, nb_total_classes+1))
    all_unknown_classes = known_unknown_classes + unknown_unknown_classes
    all_unknown_classes.sort()
    known_known_classes = [x for x in all_classes if x not in all_unknown_classes]

    # Fill all the class indices to 5 digits
    known_unknown_classes = [str(x).zfill(5) for x in known_unknown_classes]
    unknown_unknown_classes = [str(x).zfill(5) for x in unknown_unknown_classes]
    known_known_classes = [str(x).zfill(5) for x in known_known_classes]

    if cp_known_known:
        for one_class in known_known_classes:
            train_valid_source_path = source_dir + "/train/" + one_class
            test_source_path = source_dir + "/val/" + one_class

            if not make_debug_set:
                train_valid_target_path = target_dir + "/train_valid/known_known/" + one_class
                test_target_path = target_dir + "/test/known_known/" + one_class

            else:
                train_valid_target_path = target_dir + "/debug/train_valid/known_known/" + one_class
                test_target_path = target_dir + "/debug/test/known_known/" + one_class

            if not os.path.exists(train_valid_target_path):
                os.mkdir(train_valid_target_path)
                print("Making dir: %s" % train_valid_target_path)

            if not os.path.exists(test_target_path):
                os.mkdir(test_target_path)
                print("Making dir: %s" % test_target_path)

            all_train_valid_images_list = os.listdir(train_valid_source_path)
            all_test_images_list = os.listdir(test_source_path)

            if not make_debug_set:
                train_valid_image_list = all_train_valid_images_list
                test_image_list = all_test_images_list
            else:
                nb_taking_train_valid = int(len(all_train_valid_images_list) * extract_ratio)
                train_valid_image_list = all_train_valid_images_list[:nb_taking_train_valid]

                nb_taking_test = int(len(all_test_images_list) * extract_ratio)
                test_image_list = all_test_images_list[:nb_taking_test]

            for image_path in train_valid_image_list:
                image_source_path = os.path.join(train_valid_source_path, image_path)
                image_target_path = os.path.join(train_valid_target_path, image_path)

                print("copying from %s to %s" % (image_source_path, image_target_path))
                copyfile(image_source_path, image_target_path)

            for image_path in test_image_list:
                image_source_path = os.path.join(test_source_path, image_path)
                image_target_path = os.path.join(test_target_path, image_path)

                print("copying from %s to %s" % (image_source_path, image_target_path))
                copyfile(image_source_path, image_target_path)

    if cp_known_unknown:
        for one_class in known_unknown_classes:
            train_valid_source_path = source_dir + "/train/" + one_class
            test_source_path = source_dir + "/val/" + one_class

            if not make_debug_set:
                train_valid_target_path = target_dir + "/train_valid/known_unknown/" + one_class
                test_target_path = target_dir + "/test/known_unknown/" + one_class
            else:
                train_valid_target_path = target_dir + "/debug/train_valid/known_unknown/" + one_class
                test_target_path = target_dir + "/debug/test/known_unknown/" + one_class

            if not os.path.exists(train_valid_target_path):
                os.mkdir(train_valid_target_path)
                print("Making dir: %s" % train_valid_target_path)

            if not os.path.exists(test_target_path):
                os.mkdir(test_target_path)
                print("Making dir: %s" % test_target_path)

            all_train_valid_images_list = os.listdir(train_valid_source_path)
            all_test_images_list = os.listdir(test_source_path)

            if not make_debug_set:
                train_valid_image_list = all_train_valid_images_list
                test_image_list = all_test_images_list
            else:
                nb_taking_train_valid = int(len(all_train_valid_images_list) * extract_ratio)
                train_valid_image_list = all_train_valid_images_list[:nb_taking_train_valid]

                nb_taking_test = int(len(all_test_images_list) * extract_ratio)
                test_image_list = all_test_images_list[:nb_taking_test]

            for image_path in train_valid_image_list:
                image_source_path = os.path.join(train_valid_source_path, image_path)
                image_target_path = os.path.join(train_valid_target_path, image_path)

                print("copying from %s to %s" % (image_source_path, image_target_path))
                copyfile(image_source_path, image_target_path)

            for image_path in test_image_list:
                image_source_path = os.path.join(test_source_path, image_path)
                image_target_path = os.path.join(test_target_path, image_path)

                print("copying from %s to %s" % (image_source_path, image_target_path))
                copyfile(image_source_path, image_target_path)

    # Unknown_unknown only exists in test
    if cp_unknown_unknown:
        for one_class in unknown_unknown_classes:
            test_source_path = source_dir + "/val/" + one_class

            if not make_debug_set:
                test_target_path = target_dir + "/test/unknown_unknown/" + one_class
            else:
                test_target_path = target_dir + "/debug/test/unknown_unknown/" + one_class

            if not os.path.exists(test_target_path):
                os.mkdir(test_target_path)
                print("Making dir: %s" % test_target_path)

            all_test_images_list = os.listdir(test_source_path)

            if not make_debug_set:
                test_image_list = all_test_images_list
            else:
                nb_taking_test = int(len(all_test_images_list) * extract_ratio)
                test_image_list = all_test_images_list[:nb_taking_test]

            for image_path in test_image_list:
                image_source_path = os.path.join(test_source_path, image_path)
                image_target_path = os.path.join(test_target_path, image_path)

                print("copying from %s to %s" % (image_source_path, image_target_path))
                copyfile(image_source_path, image_target_path)




if __name__ == "__main__":
    # generate_three_partitions(source_dir=original_413_path,
    #                           target_dir=target_path,
    #                           known_unknown_classes=first_round_40_classes,
    #                           unknown_unknown_classes=second_round_38_classes,
    #                           cp_unknown_unknown=True,
    #                           make_debug_set=True)

    # generate_sub_training_set(source_data_path=validation_data_path,
    #                           target_data_path=small_test_413_path,
    #                           extract_ratio=0.01)
    # generate the subset for calculating threshold
    # generate_sub_training_set(source_data_path=training_data_path,
    #                           target_data_path=sub_training_set_path,
    #                           extract_ratio=0.2)

    # Generate training and validation csvs
    # generate_csv(data_dir=training_data_path,
    #                        csv_path=training_csv_save_path)
    # generate_csv(data_dir=validation_data_path,
    #                        csv_path=valid_csv_save_path)

    # Generate test csv
    # generate_csv(data_dir=test_data_path,
    #                        csv_path=test_csv_save_path)

    gen_label_match(first_round=first_round_40_classes,
                    second_round=second_round_38_classes,
                    nb_total_classes=413)