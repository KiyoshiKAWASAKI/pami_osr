# Map the psyshy RT to the image - updated version with known RTs
# Initial version date: 08/29/2020
# Author: Jin Huang

import numpy as np
import json
import sys
import os
import ast
import pickle
from json import JSONDecoder
from collections import OrderedDict
import itertools
import random
# import pandas as pd

# dataset_v1_3_partition/train_valid:dataset_v1/known_classes/images/train
# dataset_v1_3_partition/test:dataset_v1/known_classes/images/val (we did MTurk data collection on this)


# unknown_rt_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/npy_json_files/unknown.npy"
# known_rt_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/npy_json_files/known.npy"
unknown_rt_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                  "derivatives/dataset_v1_3_partition/npy_json_files/unknown_rt_combined.npy"
known_rt_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                "derivatives/dataset_v1_3_partition/npy_json_files/0212_with_known/known_rt_processed.npy"

"""
02/12/2021 - Split the data and generate json files with known RTs:

RT for 1st round 40 unknown classes: will be used as known_unknown
RT for 40 known classes: will be used as training and validation for known_known

Other 335 classes are known_known

How to prepare the data:
1. Split the first round rt instances into 8:2 according to their class label for training and validation,
    to make sure we have them in both phases.
2. For each class in those 335, split the data into 8:2 for training and validation.
3. Combine step 1 and step 2 => training and validation json
4. Process the second round alone and make the test json

"""

known_classes = [  4,   5,  10,  11,  55,  73,  91, 108, 114, 115,
                133, 135, 141, 142, 144, 154, 155, 156, 162, 163,
                164, 171, 172, 173, 202, 205, 206, 207, 236, 237,
                238, 273, 274, 314, 315, 343, 386, 402, 403, 410]

#################################################################################
# Data directory
#################################################################################
# Data directories.
# Reminder: Data was already switched when doing copy and paste

# known known data
known_known_with_rt_train_val_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data" \
                                     "/dataset_v1_3_partition/train_valid/known_known_with_rt"
known_known_with_rt_test_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                "dataset_v1_3_partition/test/known_known_with_rt"
known_known_without_rt_train_val_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data" \
                                     "/dataset_v1_3_partition/train_valid/known_known_without_rt"
known_known_without_rt_test_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                "dataset_v1_3_partition/test/known_known_without_rt"

# known unknown data
known_unknown_with_rt_train_val_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                       "dataset_v1_3_partition/train_valid/known_unknown"
known_unknown_without_rt_test_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                     "dataset_v1_3_partition/test/known_unknown"

# unknown known (no RT and testing only)
unknown_unknown_without_rt_test_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                       "dataset_v1_3_partition/test/unknown_unknown"


#################################################################################
# Path for saving stuff
#################################################################################
# Paths for saving RT npy
save_known_train_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/" \
                            "image_net/derivatives/dataset_v1_3_partition/npy_json_files_shuffle/known_known_rt_train.npy"
save_known_valid_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/" \
                            "image_net/derivatives/dataset_v1_3_partition/npy_json_files_shuffle/known_known_rt_valid.npy"

save_unknown_train_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/" \
                            "image_net/derivatives/dataset_v1_3_partition/npy_json_files_shuffle/known_unknown_rt_train.npy"
save_unknown_valid_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/" \
                            "image_net/derivatives/dataset_v1_3_partition/npy_json_files_shuffle/known_unknown_rt_valid.npy"


##################################################################################
# Json save path
# known known
train_known_known_with_rt_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                                      "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/train_known_known_with_rt.json"
train_known_known_without_rt_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                                         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/train_known_known_without_rt.json"

valid_known_known_with_rt_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                                      "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/valid_known_known_with_rt.json"
valid_known_known_without_rt_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                                         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/valid_known_known_without_rt.json"

test_known_known_with_rt_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                                     "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/test_known_known_with_rt.json"
test_known_known_without_rt_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                                        "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/test_known_known_without_rt.json"

######################################################################
# known unknown (training and validation have RTs, testing doesnt)
train_known_unknown_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                                        "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/train_known_unknown.json"

valid_known_unknown_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                                        "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/valid_known_unknown.json"

test_known_unknown_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                                          "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/test_known_unknown.json"

######################################################################
# unknown unknown (No RT at all)
test_unknown_unknown_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                                "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/test_unknown_unknown.json"


######################################################################
# TXT paths
save_known_train_txt_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                            "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/known_train_rt.txt"
save_known_valid_txt_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                            "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/known_valid_rt.txt"

save_unknown_train_txt_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                            "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/unknown_train_rt.txt"
save_unknown_valid_txt_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                            "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/unknown_valid_rt.txt"

######################################################################
# Combined Jsons
train_known_known_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                              "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/train_known_known.json"
valid_known_known_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                              "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/valid_known_known.json"
test_known_known_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                              "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/test_known_known.json"

save_split_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                       "dataset_v1_3_partition/npy_json_files_shuffled/"


#################################################################################
# Functions
#################################################################################
def remove_outliers(instance_rt_path,
                    rt_thresh=28.0):
    """

    :param instance_rt_path:
    :param rt_thresh:
    :return:
    """
    # Load instance level RT npy file and sort the entries by class label and convert to second.
    instance_rt = np.load(instance_rt_path)
    instance_rt_sorted = instance_rt[instance_rt[:, 0].argsort()]
    print("There are %d entries." % instance_rt_sorted.shape[0])

    class_labels = instance_rt_sorted[:, 0]
    image_numbers = instance_rt_sorted[:, 1]
    rts = instance_rt_sorted[:, 2].astype(float)
    rts /= 1000

    # Find all the indices where RT is larger than threshold
    remove_index = []
    for i in range(len(rts)):
        if rts[i] > rt_thresh:
            remove_index.append(i)

    # Remove them from the list
    new_class_labels = np.delete(class_labels, remove_index) #40
    new_img_numbers = np.delete(image_numbers,remove_index)
    new_rts = np.delete(rts, remove_index)
    print("There are %d entries after thresholding." % new_class_labels.shape)

    # print("Number of classes:", np.unique(new_class_labels))


    return new_class_labels, new_img_numbers, new_rts




def make_data_dict(class_labels,
                   image_names,
                   rts,
                   category,
                   save_train_dict_path,
                   save_valid_dict_path,
                   train_ratio=0.8):
    """
    Split the RT data into training and validation

    :param class_labels:
    :param image_names:
    :param rts:
    :return:
    """
    # Count number of entries for each class
    class_label_list, class_index, class_counts = np.unique(class_labels,
                                               return_index=True,
                                               return_counts=True)
    print(class_label_list)
    print(class_index)
    print(class_counts)


    # Convert numpy input into list
    class_label_full_list = list(class_labels)
    image_name_list = list(image_names)
    rt_list = list(rts)


    # Get the counts for training and validation
    training_counts = []
    validation_counts = []

    for count in class_counts:
        nb_training = int(count * train_ratio)
        nb_valid = count - nb_training

        training_counts.append(nb_training)
        validation_counts.append(nb_valid)

    # Split everything into training and validation first
    training_class_labels = []
    validation_class_labels = []

    training_img_names = []
    validation_img_names = []

    training_rts = []
    validation_rts = []

    count_train = 0
    count_valid = 0

    # index_list = list(range(len(class_label_full_list)))
    # random.shuffle(index_list)

    print(len(class_label_full_list))
    # sys.exit()

    for i in range(len(class_label_full_list)):
    # for i in index_list:
        class_label = class_label_full_list[i]
        img_name = image_name_list[i]
        rt = rt_list[i]
        # print("Class: %s" % class_label)

        # Find the counts first
        nb_training_sample = training_counts[list(class_label_list).index(class_label)]
        nb_validation_sample = validation_counts[list(class_label_list).index(class_label)]

        if (count_train != nb_training_sample) and (count_valid != nb_validation_sample):
            # print("CASE 1")
            training_class_labels.append(class_label)
            training_img_names.append(img_name)
            training_rts.append(rt)

            count_train += 1

        elif (count_train == nb_training_sample) and (count_valid != nb_validation_sample - 1):
            # print("CASE 2")
            validation_class_labels.append(class_label)
            validation_img_names.append(img_name)
            validation_rts.append(rt)

            count_valid += 1

        elif (count_train == nb_training_sample) and (count_valid == nb_validation_sample - 1):
            # print("CASE 3")
            print("Class: %s" % class_label)
            print("%d training samples for this class." % nb_training_sample)
            print("%d validation samples for this class." % nb_validation_sample)

            count_train = 0
            count_valid = 0

        else:
            print("CASE 4: something is wrong")
            sys.exit(0)

    # Initialize the dictionaries
    training_dict = {}
    valid_dict = {}

    for i in range(len(training_class_labels)):
        training_dict[i] = None
    for i in range(len(validation_class_labels)):
        valid_dict[i] = None

    # Assign value to all dictionary entries
    for i in range(len(training_class_labels)):
        one_dict = {}

        # key_list = ["class_label", "image_name", "rt", "category"]
        # for k in key_list:
        one_dict["class_label"] = training_class_labels[i]
        one_dict["image_name"] = training_img_names[i]
        one_dict["rt"] = training_rts[i]
        one_dict["category"] = category

        training_dict[i] = one_dict

    for i in range(len(validation_class_labels)):
        one_dict = {}

        # key_list = ["class_label", "image_name", "rt", "category"]
        # for k in key_list:
        one_dict["class_label"] = validation_class_labels[i]
        one_dict["image_name"] = validation_img_names[i]
        one_dict["rt"] = validation_rts[i]
        one_dict["category"] = category

        valid_dict[i] = one_dict

    np.save(save_train_dict_path, training_dict)
    np.save(save_valid_dict_path, valid_dict)



def process_npy(training_rt_dict_path,
                valid_rt_dict_path,
                save_train_txt_path,
                save_valid_txt_path,
                rt_upper_bound=20.00):
    """
    The npy was not saved in a good format, so process the dict to make it easier
    to match the data.

    :param training_rt_dict_path:
    :param valid_rt_dict_path:
    :return:
    """
    # Load instance level RT npy file for training and validation (these are dictionaries)
    training_rt_dict = np.load(training_rt_dict_path, allow_pickle=True).item()
    valid_rt_dict = np.load(valid_rt_dict_path, allow_pickle=True).item()

    training_rt_dict = ast.literal_eval(json.dumps(training_rt_dict))
    valid_rt_dict = ast.literal_eval(json.dumps(valid_rt_dict))

    training_list = []
    valid_list = []

    # Use python list instead
    training_class_label_img = []
    training_rt = []

    valid_class_label_img = []
    valid_rt = []


    for i in range(len(training_rt_dict)):
        one_entry = training_rt_dict[str(i)]

        # Append all the result: one Rt for one image
        training_class_label_img.append(one_entry["class_label"]+"/"+one_entry["image_name"])
        training_rt.append(one_entry["rt"])

    # Find the unique class+image set
    training_class_label_img_np = np.asarray(training_class_label_img)
    training_class_label_img_set = list(set(training_class_label_img))

    print("Processing training data")
    for one_img in training_class_label_img_set:
        rts = []
        indices = np.where(training_class_label_img_np == one_img)

        for i, index_list in enumerate(indices):
            for index in index_list:
                rts.append(training_rt[int(index)])

        total_rt = 0
        rt_count = 0
        for one_rt in rts:
            if one_rt <= rt_upper_bound:
                total_rt += one_rt
                rt_count += 1
            else:
                pass

        try:
            rt_average = float(total_rt/rt_count)
            training_list.append([one_img, rt_average])

        except:
            print("No valid RT for this image")
            print(one_img)
            continue


        # print("Processed one training entry: There are %d RTs for this image, avg RT %f." % (len(rts), rt_average))
        training_list.append([one_img, rt_average])


    for i in range(len(valid_rt_dict)):
        one_entry = valid_rt_dict[str(i)]

        valid_class_label_img.append(one_entry["class_label"]+"/"+one_entry["image_name"])
        valid_rt.append(one_entry["rt"])

    valid_class_label_img_np = np.asarray(valid_class_label_img)
    valid_class_label_img_set = list(set(valid_class_label_img))

    print("processing validation data")
    for one_img in valid_class_label_img_set:
        rts = []
        indices = np.where(valid_class_label_img_np == one_img)

        for i, index_list in enumerate(indices):
            for index in index_list:
                rts.append(valid_rt[int(index)])

        total_rt = 0
        rt_count = 0
        for one_rt in rts:
            if one_rt <= rt_upper_bound:
                total_rt += one_rt
                rt_count += 1
            else:
                pass

        try:
            rt_average = float(total_rt/rt_count)
            valid_list.append([one_img, rt_average])
        except:
            print("No valid RT for this image")
            print(one_img)
            continue


        # print("Processed one valid entry: There are %d RTs for this image, avg RT %f." % (len(rts), rt_average))
        # valid_list.append([one_img, rt_average])


    # Save the data into files...
    with open(save_train_txt_path, 'wb') as fp:
        pickle.dump(training_list, fp)

    with open(save_valid_txt_path, 'wb') as fp:
        pickle.dump(valid_list, fp)




def gen_known_known_json(train_valid_known_known_dir,
                         test_known_known_dir,
                         save_train_path,
                         save_valid_path,
                         save_test_path,
                         gen_train_valid=False,
                         gen_test=False,
                         training_ratio=0.8):
    """
    This is for the part without RTs

    :param train_valid_known_known_dir:
    :param test_known_known_dir:
    :param save_train_path:
    :param save_valid_path:
    :return:
    """
    # Initialize the dicts
    train_known_known_dict = {}
    valid_known_known_dict = {}
    test_known_known_dict = {}

    # Set the label for the class: the labels in training has to be continuous integers starting from 0
    label = 0

    # Loop thru all the folders in training first: for training and validation data
    if gen_train_valid:
        for path, subdirs, files in os.walk(train_valid_known_known_dir):
            print("Processing folder: %s" % path)
            print("Number of files", len(files))

            training_count = 0
            valid_count = 0

            if len(files) == 0:
                continue
            else:
                nb_training = int(len(files) * training_ratio)
                nb_valid = len(files) - nb_training
                print("There are %d training samples and %d validation samples for this class" % (nb_training, nb_valid))

                random.shuffle(files)

            for one_file_name in files:
                full_path = os.path.join(path, one_file_name)

                one_file_dict = {}
                key_list = ["img_path", "label", "RT", "category"]
                for key in key_list:
                    one_file_dict[key] = None

                if training_count!= nb_training:

                    one_file_dict["img_path"] = full_path
                    one_file_dict["label"] = label
                    one_file_dict["RT"] = None
                    one_file_dict["category"] = "known_known"

                    train_known_known_dict[len(train_known_known_dict)+1] = one_file_dict
                    training_count += 1


                elif training_count == nb_training and valid_count != nb_valid:
                    one_file_dict["img_path"] = full_path
                    one_file_dict["label"] = label
                    one_file_dict["RT"] = None
                    one_file_dict["category"] = "known_known"

                    valid_known_known_dict[len(valid_known_known_dict)+1] = one_file_dict
                    valid_count += 1

                elif training_count == nb_training and valid_count == nb_valid:
                    training_count = 0
                    valid_count = 0

            label += 1

        print("LABEL", label)

        with open(save_train_path, 'w') as train_f:
            json.dump(train_known_known_dict, train_f)
            print("Saving file to %s" % save_train_path)

        with open(save_valid_path, 'w') as valid_f:
            json.dump(valid_known_known_dict, valid_f)
            print("Saving file to %s" % save_valid_path)


    if gen_test:
        for path, subdirs, files in os.walk(test_known_known_dir):
            print("Processing folder: %s" % path)

            if len(files) == 0:
                continue

            else:

                for one_file_name in files:
                    full_path = os.path.join(path, one_file_name)

                    one_file_dict = {}
                    key_list = ["img_path", "label", "RT", "category"]
                    for key in key_list:
                        one_file_dict[key] = None

                    one_file_dict["img_path"] = full_path
                    one_file_dict["label"] = label
                    one_file_dict["RT"] = None
                    one_file_dict["category"] = "known_known"

                    test_known_known_dict[len(test_known_known_dict)] = one_file_dict

                label += 1

        # print(test_known_known_dict)
        with open(save_test_path, 'w') as test_f:
            json.dump(test_known_known_dict, test_f)
            print("Saving file to %s" % save_test_path)




def gen_known_unknown_json(train_list_path,
                           valid_list_path,
                           train_valid_known_unknown_dir,
                           test_known_unknown_dir,
                           save_train_path,
                           save_valid_path,
                           save_test_path,
                           nb_known_classes=293,
                           gen_train_valid=False,
                           gen_test=False,
                           training_ratio=0.80):
    """

    :param training_rt_dict_path:
    :param valid_rt_dict_path:
    :param train_valid_known_known_dir:
    :param test_known_known_dir:
    :param train_valid_known_unknown_dir:
    :param test_known_unknown_dir:
    :param test_unknown_unknown_dir:
    :param training_ratio:
    :return:
    """

    with open(train_list_path, 'rb') as fp:
        training_list = pickle.load(fp)
    with open(valid_list_path, 'rb') as fp:
        valid_list = pickle.load(fp)

    # Initialize the dicts
    train_known_unknown_dict = {}
    valid_known_unknown_dict = {}
    test_known_unknown_dict = {}

    label_list = []

    label = nb_known_classes

    # total_training_count = 0
    # total_valid_count = 0
    # total_test_count = 0

    # Get the list of labels
    for i in range(len(training_list)):
        one_label = training_list[i][0].split("/")[0]
        label_list.append(one_label)
    label_list, label_counts = np.unique(np.asarray(label_list), return_counts=True)
    label_list = label_list.tolist()
    label_counts = label_counts.tolist()

    print(label_list, label_counts)

    if gen_train_valid:
        """
        First, put all the RTs from training list into train dict 
        and valid list into valid dict
        """

        for i in range(len(training_list)):
            one_file_dict = {}
            key_list = ["img_path", "label", "RT", "category"]
            for key in key_list:
                one_file_dict[key] = None

            one_file_dict["img_path"] = os.path.join(train_valid_known_unknown_dir, training_list[i][0])
            one_file_dict["label"] = label
            one_file_dict["RT"] = training_list[i][1]
            one_file_dict["category"] = "known_unknown"

            train_known_unknown_dict[len(train_known_unknown_dict) + 1] = one_file_dict

        # print(train_known_unknown_dict)

        for i in range(len(valid_list)):
            one_file_dict = {}
            key_list = ["img_path", "label", "RT", "category"]
            for key in key_list:
                one_file_dict[key] = None

            one_file_dict["img_path"] = os.path.join(train_valid_known_unknown_dir, valid_list[i][0])
            one_file_dict["label"] = label
            one_file_dict["RT"] = valid_list[i][1]
            one_file_dict["category"] = "known_unknown"

            valid_known_unknown_dict[len(valid_known_unknown_dict) + 1] = one_file_dict

        """
        For the whole directory:

        Loop thru each directory, find the number of samples left apart from the samples that have RT
        Split the rest of them into training and validation

        """
        for path, subdirs, files in os.walk(train_valid_known_unknown_dir):
            print("Processing folder: %s" % path)

            if len(files) == 0:
                continue
            else:
                # Get the class label from path first
                class_label = path.split("/")[-1]
                training_count = 0
                valid_count = 0

            """
            Check whether this class is in the 78 labels.

            If yes, calculate the number of entries, and decide the train/valid number
            If no, directly split into train and valid
            """
            if class_label in label_list:
                # Find how many entries are there in this class
                class_count = label_counts[label_list.index(class_label)]

                # Calculate the nb of images for training and validation
                nb_total_use = len(files) - class_count
                total_training_count = int(nb_total_use * training_ratio)
                total_valid_count = int(nb_total_use - total_training_count)

                # Find all the images that has this label
                img_list = []
                for one_pair in training_list:
                    if one_pair[0].startswith(class_label):
                        img_list.append(one_pair[1])

                for one_file in files:
                    if one_file.split("/")[-1] in img_list:
                        # print("This image has RT, pass")
                        continue
                    else:
                        full_path = os.path.join(path, one_file)

                        # Initialize a dictionary for this image
                        one_file_dict = {}
                        key_list = ["img_path", "label", "RT", "category"]
                        for key in key_list:
                            one_file_dict[key] = None

                        one_file_dict["img_path"] = full_path
                        one_file_dict["label"] = label
                        one_file_dict["RT"] = None
                        one_file_dict["category"] = "known_unknown"

                        if training_count != total_training_count:
                            train_known_unknown_dict[len(train_known_unknown_dict) + 1] = one_file_dict

                        elif (training_count == total_training_count) and (valid_count != total_valid_count):
                            valid_known_unknown_dict[len(valid_known_unknown_dict) + 1] = one_file_dict

                        elif (training_count == total_training_count) and (valid_count == total_valid_count):
                            training_count = 0
                            valid_count = 0

            else:

                # Calculate the nb of images for training and validation
                total_training_count = int(len(files) * training_ratio)
                total_valid_count = int(len(files) - total_training_count)

                for one_file in files:
                    full_path = os.path.join(path, one_file)

                    # Initialize a dictionary for this image
                    one_file_dict = {}
                    key_list = ["img_path", "label", "RT", "category"]
                    for key in key_list:
                        one_file_dict[key] = None

                    one_file_dict["img_path"] = full_path
                    one_file_dict["label"] = label
                    one_file_dict["RT"] = None
                    one_file_dict["category"] = "known_unknown"

                    if training_count != total_training_count:
                        train_known_unknown_dict[len(train_known_unknown_dict) + 1] = one_file_dict

                    elif (training_count == total_training_count) and (valid_count != total_valid_count):
                        valid_known_unknown_dict[len(valid_known_unknown_dict) + 1] = one_file_dict

                    elif (training_count == total_training_count) and (valid_count == total_valid_count):
                        training_count = 0
                        valid_count = 0

        with open(save_train_path, 'w') as train_f:
            json.dump(train_known_unknown_dict, train_f)
            print("Saving file to %s" % save_train_path)

        with open(save_valid_path, 'w') as valid_f:
            json.dump(valid_known_unknown_dict, valid_f)
            print("Saving file to %s" % save_valid_path)

    # For the "test_known_unknown" just save all the images
    elif gen_test:
        for path, subdirs, files in os.walk(test_known_unknown_dir):
            print("Processing folder: %s" % path)

            for one_file_name in files:
                full_path = os.path.join(path, one_file_name)

                one_file_dict = {}
                key_list = ["img_path", "label", "RT", "category"]
                for key in key_list:
                    one_file_dict[key] = None

                one_file_dict["img_path"] = full_path
                one_file_dict["label"] = label
                one_file_dict["RT"] = None
                one_file_dict["category"] = "known_unknown"

                test_known_unknown_dict[len(test_known_unknown_dict) + 1] = one_file_dict

        with open(save_test_path, 'w') as test_f:
            json.dump(test_known_unknown_dict, test_f)
            print("Saving file to %s" % save_test_path)




def gen_unknown_unknown(test_unknown_unknown_dir,
                        save_test_path,
                        nb_known_classes=293):
    """

    :param test_unknown_unknown_dir:
    :param save_test_dir:
    :return:
    """
    test_unknown_unknown_dict = {}

    for path, subdirs, files in os.walk(test_unknown_unknown_dir):
        print("Processing folder: %s" % path)

        for one_file_name in files:
            full_path = os.path.join(path, one_file_name)

            one_file_dict = {}
            key_list = ["img_path", "label", "RT", "category"]
            for key in key_list:
                one_file_dict[key] = None

            one_file_dict["img_path"] = full_path
            one_file_dict["label"] = nb_known_classes + 1
            one_file_dict["RT"] = None
            one_file_dict["category"] = "unknown_unknown"

            test_unknown_unknown_dict[len(test_unknown_unknown_dict) + 1] = one_file_dict

    with open(save_test_path, 'w') as test_f:
        json.dump(test_unknown_unknown_dict, test_f)
        print("Saving file to %s" % save_test_path)





def gen_class_label_map(known_classes_with_rt,
                        nb_known_without_rt=253):
    """

    :param known_classes_with_rt:
    :return:
    """
    label_dict = {}

    for one_class in known_classes_with_rt:
        one_class_label = str(one_class).zfill(5)
        label_dict[one_class_label] = nb_known_without_rt
        nb_known_without_rt += 1

    # print(label_dict)
    return label_dict




def gen_known_known_rt_json(train_list_path,
                            valid_list_path,
                            train_valid_known_known_dir,
                            test_known_known_dir,
                            save_train_path,
                            save_valid_path,
                            save_test_path,
                            class_map,
                            nb_known_classes_without_rt=253,
                            gen_train_valid=False,
                            gen_test=False,
                            training_ratio=0.80):
    """

    :param training_rt_dict_path:
    :param valid_rt_dict_path:
    :param train_valid_known_known_dir:
    :param test_known_known_dir:
    :param train_valid_known_unknown_dir:
    :param test_known_unknown_dir:
    :param test_unknown_unknown_dir:
    :param training_ratio:
    :return:
    """

    with open(train_list_path, 'rb') as fp:
        training_list = pickle.load(fp)
    with open(valid_list_path, 'rb') as fp:
        valid_list = pickle.load(fp)


    # Initialize the dicts
    train_known_known_dict = {}
    valid_known_known_dict = {}
    test_known_known_dict = {}

    full_label_list = []
    full_valid_label_list = []
    # total_training_count = 0
    # total_valid_count = 0
    # total_test_count = 0

    # Get the list of labels
    # full_label_list is the list of all labels (not sorted)
    for i in range(len(training_list)):
        one_label = training_list[i][0].split("/")[0]
        full_label_list.append(one_label)
    label_list, label_counts = np.unique(np.asarray(full_label_list), return_counts=True)
    label_list = label_list.tolist()
    label_counts = label_counts.tolist()


    for j in range(len(valid_list)):
        one_label = valid_list[j][0].split("/")[0]
        full_valid_label_list.append(one_label)

    valid_label_list, valid_label_counts = np.unique(np.asarray(full_valid_label_list), return_counts=True)
    valid_label_list = valid_label_list.tolist()
    valid_label_counts = valid_label_counts.tolist()


    # sys.exit()


    if gen_train_valid:
        """
        First, put all the RTs from training list into train dict 
        and valid list into valid dict
        """

        for i in range(len(training_list)):
            one_file_dict = {}
            key_list = ["img_path", "label", "RT", "category"]
            for key in key_list:
                one_file_dict[key] = None

            label = class_map[full_label_list[i]]

            one_file_dict["img_path"] = os.path.join(train_valid_known_known_dir, training_list[i][0])
            one_file_dict["label"] = label
            one_file_dict["RT"] = training_list[i][1]
            one_file_dict["category"] = "known_known"

            train_known_known_dict[len(train_known_known_dict) + 1] = one_file_dict

        ########################################################################
        #
        ########################################################################
        label_class_pair = []
        try:
            for i in range(len(train_known_known_dict)):
                class_num = train_known_known_dict[i]["img_path"].split("/")[-2]
                one_pair = [train_known_known_dict[i]["label"],class_num]

                if one_pair not in label_class_pair:
                    label_class_pair.append(one_pair)
        except:
            for i in range(len(train_known_known_dict)):
                class_num = train_known_known_dict[i + 1]["img_path"].split("/")[-2]
                one_pair = [train_known_known_dict[i + 1]["label"], class_num]

                if one_pair not in label_class_pair:
                    label_class_pair.append(one_pair)

        print(len(label_class_pair))

        # print(train_known_known_dict)
        # sys.exit()

        ########################################################################
        #
        ########################################################################

        for i in range(len(valid_list)):
            one_file_dict = {}
            key_list = ["img_path", "label", "RT", "category"]
            for key in key_list:
                one_file_dict[key] = None

            label = class_map[full_valid_label_list[i]]



            one_file_dict["img_path"] = os.path.join(train_valid_known_known_dir, valid_list[i][0])
            one_file_dict["label"] = label
            one_file_dict["RT"] = valid_list[i][1]
            one_file_dict["category"] = "known_known"

            valid_known_known_dict[len(valid_known_known_dict) + 1] = one_file_dict

        label_class_pair = []

        try:
            for i in range(len(valid_known_known_dict)):
                class_num = valid_known_known_dict[i]["img_path"].split("/")[-2]
                one_pair = [valid_known_known_dict[i]["label"],class_num]

                if one_pair not in label_class_pair:
                    label_class_pair.append(one_pair)
        except:
            for i in range(len(valid_known_known_dict)):
                class_num = valid_known_known_dict[i + 1]["img_path"].split("/")[-2]
                one_pair = [valid_known_known_dict[i + 1]["label"], class_num]

                if one_pair not in label_class_pair:
                    label_class_pair.append(one_pair)

        print(len(label_class_pair))
        # sys.exit()

        """
        For the whole directory:

        Loop thru each directory, find the number of samples left apart from the samples that have RT
        Split the rest of them into training and validation

        """
        for path, subdirs, files in os.walk(train_valid_known_known_dir):
            print("Processing folder: %s" % path)

            if len(files) == 0:
                continue
            else:
                # Get the class label from path first
                class_label = path.split("/")[-1]
                training_count = 0
                valid_count = 0

            """
            Check whether this class is in the 40 labels.

            If yes, calculate the number of entries, and decide the train/valid number
            If no, directly split into train and valid
            """
            if class_label in label_list:
                # Find how many entries are there in this class
                class_count = label_counts[label_list.index(class_label)]

                # Calculate the nb of images for training and validation
                nb_total_use = len(files) - class_count
                total_training_count = int(nb_total_use * training_ratio)
                total_valid_count = int(nb_total_use - total_training_count)

                # Find all the images that has this label
                img_list = []
                for one_pair in training_list:
                    if one_pair[0].startswith(class_label):
                        img_list.append(one_pair[1])

                for one_file in files:
                    if one_file.split("/")[-1] in img_list:
                        print("This image has RT, pass")
                        continue
                    else:
                        full_path = os.path.join(path, one_file)

                        # Initialize a dictionary for this image
                        one_file_dict = {}
                        key_list = ["img_path", "label", "RT", "category"]
                        for key in key_list:
                            one_file_dict[key] = None

                        label = class_map[class_label]

                        one_file_dict["img_path"] = full_path
                        one_file_dict["label"] = label
                        one_file_dict["RT"] = None
                        one_file_dict["category"] = "known_known"

                        if training_count != total_training_count:
                            train_known_known_dict[len(train_known_known_dict) + 1] = one_file_dict

                        elif (training_count == total_training_count) and (valid_count != total_valid_count):
                            valid_known_known_dict[len(valid_known_known_dict) + 1] = one_file_dict

                        elif (training_count == total_training_count) and (valid_count == total_valid_count):
                            training_count = 0
                            valid_count = 0

            else:

                # Calculate the nb of images for training and validation
                total_training_count = int(len(files) * training_ratio)
                total_valid_count = int(len(files) - total_training_count)

                for one_file in files:
                    full_path = os.path.join(path, one_file)

                    # Initialize a dictionary for this image
                    one_file_dict = {}
                    key_list = ["img_path", "label", "RT", "category"]
                    for key in key_list:
                        one_file_dict[key] = None

                    label = class_map[class_label]

                    one_file_dict["img_path"] = full_path
                    one_file_dict["label"] = label
                    one_file_dict["RT"] = None
                    one_file_dict["category"] = "known_known"

                    if training_count != total_training_count:
                        train_known_known_dict[len(train_known_known_dict) + 1] = one_file_dict

                    elif (training_count == total_training_count) and (valid_count != total_valid_count):
                        valid_known_known_dict[len(valid_known_known_dict) + 1] = one_file_dict

                    elif (training_count == total_training_count) and (valid_count == total_valid_count):
                        training_count = 0
                        valid_count = 0

        with open(save_train_path, 'w') as train_f:
            json.dump(train_known_known_dict, train_f)
            print("Saving file to %s" % save_train_path)

        with open(save_valid_path, 'w') as valid_f:
            json.dump(valid_known_known_dict, valid_f)
            print("Saving file to %s" % save_valid_path)

    # For the "test_known_known" just save all the images
    elif gen_test:
        for path, subdirs, files in os.walk(test_known_known_dir):
            print("Processing folder: %s" % path)

            for one_file_name in files:
                full_path = os.path.join(path, one_file_name)

                label = path.split("/")[-1]

                label = class_map[label]

                one_file_dict = {}
                key_list = ["img_path", "label", "RT", "category"]
                for key in key_list:
                    one_file_dict[key] = None

                one_file_dict["img_path"] = full_path
                one_file_dict["label"] = label
                one_file_dict["RT"] = None
                one_file_dict["category"] = "known_known"

                test_known_known_dict[len(test_known_known_dict) + 1] = one_file_dict

        with open(save_test_path, 'w') as test_f:
            json.dump(test_known_known_dict, test_f)
            print("Saving file to %s" % save_test_path)




def combine_test_json(test_known_known_with_rt_path,
                      test_known_known_without_rt_path,
                      save_test_json_path):
    """

    :param test_known_known_with_rt_path:
    :param test_known_known_without_rt_path:
    :param save_test_json_path:
    :return:
    """
    with open(test_known_known_with_rt_path) as test_known_known_with_rt:
        test_known_known_with_rt_json = json.load(test_known_known_with_rt)
    with open(test_known_known_without_rt_path) as test_known_known_without_rt:
        test_known_known_without_rt_json = json.load(test_known_known_without_rt)

    # Merge Training Jsons
    for i in range(len(test_known_known_without_rt_json)):
        try:
            one_entry = test_known_known_without_rt_json[str(i + 1)]
            test_known_known_with_rt_json[str(len(test_known_known_with_rt_json) + i + 1)] = one_entry
        except Exception as e:
            print(e)
            continue

    print(len(test_known_known_with_rt_json.keys()))
    print(len(test_known_known_with_rt_json.values()))

    test_known_known_dict = {}

    for i in range(len(test_known_known_with_rt_json.keys())):
        one_entry = list(test_known_known_with_rt_json.values())[i]
        test_known_known_dict[i] = one_entry

    labels = []

    for i in range(len(test_known_known_dict)):
        try:
            labels.append(test_known_known_dict[i]["label"])
        except Exception as e:
            print(e)
            continue

    print(np.unique(np.asarray(labels)))

    with open(save_test_json_path, 'w') as f:
        json.dump(test_known_known_dict, f)
        print("Saving file to %s" % save_test_json_path)



def combine_json(train_known_known_with_rt_path,
                 train_known_known_without_rt_path,
                 valid_known_known_with_rt_path,
                 valid_known_known_without_rt_path,
                 save_training_json_path,
                 save_valid_json_path):
    """
    Combine train jsons into 1 file
    Combine valid jsons into 1 file
    Leave test Jsons separate

    :param train_known_known_path:
    :param train_known_unknown_path:
    :param valid_known_known_path:
    :param valid_known_unknown_path:
    :return:
    """
    customdecoder = JSONDecoder(object_pairs_hook=OrderedDict)

    #######################################
    # Load all Json files
    #######################################
    with open(train_known_known_with_rt_path) as train_known_known_with_rt:
        train_known_known_with_rt_json = json.load(train_known_known_with_rt)
    with open(train_known_known_without_rt_path) as train_known_known_without_rt:
        train_known_known_without_rt_json = json.load(train_known_known_without_rt)

    with open(valid_known_known_with_rt_path) as valid_known_known_with_rt:
        valid_known_known_with_rt_json = json.load(valid_known_known_with_rt)
    with open(valid_known_known_without_rt_path) as valid_known_known_without_rt:
        valid_known_known_without_rt_json = json.load(valid_known_known_without_rt)

    print("train_known_known_with_rt_json", len(train_known_known_with_rt_json))
    print("train_known_known_without_rt_json", len(train_known_known_without_rt_json))
    print("valid_known_known_with_rt_json", len(valid_known_known_with_rt_json))
    print("valid_known_known_without_rt_json", len(valid_known_known_without_rt_json))

    #######################################
    # Check whether there is any missing keys in 2 training files
    #######################################
    print("*" * 60)

    # Merge Training Jsons
    for i in range(len(train_known_known_without_rt_json)):
        try:
            # print(i)
            one_entry = train_known_known_without_rt_json[str(i+1)]
            train_known_known_with_rt_json[str(len(train_known_known_with_rt_json) + i + 1)] = one_entry
        except Exception as e:
            print(e)
            continue

    print(len(train_known_known_with_rt_json.keys()))
    print(len(train_known_known_with_rt_json.values()))

    train_known_known_dict = {}

    for i in range(len(train_known_known_with_rt_json.keys())):
        one_entry = list(train_known_known_with_rt_json.values())[i]
        train_known_known_dict[i] = one_entry

    labels = []

    for i in range(len(train_known_known_dict)):
        try:
            labels.append(train_known_known_dict[i]["label"])
        except Exception as e:
            print(e)
            continue

    print(np.unique(np.asarray(labels)))

    with open(save_training_json_path, 'w') as f:
        json.dump(train_known_known_dict, f)
        print("Saving file to %s" % save_training_json_path)

    # Merge valid Jsons
    for i in range(len(valid_known_known_without_rt_json)):
        try:
            one_entry = valid_known_known_without_rt_json[str(i+1)]
            valid_known_known_with_rt_json[str(len(valid_known_known_with_rt_json) + i + 1)] = one_entry
        except Exception as e:
            print(e)
            continue

    print(len(valid_known_known_with_rt_json.keys()))
    print(len(valid_known_known_with_rt_json.values()))

    valid_known_known_dict = {}

    for i in range(len(valid_known_known_with_rt_json.keys())):
        one_entry = list(valid_known_known_with_rt_json.values())[i]
        valid_known_known_dict[i] = one_entry

    labels = []

    for i in range(len(valid_known_known_dict)):
        try:
            labels.append(valid_known_known_dict[i]["label"])
        except Exception as e:
            print(e)
            continue

    print(np.unique(np.asarray(labels)))

    with open(save_valid_json_path, 'w') as f:
        json.dump(valid_known_known_dict, f)
        print("Saving file to %s" % save_valid_json_path)





def adjust_json_index(train_json_path,
                      valid_json_path):
    """
    Adjust the indices for dictionaries:
        (1) Starting from 0 instead of 1
        (2) Use int as indices instead of string

    :param train_json_path:
    :param valid_json_path:
    :param test_known_known_path:
    :param test_known_unknown_path:
    :param test_unknown_unknown_path:
    :return:
    """

    with open(train_json_path) as f_train:
        print(train_json_path)
        train_json = json.load(f_train)

    with open(valid_json_path) as f_valid:
        print(valid_json_path)
        valid_json = json.load(f_valid)

    print(len(train_json))
    print(len(valid_json))

    # Correct training json indices
    train_dict = {}
    for i in range(len(train_json)):
        try:
            one_entry = train_json[str(i+1)]
        except:
            # print(train_json[str(i+2)])
            one_entry = train_json[str(i+2)]
        train_dict[int(i)] = one_entry

    train_dict = {int(k): v for k, v in train_dict.items()}

    with open(train_json_path, 'w') as f:
        json.dump(train_dict, f)
        print("Saving file to %s" % train_json_path)

    # Correct valid json indices
    valid_dict={}
    for i in range(len(valid_json)):
        try:
            one_entry = valid_json[str(i+1)]
        except:
            one_entry = valid_json[str(i+2)]
        valid_dict[int(i)] = one_entry

    valid_dict = {int(k): v for k, v in valid_dict.items()}

    with open(valid_json_path, 'w') as f:
        json.dump(valid_dict, f)
        print("Saving file to %s" % valid_json_path)




def split_json_file(json_path,
                    save_json_path,
                    nb_split=4):
    """
    Split a large json into several small jsons

    :param json_path:
    :return:
    """
    # Load Json file
    with open(json_path) as f_train:
        json_file = json.load(f_train)


    nb_img_per_file = round(len(json_file)/nb_split)
    print("Number of images per file:", nb_img_per_file)


    # Loop thru the Json and save
    test_dict = {}
    count = 0
    json_index = 0

    for i in range(len(json_file)):
        # When we haven't finished on one json
        if count != nb_img_per_file:
            if i != len(json_file) - 1:
                one_entry = json_file[str(i)]
                test_dict[str(i)] = one_entry
                count +=1
            else:
                print("Processing last sub-json")
                # Adjust the indices of the jsons
                final_dict = {}
                for (j, key) in enumerate(test_dict.keys()):
                    final_dict[str(j)] = test_dict[key]

                # Save this sub-json
                save_sub_json_path = save_json_path + "test_known_known_part_" + str(json_index) + ".json"

                with open(save_sub_json_path, 'w') as f:
                    json.dump(final_dict, f)
                    print("Saving file to %s" % save_sub_json_path)


        # After we get a json done
        else:
            print("Processing one sub-json")
            # Adjust the indices of the jsons
            final_dict = {}
            for (j, key) in enumerate(test_dict.keys()):
                final_dict[str(j)] = test_dict[key]

            # Save this sub-json
            save_sub_json_path = save_json_path + "test_known_known_part_" + str(json_index) + ".json"

            with open(save_sub_json_path, 'w') as f:
                json.dump(final_dict, f)
                print("Saving file to %s" % save_sub_json_path)

            # Reset everything
            test_dict = {}
            count = 0
            json_index += 1




if __name__ == '__main__':
    split_json_file(json_path=test_known_known_json_path,
                    save_json_path=save_split_json_path)

    # Process RT file with known and unknown respectively
    # known_class_labels, known_image_names, known_rts = remove_outliers(instance_rt_path=known_rt_path)
    # unknown_class_labels, unknown_image_names, unknown_rts = remove_outliers(instance_rt_path=unknown_rt_path)
    #
    # Make split dictionary for knowns and unknowns
    # make_data_dict(class_labels=known_class_labels,
    #                image_names=known_image_names,
    #                rts=known_rts,
    #                category="known_known",
    #                save_train_dict_path=save_known_train_npy_path,
    #                save_valid_dict_path=save_known_valid_npy_path)
    #
    # make_data_dict(class_labels=unknown_class_labels,
    #                image_names=unknown_image_names,
    #                rts=unknown_rts,
    #                category="known_unknown",
    #                save_train_dict_path=save_unknown_train_npy_path,
    #                save_valid_dict_path=save_unknown_valid_npy_path)

    # Process known and unknown npy files
    # process_npy(training_rt_dict_path=save_known_train_npy_path,
    #             valid_rt_dict_path=save_known_valid_npy_path,
    #             save_train_txt_path=save_known_train_txt_path,
    #             save_valid_txt_path=save_known_valid_txt_path)
    #
    # process_npy(training_rt_dict_path=save_unknown_train_npy_path,
    #             valid_rt_dict_path=save_unknown_valid_npy_path,
    #             save_train_txt_path=save_unknown_train_txt_path,
    #             save_valid_txt_path=save_unknown_valid_txt_path)

    # Generate known_known json (the part without RT)
    # gen_known_known_json(train_valid_known_known_dir=known_known_without_rt_train_val_path,
    #                      test_known_known_dir=known_known_without_rt_test_path,
    #                      save_train_path=train_known_known_without_rt_json_path,
    #                      save_valid_path=valid_known_known_without_rt_json_path,
    #                      save_test_path=test_known_known_without_rt_json_path,
    #                      gen_train_valid=True,
    #                      gen_test=False,
    #                      training_ratio=0.8)

    # gen_known_known_json(train_valid_known_known_dir=known_known_without_rt_train_val_path,
    #                      test_known_known_dir=known_known_without_rt_test_path,
    #                      save_train_path=train_known_known_without_rt_json_path,
    #                      save_valid_path=valid_known_known_without_rt_json_path,
    #                      save_test_path=test_known_known_without_rt_json_path,
    #                      gen_train_valid=False,
    #                      gen_test=True,
    #                      training_ratio=0.8)

    # Generate known_unknown json
    # gen_known_unknown_json(train_list_path=save_unknown_train_txt_path,
    #                        valid_list_path=save_unknown_valid_txt_path,
    #                        train_valid_known_unknown_dir=known_unknown_with_rt_train_val_path,
    #                        test_known_unknown_dir=known_unknown_without_rt_test_path,
    #                        save_train_path=train_known_unknown_json_path,
    #                        save_valid_path=valid_known_unknown_json_path,
    #                        save_test_path=test_known_unknown_json_path,
    #                        nb_known_classes=293,
    #                        gen_train_valid=True,
    #                        gen_test=False,
    #                        training_ratio=0.80)
    #
    # gen_known_unknown_json(train_list_path=save_unknown_train_txt_path,
    #                        valid_list_path=save_unknown_valid_txt_path,
    #                        train_valid_known_unknown_dir=known_unknown_with_rt_train_val_path,
    #                        test_known_unknown_dir=known_unknown_without_rt_test_path,
    #                        save_train_path=train_known_unknown_json_path,
    #                        save_valid_path=valid_known_unknown_json_path,
    #                        save_test_path=test_known_unknown_json_path,
    #                        nb_known_classes=293,
    #                        gen_train_valid=False,
    #                        gen_test=True,
    #                        training_ratio=0.80)

    # Generate unknown_unknown for test
    # gen_unknown_unknown(test_unknown_unknown_dir=unknown_unknown_without_rt_test_path,
    #                     save_test_path=test_unknown_unknown_json_path,
    #                     nb_known_classes=293)

    # Generate class label matching
    # class_mapping = gen_class_label_map(known_classes_with_rt=known_classes,
    #                                     nb_known_without_rt=253)

    # Generate known_known json (the part with RT)
    # gen_known_known_rt_json(train_list_path=save_known_train_txt_path,
    #                         valid_list_path=save_known_valid_txt_path,
    #                         train_valid_known_known_dir=known_known_with_rt_train_val_path,
    #                         test_known_known_dir=known_known_with_rt_test_path,
    #                         save_train_path=train_known_known_with_rt_json_path,
    #                         save_valid_path=valid_known_known_with_rt_json_path,
    #                         save_test_path=test_known_known_with_rt_json_path,
    #                         class_map=class_mapping,
    #                         nb_known_classes_without_rt=253,
    #                         gen_train_valid=True,
    #                         gen_test=False,
    #                         training_ratio=0.80)

    # gen_known_known_rt_json(train_list_path=save_known_train_txt_path,
    #                         valid_list_path=save_known_valid_txt_path,
    #                         train_valid_known_known_dir=known_known_with_rt_train_val_path,
    #                         test_known_known_dir=known_known_with_rt_test_path,
    #                         save_train_path=train_known_known_with_rt_json_path,
    #                         save_valid_path=valid_known_known_with_rt_json_path,
    #                         save_test_path=test_known_known_with_rt_json_path,
    #                         class_map=class_mapping,
    #                         nb_known_classes_without_rt=253,
    #                         gen_train_valid=False,
    #                         gen_test=True,
    #                         training_ratio=0.80)

    # Combine Json files and adjust the indices
    # combine_json(train_known_known_with_rt_path=train_known_known_with_rt_json_path,
    #              train_known_known_without_rt_path=train_known_known_without_rt_json_path,
    #              valid_known_known_with_rt_path=valid_known_known_with_rt_json_path,
    #              valid_known_known_without_rt_path=valid_known_known_without_rt_json_path,
    #              save_training_json_path=train_known_known_json_path,
    #              save_valid_json_path=valid_known_known_json_path)

    # combine_test_json(test_known_known_with_rt_path=test_known_known_with_rt_json_path,
    #                   test_known_known_without_rt_path=test_known_known_without_rt_json_path,
    #                   save_test_json_path=test_known_known_json_path)

    # Adjust json: known_known
    # adjust_json_index(train_json_path=train_known_known_json_path,
    #                   valid_json_path=valid_known_known_json_path)
    #
    # # Adjust json: known_unknown
    # adjust_json_index(train_json_path=train_known_unknown_json_path,
    #                   valid_json_path=valid_known_unknown_json_path)


