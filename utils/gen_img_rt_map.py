# Map the psyshy RT to the image
# Initial version date: 08/29/2020
# Author: Jin Huang

import numpy as np
import json
import sys
import os




first_round_rt_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/turk_results/instance_rt_processed_first_round.npy"

"""
Split the data and generate json files:

RT for 1st round 40 classes: will be used as training and validation for known_unknown
RT for 2nd round 38 classes: will be used as test for unknown_unknown
Other 335 classes are known_known

How to prepare the data:
1. Split the first round rt instances into 8:2 according to their class label for training and validation,
    to make sure we have them in both phases.
2. For each class in those 335, split the data into 8:2 for training and validation.
3. Combine step 1 and step 2 => training and validation json
4. Process the second round alone and make the test json

"""

# Paths for saving RT npy
save_train_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/first_40_train.npy"
save_valid_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/first_40_valid.npy"

# Data directories.
# Reminder: Data switched for train_valid and test, because we did data collection on val folder.
known_known_train_val_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/train_valid/known_known"
known_known_test_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/test/known_known"
known_unknown_train_val_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/test/known_unknown"
known_unknown_test_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/train_valid/known_unknown"
unknown_unknown_test_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/test/unknown_unknown"

# Json save path
train_known_known_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/train_known_known.json"
valid_known_known_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/valid_known_known.json"
test_known_known_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/test_known_known.json"

train_known_unknown_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/train_known_unknown.json"
valid_known_unknown_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/valid_known_unknown.json"
test_known_unknown_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/test_known_unknown.json"

test_unknown_unknown_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/test_unknown_unknown.json"


def remove_outliers(instance_rt_path,
                    rt_thresh=20.0):
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


    return new_class_labels, new_img_numbers, new_rts




def make_data_dict(class_labels,
                   image_names,
                   rts,
                   category,
                   save_train_dict_path,
                   save_valid_dict_path,
                   train_ratio=0.8):
    """

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

    for i in range(len(class_label_full_list)):
        class_label = class_label_full_list[i]
        img_name = image_name_list[i]
        rt = rt_list[i]
        print("Class: %s" % class_label)

        # Find the counts first
        nb_training_sample = training_counts[list(class_label_list).index(class_label)]
        nb_validation_sample = validation_counts[list(class_label_list).index(class_label)]
        print("%d training samples for this class." % nb_training_sample)
        print("%d validation samples for this class." % nb_validation_sample)

        if (count_train != nb_training_sample) and (count_valid != nb_validation_sample):
            print("CASE 1")
            training_class_labels.append(class_label)
            training_img_names.append(img_name)
            training_rts.append(rt)

            count_train += 1

        elif (count_train == nb_training_sample) and (count_valid != nb_validation_sample):
            print("CASE 2")
            validation_class_labels.append(class_label)
            validation_img_names.append(img_name)
            validation_rts.append(rt)

            count_valid += 1

        elif (count_train == nb_training_sample) and (count_valid == nb_validation_sample):
            print("CASE 3")
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




def gen_known_known_json(train_valid_known_known_dir,
                         test_known_known_dir,
                         save_train_path,
                         save_valid_path,
                         save_test_path,
                         gen_train_valid=False,
                         gen_test=False,
                         training_ratio=0.8):
    """

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
    total_training_count = 0
    total_valid_count = 0
    total_test_count = 0

    # Loop thru all the folders in training first: for training and validation data
    if gen_train_valid:
        for path, subdirs, files in os.walk(train_valid_known_known_dir):
            print("Processing folder: %s" % path)

            training_count = 0
            valid_count = 0

            if len(files) == 0:
                continue
            else:
                nb_training = int(len(files) * training_ratio)
                nb_valid = len(files) - nb_training
                print("There are %d training samples and %d validation samples for this class" % (nb_training, nb_valid))

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

                    train_known_known_dict[total_training_count] = one_file_dict

                    training_count += 1
                    total_training_count += 1

                elif training_count == nb_training and valid_count != nb_valid:
                    one_file_dict["img_path"] = full_path
                    one_file_dict["label"] = label
                    one_file_dict["RT"] = None
                    one_file_dict["category"] = "known_known"

                    valid_known_known_dict[total_valid_count] = one_file_dict

                    valid_count += 1
                    total_valid_count += 1

                elif training_count == nb_training and valid_count == nb_valid:
                    training_count = 0
                    valid_count = 0

            label += 1

            with open(save_train_path, 'w') as train_f:
                json.dump(save_train_path, train_f)
                print("Saving file to %s" % save_train_path)

            with open(save_valid_path, 'w') as valid_f:
                json.dump(save_valid_path, valid_f)
                print("Saving file to %s" % save_valid_path)


    if gen_test:
        for path, subdirs, files in os.walk(test_known_known_dir):
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
                one_file_dict["category"] = "known_known"

                test_known_known_dict[total_training_count] = one_file_dict

                total_test_count += 1

            label += 1

            with open(save_test_path, 'w') as test_f:
                json.dump(save_test_path, test_f)
                print("Saving file to %s" % save_test_path)






def gen_known_unknown_json(training_rt_dict_path,
                           valid_rt_dict_path,
                           train_valid_known_unknown_dir,
                           test_known_unknown_dir,
                           test_unknown_unknown_dir,
                           save_train_path,
                           save_valid_path,
                           save_test_path,
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
    # Load instance level RT npy file for training and validation
    training_rt_nyp = np.load(training_rt_dict_path)
    valid_rt_npy = np.load(valid_rt_dict_path)

    #



    pass



if __name__ == '__main__':
    # class_labels, image_names, rts = remove_outliers(instance_rt_path=first_round_rt_path)

    # make_data_dict(class_labels=class_labels,
    #                image_names=image_names,
    #                rts=rts,
    #                category="known_unknown",
    #                save_train_dict_path=save_train_npy_path,
    #                save_valid_dict_path=save_valid_npy_path)

    # gen_known_known_json(train_valid_known_known_dir=known_known_train_val_path,
    #                      test_known_known_dir=known_known_test_path,
    #                      save_train_path=train_known_known_json_path,
    #                      save_valid_path=valid_known_known_json_path,
    #                      save_test_path=test_known_known_json_path,
    #                      gen_test=True)

