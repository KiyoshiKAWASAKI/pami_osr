


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



path_0 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/train_known_known_with_rt.json"
path_1 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/train_known_known_without_rt.json"

path_2 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/valid_known_known_with_rt.json"
path_3 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/valid_known_known_without_rt.json"


path_4 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/test_known_known_with_rt.json"
path_5 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/test_known_known_without_rt.json"
path_6 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/test_known_unknown.json"
path_7 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/test_unknown_unknown.json"


path_8 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/train_known_known.json"
path_9 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/valid_known_known.json"

path_10 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/train_known_unknown.json"
path_11 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/valid_known_unknown.json"

path_12 = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
         "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/test_known_known.json"


train_rt_file = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/known_train_rt.txt"
valid_rt_file = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/" \
                "derivatives/dataset_v1_3_partition/npy_json_files_shuffle/known_valid_rt.txt"


def check_rt(file_path):
    """

    :param npy_file:
    :return:
    """

    with open(file_path, 'rb') as fp:
        rt = pickle.load(fp)

    print(rt[0])

    classes = []

    for pair in rt:
        classes.append(pair[0].split("/")[0])


    # print(classes)
    print(len(np.unique(np.asarray(classes))))




def check_json_data(json_path):
    """

    :return:
    """
    print("*" * 60)

    with open(json_path) as f:
        data = json.load(f)

    label_class_pair= []

    try:
        for i in range(len(data)):
            class_num = data[str(i)]["img_path"].split("/")[-2]
            one_pair = [data[str(i)]["label"],
                                     class_num]

            if one_pair not in label_class_pair:
                label_class_pair.append(one_pair)
    except:
        for i in range(len(data)):
            class_num = data[str(i+1)]["img_path"].split("/")[-2]
            one_pair = [data[str(i+1)]["label"],
                                     class_num]

            if one_pair not in label_class_pair:
                label_class_pair.append(one_pair)

    # print(np.sort(label_class_pair))
    print(len(data))
    print(len(label_class_pair))




if __name__ == '__main__':

    # check_rt(train_rt_file)
    # check_rt(valid_rt_file)

    # check_json_data(json_path=path_0)
    # check_json_data(json_path=path_1)
    # check_json_data(json_path=path_8)
    # check_json_data(json_path=path_10)

    # check_json_data(json_path=path_2)
    # check_json_data(json_path=path_3)
    # check_json_data(json_path=path_9)
    # check_json_data(json_path=path_11)

    # check_json_data(json_path=path_4)
    # check_json_data(json_path=path_5)
    # check_json_data(json_path=path_6)
    # check_json_data(json_path=path_7)

    check_json_data(json_path=path_12)


