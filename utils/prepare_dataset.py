# Prepare the dataset csvs files for MS-DenseNet baseline study
# Initial version date: 07/18/2020
# 07/29/2020: Making a small subset of training set
# Author: Jin Huang


import os
import csv
from shutil import copyfile

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
    Extract a portion of training set for getting novelty threshold.

    :param source_data_path:
    :param target_data_path:
    :param extra_ratio:
    :return:
    """

    for dir, _, _ in os.walk(source_data_path):
        if os.path.basename(os.path.normpath(dir)) == "train":
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


if __name__ == "__main__":
    #
    generate_sub_training_set(source_data_path=training_data_path,
                              target_data_path=sub_training_set_path,
                              extract_ratio=0.2)

    # Generate training and validation csvs
    # generate_csv(data_dir=training_data_path,
    #                        csv_path=training_csv_save_path)
    # generate_csv(data_dir=validation_data_path,
    #                        csv_path=valid_csv_save_path)

    # Generate test csv
    # generate_csv(data_dir=test_data_path,
    #                        csv_path=test_csv_save_path)