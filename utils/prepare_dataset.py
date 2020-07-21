# Prepare the dataset csvs files for MS-DenseNet baseline study
# Initial version date:07/18/2020
# Author: Jin Huang


import os
import csv

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

def gen_training_valid_csv(data_dir, csv_path):
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





def gen_test_csv():
    """

    :return:
    """
    pass




if __name__ == "__main__":
    # gen_training_valid_csv(data_dir=training_data_path,
    #                        csv_path=training_csv_save_path)
    # gen_training_valid_csv(data_dir=validation_data_path,
    #                        csv_path=valid_csv_save_path)
    gen_training_valid_csv(data_dir=test_data_path,
                           csv_path=test_csv_save_path)