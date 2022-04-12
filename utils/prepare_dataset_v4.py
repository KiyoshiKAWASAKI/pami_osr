# Prepare test dataset v4 from UCCS (test 07 and test 08)


import os
import pandas as pd
import cv2
import numpy as np
import json


###################################################
# Data paths
###################################################
subject_tensor = {0:294, 1:63, 2:294, 3:253, 4:294}
object_tensor = {0:294, 1:294, 2:294, 3:109,
                 4:294, 5:294, 6:294, 7:294,
                 8:294, 9:294, 10:253, 11:294}


all_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24"
save_json_path_base = os.path.join(all_path_base, "dataset_v4/json_files")

test_7_csv_path = os.path.join(all_path_base, "test_nd/OND.valtests.0000.0007_single_df.csv")
test_8_csv_path = os.path.join(all_path_base, "test_nd/OND.valtests.0000.0008_single_df.csv")

test_7_save_known_subject_img_path = os.path.join(all_path_base, "dataset_v4/processed_val_test_07/subject_known")
test_7_save_unknown_subject_img_path = os.path.join(all_path_base, "dataset_v4/processed_val_test_07/subject_unknown")
test_7_save_known_object_img_path = os.path.join(all_path_base, "dataset_v4/processed_val_test_07/object_known")
test_7_save_unknown_object_img_path = os.path.join(all_path_base, "dataset_v4/processed_val_test_07/object_unknown")

test_8_save_known_subject_img_path = os.path.join(all_path_base, "dataset_v4/processed_val_test_08/subject_known")
test_8_save_unknown_subject_img_path = os.path.join(all_path_base, "dataset_v4/processed_val_test_08/subject_unknown")
test_8_save_known_object_img_path = os.path.join(all_path_base, "dataset_v4/processed_val_test_08/object_known")
test_8_save_unknown_object_img_path = os.path.join(all_path_base, "dataset_v4/processed_val_test_08/object_unknown")

###################################################
# Define functions
###################################################
def process_csv(csv_path,
                csv_path_base=all_path_base):
    """
    Getting the columns we need:

    img_path, subject_id, object_id,
    subj_ymin, subj_ymax, subj_xmin, subj_xmax,
    obj_ymin, obj_ymax, obj_xmin, obj_xmax

    :param csv_path:
    :return: a dictionary
    """

    csv_data = pd.read_csv(csv_path,
                           usecols=["new_image_path", "subject_id", "object_id",
                                    "subject_ymin", "subject_xmin", "subject_ymax", "subject_xmax",
                                    "object_ymin", "object_xmin", "object_ymax", "object_xmax"])

    all_data_dict = {}

    for i in range(len(csv_data)):
        one_dict = {}

        one_dict["img_path"] = os.path.join(csv_path_base, csv_data.iloc[i]["new_image_path"])
        one_dict["subject_id"] = csv_data.iloc[i]["subject_id"]
        one_dict["object_id"] = csv_data.iloc[i]["object_id"]
        one_dict["subject_ymin"] = csv_data.iloc[i]["subject_ymin"]
        one_dict["subject_xmin"] = csv_data.iloc[i]["subject_xmin"]
        one_dict["subject_ymax"] = csv_data.iloc[i]["subject_ymax"]
        one_dict["subject_xmax"] = csv_data.iloc[i]["subject_xmax"]
        one_dict["object_ymin"] = csv_data.iloc[i]["object_ymin"]
        one_dict["object_xmin"] = csv_data.iloc[i]["object_xmin"]
        one_dict["object_ymax"] = csv_data.iloc[i]["object_ymax"]
        one_dict["object_xmax"] = csv_data.iloc[i]["object_xmax"]

        all_data_dict[i] = one_dict

    return all_data_dict




def save_imgs_and_json(data_dict,
                       save_known_subj_path,
                       save_unknown_subj_path,
                       save_known_obj_path,
                       save_unknown_obj_path,
                       target_shape=224,
                       nb_channel=3):
    """

    :param data_dict:
    :param save_known_subj_path:
    :param save_unknown_subj_path:
    :param save_known_obj_path:
    :param save_unknown_obj_path:
    :param reshape:
    :return:
    """

    subj_known_json_dict = {}
    subj_unknown_json_dict = {}

    obj_known_json_dict = {}
    obj_unknown_json_dict = {}

    for i in range(len(data_dict)):
        one_entry = data_dict[i]

        one_subj_dict = {}
        one_obj_dict = {}

        # Load image
        img_path = one_entry["img_path"]
        img = cv2.imread(img_path)

        subj_label = one_entry["subject_id"]
        obj_label = one_entry["object_id"]

        # TODO: Process subject part
        if subj_label != -1:
            cropped_subj= img[one_entry["subject_ymin"]:one_entry["subject_ymax"],
                              one_entry["subject_xmin"]:one_entry["subject_xmax"]]

            resized_subj = cv2.resize(cropped_subj, (target_shape, target_shape))
            assert resized_subj.shape == (target_shape, target_shape, nb_channel)

            label = subject_tensor[subj_label]

            one_subj_dict["label"] = label
            one_subj_dict["RT"] = None
            one_subj_dict["category"] = "uccs_known"

            # if this is a known sample
            if label != 294:
                subj_img_path = save_known_subj_path + "/processed_" + img_path.split("/")[-1]
                cv2.imwrite(subj_img_path, resized_subj)

                one_subj_dict["img_path"] = subj_img_path
                subj_known_json_dict[len(subj_known_json_dict)] = one_subj_dict

            # if this is an unknown sample
            else:
                subj_img_path = save_unknown_subj_path + "/processed_" + img_path.split("/")[-1]
                cv2.imwrite(subj_img_path, resized_subj)

                one_subj_dict["img_path"] = subj_img_path
                subj_unknown_json_dict[len(subj_unknown_json_dict)] = one_subj_dict


        # TODO: process object part
        if obj_label != -1:
            cropped_obj = img[one_entry["object_ymin"]:one_entry["object_ymax"],
                              one_entry["object_xmin"]:one_entry["object_xmax"]]

            resized_obj = cv2.resize(cropped_obj, (target_shape, target_shape))
            assert resized_obj.shape == (target_shape, target_shape, nb_channel)

            label = object_tensor[obj_label]

            one_obj_dict["label"] = label
            one_obj_dict["RT"] = None
            one_obj_dict["category"] = "uccs_known"

            # if this is a known sample
            if label != 294:
                obj_img_path = save_known_obj_path + "/processed_" + img_path.split("/")[-1]
                cv2.imwrite(obj_img_path, resized_obj)

                one_obj_dict["img_path"] = obj_img_path
                obj_known_json_dict[len(obj_known_json_dict)] = one_obj_dict

            # if this is an unknown sample
            else:
                obj_img_path = save_unknown_obj_path + "/processed_" + img_path.split("/")[-1]
                cv2.imwrite(obj_img_path, resized_obj)

                one_obj_dict["img_path"] = obj_img_path
                obj_unknown_json_dict[len(obj_unknown_json_dict)] = one_obj_dict


    return subj_known_json_dict, subj_unknown_json_dict, \
           obj_known_json_dict, obj_unknown_json_dict




def adjust_index_and_save_json(json_data,
                               save_json_path):
    """
    Adjust the indices for dictionaries:
        (1) Starting from 0 instead of 1
        (2) Use int as indices instead of string
    """

    # Correct training json indices to fit my scheme
    json_dict = {}

    for i in range(len(json_data)):
        one_entry = json_data[i]
        json_dict[int(i+1)] = one_entry

    json_dict = {int(k): v for k, v in json_dict.items()}

    with open(save_json_path, 'w') as f:
        json.dump(json_dict, f)
        print("Saving file to %s" % save_json_path)





if __name__ == '__main__':
    # TODO: Process csv and obtain data for images
    test_7_data_dict = process_csv(test_7_csv_path)
    test_8_data_dict = process_csv(test_8_csv_path)

    # TODO: Save images to the disk (for future use)
    subj_known_json_dict_07, subj_unknown_json_dict_07, \
    obj_known_json_dict_07, obj_unknown_json_dict_07 = save_imgs_and_json(data_dict=test_7_data_dict,
                                                               save_known_subj_path=test_7_save_known_subject_img_path,
                                                               save_unknown_subj_path=test_7_save_unknown_subject_img_path,
                                                               save_known_obj_path=test_7_save_known_object_img_path,
                                                               save_unknown_obj_path=test_7_save_unknown_object_img_path)

    subj_known_json_dict_08, subj_unknown_json_dict_08, \
    obj_known_json_dict_08, obj_unknown_json_dict_08 = save_imgs_and_json(data_dict=test_8_data_dict,
                                                               save_known_subj_path=test_8_save_known_subject_img_path,
                                                               save_unknown_subj_path=test_8_save_unknown_subject_img_path,
                                                               save_known_obj_path=test_8_save_known_object_img_path,
                                                               save_unknown_obj_path=test_8_save_unknown_object_img_path)

    # TODO: Save Json files
    adjust_index_and_save_json(json_data=subj_known_json_dict_07,
                               save_json_path= os.path.join(save_json_path_base, "known_subject_07.json"))
    adjust_index_and_save_json(json_data=subj_unknown_json_dict_07,
                               save_json_path=os.path.join(save_json_path_base, "unknown_subject_07.json"))
    adjust_index_and_save_json(json_data=obj_known_json_dict_07,
                               save_json_path=os.path.join(save_json_path_base, "known_object_07.json"))
    adjust_index_and_save_json(json_data=obj_unknown_json_dict_07,
                               save_json_path=os.path.join(save_json_path_base, "unknown_object_07.json"))

    adjust_index_and_save_json(json_data=subj_known_json_dict_08,
                               save_json_path=os.path.join(save_json_path_base, "known_subject_08.json"))
    adjust_index_and_save_json(json_data=subj_unknown_json_dict_08,
                               save_json_path=os.path.join(save_json_path_base, "unknown_subject_08.json"))
    adjust_index_and_save_json(json_data=obj_known_json_dict_08,
                               save_json_path=os.path.join(save_json_path_base, "known_object_08.json"))
    adjust_index_and_save_json(json_data=obj_unknown_json_dict_08,
                               save_json_path=os.path.join(save_json_path_base, "unknown_object_08.json"))