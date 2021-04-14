import sys
import json






# Training data Json save paths
tkk_rt_rt_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                      "npy_json_files/rt_group_json/tkk_rt_rt_grouped.json"
tkk_rt_none_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                        "npy_json_files/rt_group_json/tkk_rt_none_grouped.json"
tkk_no_rt_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                      "npy_json_files/rt_group_json/tkk_no_rt_grouped.json"
tkuk_rt_rt_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                       "npy_json_files/rt_group_json/tkuk_rt_rt_grouped.json"
tkuk_rt_none_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                         "npy_json_files/rt_group_json/tkuk_rt_none_grouped.json"

# Validation data Json save paths
vkk_rt_rt_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                      "npy_json_files/rt_group_json/vkk_rt_rt_grouped.json"
vkk_rt_none_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                        "npy_json_files/rt_group_json/vkk_rt_none_grouped.json"
vkk_no_rt_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                      "npy_json_files/rt_group_json/vkk_no_rt_grouped.json"
vkuk_rt_rt_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                       "npy_json_files/rt_group_json/vkuk_rt_rt_grouped.json"
vkuk_rt_no_rt_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                          "npy_json_files/rt_group_json/vkuk_rt_none_grouped.json"

training_rt_paths = [tkk_rt_rt_save_path, tkk_rt_none_save_path,
                     tkk_no_rt_save_path,
                     tkuk_rt_rt_save_path, tkuk_rt_none_save_path]

valid_json_paths = [vkk_rt_rt_save_path, vkk_rt_none_save_path,
                    vkk_no_rt_save_path,
                    vkuk_rt_rt_save_path, vkuk_rt_no_rt_save_path]

# tkk_paths = [tkk_rt_rt_save_path, tkk_rt_none_save_path, tkk_no_rt_save_path]
# tkuk_paths = [tkuk_rt_rt_save_path, tkuk_rt_none_save_path]
#
# vkk_paths = [vkk_rt_rt_save_path, vkk_rt_none_save_path, vkk_no_rt_save_path]
# vkuk_paths = [vkuk_rt_rt_save_path, vkuk_rt_no_rt_save_path]

# Save Json path
train_known_known_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                              "npy_json_files/rt_group_json/train_known_known.json"
train_known_unknown_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                                "npy_json_files/rt_group_json/train_known_unknown.json"

valid_known_known_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                              "npy_json_files/rt_group_json/valid_known_known.json"
valid_known_unknown_save_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                                "npy_json_files/rt_group_json/valid_known_unknown.json"

if __name__ == '__main__':
    # Check all the keys first # Everything is correct, no missing keys
    # for one_path in (training_rt_paths+valid_json_paths):
    #     with open(one_path) as f:
    #         json_file = json.load(f)
    #
    #     print(one_path)
    #
    #     for i in range(len(json_file)):
    #         try:
    #             one_entry = json_file[str(i)]
    #         except Exception as e:
    #             print(e)

    # # training: Known_known
    # with open(tkk_rt_rt_save_path) as f_1:
    #     tkk_rt_rt = json.load(f_1)
    # with open(tkk_rt_none_save_path) as f_2:
    #     tkk_rt_none = json.load(f_2)
    # with open(tkk_no_rt_save_path) as f_3:
    #     tkk_no_rt = json.load(f_3)
    #
    # print(len(tkk_rt_rt))
    # print(len(tkk_rt_none))
    # print(len(tkk_no_rt))
    #
    # for i in range(len(tkk_rt_none)):
    #     tkk_rt_rt[len(tkk_rt_rt)+i] = tkk_rt_none[str(i)]
    # for j in range(len(tkk_no_rt)):
    #     tkk_rt_rt[len(tkk_rt_rt)+j] = tkk_no_rt[str(j)]
    #
    # tkk_rt_rt_fixed = {str(i): v for i, v in enumerate(tkk_rt_rt.values())}
    # print(len(tkk_rt_rt_fixed))
    #
    # for i in range(len(tkk_rt_rt_fixed)):
    #     try:
    #         one_entry = tkk_rt_rt_fixed[str(i)]
    #     except Exception as e:
    #         print(e)
    #
    # with open(train_known_known_save_path, 'w') as f:
    #     json.dump(tkk_rt_rt_fixed, f)
    #     print("Saving file to %s" % train_known_known_save_path)
    #
    # # training: Known_unknown
    # with open(tkuk_rt_rt_save_path) as f_4:
    #     tkuk_rt_rt = json.load(f_4)
    # with open(tkuk_rt_none_save_path) as f_5:
    #     tkuk_rt_none = json.load(f_5)
    #
    # print(len(tkuk_rt_rt))
    # print(len(tkuk_rt_none))
    #
    # for i in range(len(tkuk_rt_none)):
    #     tkuk_rt_rt[len(tkuk_rt_rt)+i] = tkuk_rt_rt[str(i)]
    #
    # tkuk_rt_rt_fixed = {str(i): v for i, v in enumerate(tkuk_rt_rt.values())}
    # print(len(tkuk_rt_rt_fixed))
    #
    # for i in range(len(tkuk_rt_rt_fixed)):
    #     try:
    #         one_entry = tkuk_rt_rt_fixed[str(i)]
    #     except Exception as e:
    #         print(e)
    #
    # with open(train_known_unknown_save_path, 'w') as f:
    #     json.dump(tkuk_rt_rt_fixed, f)
    #     print("Saving file to %s" % train_known_unknown_save_path)

    # Validation: Known_known
    with open(vkk_rt_rt_save_path) as f_1:
        vkk_rt_rt = json.load(f_1)
    with open(vkk_rt_none_save_path) as f_2:
        vkk_rt_none = json.load(f_2)
    with open(vkk_no_rt_save_path) as f_3:
        vkk_no_rt = json.load(f_3)

    print(len(vkk_rt_rt))
    print(len(vkk_rt_none))
    print(len(vkk_no_rt))

    for i in range(len(vkk_rt_none)):
        vkk_rt_rt[len(vkk_rt_rt) + i] = vkk_rt_none[str(i)]
    print(len(vkk_rt_rt))
    for j in range(len(vkk_no_rt)):
        vkk_rt_rt[len(vkk_rt_rt) + j] = vkk_no_rt[str(j)]
    print(len(vkk_rt_rt))

    vkk_rt_rt_fixed = {str(i): v for i, v in enumerate(vkk_rt_rt.values())}
    print(len(vkk_rt_rt_fixed))

    for i in range(len(vkk_rt_rt_fixed)):
        print(i)
        try:
            one_entry = vkk_rt_rt_fixed[str(i)]
        except Exception as e:
            print(e)

    with open(valid_known_known_save_path, 'w') as f:
        json.dump(vkk_rt_rt_fixed, f)
        print("Saving file to %s" % valid_known_known_save_path)

    # validation: Known_unknown
    # with open(vkuk_rt_rt_save_path) as f_4:
    #     vkuk_rt_rt = json.load(f_4)
    # with open(tkuk_rt_none_save_path) as f_5:
    #     vkuk_rt_none = json.load(f_5)
    #
    # print(len(vkuk_rt_rt))
    # print(len(vkuk_rt_none))
    #
    # for i in range(len(vkuk_rt_none)):
    #     vkuk_rt_rt[len(vkuk_rt_rt)+i] = vkuk_rt_none[str(i)]
    #
    # vkuk_rt_rt_fixed = {str(i): v for i, v in enumerate(vkuk_rt_rt.values())}
    # print(len(vkuk_rt_rt_fixed))
    #
    # for i in range(len(vkuk_rt_rt_fixed)):
    #     try:
    #         one_entry = vkuk_rt_rt_fixed[str(i)]
    #     except Exception as e:
    #         print(e)
    #
    # with open(valid_known_unknown_save_path, 'w') as f:
    #     json.dump(vkuk_rt_rt_fixed, f)
    #     print("Saving file to %s" % valid_known_unknown_save_path)