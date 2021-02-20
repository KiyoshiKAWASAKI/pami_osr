import numpy as np
import sys
# import pandas as pd
# import matplotlib.pyplot as plt

test_binary = True

use_5_weight = True
use_pp = True


# The epoch index of the models
# test_msd_base_epoch = [0, 10, 20, 30, 40, 51, 60, 70, 83, 94]
# test_msd_5_weights_epoch = [0, 10, 46, 50, 60, 70, 80, 90, 95]

# test_msd_base_epoch = [0, 10, 22, 30, 40, 51, 60, 71, 80, 99]
# test_msd_5_weights_epoch = [0, 11, 46, 50, 60, 70, 80, 91]

# test_msd_base_epoch = [0, 11, 21, 30, 40, 50, 60, 72, 81, 90, 99]
# test_msd_5_weights_epoch = [0, 10, 20, 30, 40, 50, 60, 70, 79]

test_msd_base_epoch = [0]
test_msd_5_weights_epoch = [0]
test_msd_pp_epoch = [0]

threshold_list = [0.80, 0.90]


# This is the path that needs to be changed
save_path_sub = "combo_pipeline/0104_binary/msd_5_weights"
# csv_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/combo_pipeline/1214_addition_full_set/msd_base/results.csv"
# fig_save_base_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/combo_pipeline/1214_addition_full_set/msd_base/msd_base/test"


# Normally, no need to change these paths
save_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on"

# known_probs_path = save_path_base + "/" + save_path_sub + "/test/known/probs.npy"
# known_targets_path = save_path_base + "/" + save_path_sub + "/test/known/targets.npy"
# known_original_label_path = save_path_base + "/" + save_path_sub + "/test/known/labels.npy"
# known_rt_path = save_path_base + "/" + save_path_sub + "/test/known/rts.npy"
#
# unknown_probs_path = save_path_base + "/" + save_path_sub + "/test/unknown/probs.npy"
# unknown_targets_path = save_path_base + "/" + save_path_sub + "/test/unknown/targets.npy"
# unknown_original_label_path = save_path_base + "/" + save_path_sub + "/test/unknown/labels.npy"
# unknown_rt_path = save_path_base + "/" + save_path_sub + "/test/unknown/rts.npy"





def get_known_exit_stats(original_labels,
                           target_labels,
                           probs,
                           rts,
                           top_1_threshold,
                           nb_clfs=5):
    """

    :param original_labels:
    :param target_labels:
    :param probs:
    :param rts:
    :param top_1_threshold:
    :param nb_clfs:
    :param nb_training_classes:
    :return:
    """


    known_as_known_count = 0
    known_as_unknown_count = 0

    nb_correct = 0
    nb_wrong = 0

    exit_rt = []

    exit_count = [0] * nb_clfs

    print(np.max(target_labels))
    print(np.min(target_labels))

    for i in range(len(original_labels)):
        # Loop thru each sample
        # original_label = original_labels[i]
        target_label = target_labels[i]
        prob = probs[i]
        rt = rts[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]
            pred = np.argmax(one_prob)
            max_prob = np.sort(one_prob)[-1]
            # print("Max probability:")
            # print(max_prob)

            # If this is not the last classifier
            if j != nb_clfs - 1:
                # Only consider top-1 if it is not the last classifier
                if max_prob > top_1_threshold:
                    # First of all, this sample exits
                    exit_count[j] += 1

                    # Also, this sample is predicted as known no matter
                    # whether the pred label is correct or wrong
                    known_as_known_count += 1

                    # Then, check whether the prediction is right
                    # print("Prediction:")
                    # print(pred)
                    # print("Label:")
                    # print(target_label)

                    if pred == target_label-1:
                        nb_correct += 1
                    else:
                        nb_wrong += 1

                    # Record the RT for this sample
                    exit_rt.append(rt[j])

                    # If top-1 is larger than threshold,
                    # then directly go to next sample
                    break

                else:
                    # If the max prob is smaller than threshold, check next clf
                    continue

            # If this is the last classifier
            else:
                # First of all, this sample exits no matter what...
                exit_count[j] += 1

                if max_prob > top_1_threshold:
                    known_as_known_count += 1

                    # Then, check whether the prediction is right
                    if pred == target_label:
                        nb_correct += 1
                    else:
                        nb_wrong += 1

                else:
                    known_as_unknown_count += 1
                    nb_wrong += 1

                # Record the RT for this sample
                exit_rt.append(rt[j])

    acc = float(nb_correct)/(float(nb_correct)+float(nb_wrong))

    print("Total number of samples: %d" % len(original_labels))
    print("Known predicted as known: %d" % known_as_known_count)
    print("Known predicted as unknown: %d" % known_as_unknown_count)
    print("Number of right prediction: %d" % nb_correct)
    print("Number of wrong prediction: %d" % nb_wrong)
    print("Accuracy: %4f" % acc)
    print("Known exit counts:")
    print(exit_count)

    exit_count_percentage = []
    for one_count in exit_count:
        # print(one_count)
        # print(len(original_labels))
        one_percentage = float(one_count)/float(len(original_labels))
        exit_count_percentage.append(round(one_percentage*100, 2))

    print(exit_count_percentage)


    # Deal with RTs
    exit_rt_np = np.asarray(exit_rt)

    print("Known RT avg:")
    print(np.mean(exit_rt_np))
    print("Known RT median:")
    print(np.median(exit_rt_np))






def get_unknown_exit_stats(original_labels,
                           target_labels,
                           probs,
                           rts,
                           top_1_threshold,
                           nb_clfs=5):
    """

    :param original_labels:
    :param target_labels:
    :param probs:
    :param rts:
    :param top_1_threshold:
    :param nb_clfs:
    :param nb_training_classes:
    :return:
    """

    unknown_as_known_count = 0
    unknown_as_unknown_count = 0

    nb_correct = 0
    nb_wrong = 0

    exit_rt = []
    exit_count = [0] * nb_clfs

    print(np.max(target_labels))
    print(np.min(target_labels))

    for i in range(len(original_labels)):
        # Loop thru each sample
        # original_label = original_labels[i]
        # target_label = target_labels[i]
        prob = probs[i]
        rt = rts[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]
            # pred = np.argmax(one_prob)
            max_prob = np.sort(one_prob)[-1]

            # If this is not the last classifier
            if j != nb_clfs - 1:
                # Only consider top-1 if it is not the last classifier
                if max_prob > top_1_threshold:
                    # First of all, this sample exits
                    exit_count[j] += 1

                    # Also, this sample is predicted as known no matter
                    # whether the pred label is correct or wrong
                    unknown_as_known_count += 1

                    # If max prob is larger than threshold for an unknown sample,
                    # the prediction must be wrong
                    nb_wrong += 1

                    # Record the RT for this sample
                    exit_rt.append(rt[j])

                    # If top-1 is larger than threshold,
                    # then directly go to next sample
                    break

                else:
                    # If the max prob is smaller than threshold, check next clf
                    continue

            # If this is the last classifier
            else:
                # First of all, this sample exits no matter what...
                exit_count[j] += 1

                if max_prob > top_1_threshold:
                    unknown_as_known_count += 1
                    nb_wrong += 1

                else:
                    unknown_as_unknown_count += 1
                    nb_correct += 1

                # Record the RT for this sample
                exit_rt.append(rt[j])

    acc = float(nb_correct)/(float(nb_correct)+float(nb_wrong))

    print("Total number of samples: %d" % len(original_labels))
    print("Unknown predicted as unknown: %d" % unknown_as_unknown_count)
    print("Unknown predicted as known: %d" % unknown_as_unknown_count)
    print("Number of right prediction: %d" % nb_correct)
    print("Number of wrong prediction: %d" % nb_wrong)
    print("Accuracy: %4f" % acc)
    print("Unknown exit counts:")
    print(exit_count)

    exit_count_percentage = []
    for one_count in exit_count:
        one_percentage = float(one_count) / float(len(original_labels))
        exit_count_percentage.append(round(one_percentage*100, 2))

    print(exit_count_percentage)

    # Deal with RTs
    exit_rt_np = np.asarray(exit_rt)

    print("Unknown RT avg:")
    print(np.mean(exit_rt_np))
    print("Unknown RT median:")
    print(np.median(exit_rt_np))




def get_known_exit_binary(original_labels,
                           target_labels,
                           probs,
                           rts,
                           top_1_threshold,
                           nb_clfs=5):
    """

    :param original_labels:
    :param target_labels:
    :param probs:
    :param rts:
    :param top_1_threshold:
    :param nb_clfs:
    :param nb_training_classes:
    :return:
    """


    known_as_known_count = 0
    known_as_unknown_count = 0

    nb_correct = 0
    nb_wrong = 0

    exit_rt = []

    exit_count = [0] * nb_clfs

    # print(np.max(target_labels))
    # print(np.min(target_labels))

    for i in range(len(original_labels)):
        # Loop thru each sample
        # original_label = original_labels[i]
        # target_label = target_labels[i]
        prob = probs[i]
        rt = rts[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]
            pred = np.argmax(one_prob)
            max_prob = np.sort(one_prob)[-1]
            # print("Max probability:")
            # print(max_prob)

            # If this is not the last classifier
            if j != nb_clfs - 1:
                # Only consider top-1 if it is not the last classifier
                if max_prob > top_1_threshold:
                    # First of all, this sample exits
                    exit_count[j] += 1

                    # Also, this sample is predicted as known no matter
                    # whether the pred label is correct or wrong
                    known_as_known_count += 1
                    nb_correct += 1

                    # If top-1 is larger than threshold,
                    # then directly go to next sample
                    break

                else:
                    # If the max prob is smaller than threshold, check next clf
                    nb_wrong += 1
                    continue

            # If this is the last classifier
            else:
                # First of all, this sample exits no matter what...
                exit_count[j] += 1

                if max_prob > top_1_threshold:
                    known_as_known_count += 1
                    nb_correct += 1

                else:
                    known_as_unknown_count += 1
                    nb_wrong += 1

                # Record the RT for this sample
                exit_rt.append(rt[j])

    acc = float(nb_correct)/(float(nb_correct)+float(nb_wrong))

    print("Total number of samples: %d" % len(original_labels))
    print("Known predicted as known: %d" % known_as_known_count)
    print("Known predicted as unknown: %d" % known_as_unknown_count)
    print("Number of right prediction: %d" % nb_correct)
    print("Number of wrong prediction: %d" % nb_wrong)
    print("Accuracy: %4f" % acc)
    print("Known exit counts:")
    print(exit_count)

    exit_count_percentage = []
    for one_count in exit_count:
        # print(one_count)
        # print(len(original_labels))
        one_percentage = float(one_count)/float(len(original_labels))
        exit_count_percentage.append(round(one_percentage*100, 2))

    print(exit_count_percentage)


    # Deal with RTs
    exit_rt_np = np.asarray(exit_rt)

    print("Known RT avg:")
    print(np.mean(exit_rt_np))
    print("Known RT median:")
    print(np.median(exit_rt_np))




def get_unknown_exit_binary(original_labels,
                           target_labels,
                           probs,
                           rts,
                           top_1_threshold,
                           nb_clfs=5):
    """

    :param original_labels:
    :param target_labels:
    :param probs:
    :param rts:
    :param top_1_threshold:
    :param nb_clfs:
    :param nb_training_classes:
    :return:
    """

    unknown_as_known_count = 0
    unknown_as_unknown_count = 0

    nb_correct = 0
    nb_wrong = 0

    exit_rt = []
    exit_count = [0] * nb_clfs

    print(np.max(target_labels))
    print(np.min(target_labels))

    for i in range(len(original_labels)):
        # Loop thru each sample
        # original_label = original_labels[i]
        # target_label = target_labels[i]
        prob = probs[i]
        rt = rts[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]
            # pred = np.argmax(one_prob)
            max_prob = np.sort(one_prob)[-1]

            # If this is not the last classifier
            if j != nb_clfs - 1:
                # Only consider top-1 if it is not the last classifier
                if max_prob > top_1_threshold:
                    # First of all, this sample exits
                    exit_count[j] += 1

                    # Also, this sample is predicted as known no matter
                    # whether the pred label is correct or wrong
                    unknown_as_known_count += 1

                    # If max prob is larger than threshold for an unknown sample,
                    # the prediction must be wrong
                    nb_wrong += 1

                    # Record the RT for this sample
                    exit_rt.append(rt[j])

                    # If top-1 is larger than threshold,
                    # then directly go to next sample
                    break

                else:
                    # If the max prob is smaller than threshold, check next clf
                    continue

            # If this is the last classifier
            else:
                # First of all, this sample exits no matter what...
                exit_count[j] += 1

                if max_prob > top_1_threshold:
                    unknown_as_known_count += 1
                    nb_wrong += 1

                else:
                    unknown_as_unknown_count += 1
                    nb_correct += 1

                # Record the RT for this sample
                exit_rt.append(rt[j])

    acc = float(nb_correct)/(float(nb_correct)+float(nb_wrong))

    print("Total number of samples: %d" % len(original_labels))
    print("Unknown predicted as unknown: %d" % unknown_as_unknown_count)
    print("Unknown predicted as known: %d" % unknown_as_unknown_count)
    print("Number of right prediction: %d" % nb_correct)
    print("Number of wrong prediction: %d" % nb_wrong)
    print("Accuracy: %4f" % acc)
    print("Unknown exit counts:")
    print(exit_count)

    exit_count_percentage = []
    for one_count in exit_count:
        one_percentage = float(one_count) / float(len(original_labels))
        exit_count_percentage.append(round(one_percentage*100, 2))

    print(exit_count_percentage)

    # Deal with RTs
    exit_rt_np = np.asarray(exit_rt)

    print("Unknown RT avg:")
    print(np.mean(exit_rt_np))
    print("Unknown RT median:")
    print(np.median(exit_rt_np))



# def get_threshold_curve(thresholds):
#     """
#
#     :param thresholds:
#     :return:
#     """
#     known_acc_list, unknown_acc_list = [], []
#
#     for thresh in thresholds:
#         known_exit, known_rts, unknown_exit, unknown_rts, predictions = get_exit_stats(original_labels=original_labels,
#                                                                                        target_labels=target_labels,
#                                                                                        probs=probs,
#                                                                                        rts=rts,
#                                                                                        top_1_threshold=thresh)
#
#         known_known, known_unknown, unknown_known, unknown_unknown = 0, 0, 0, 0
#         known_pred = predictions[:len(known_exit)]
#         known_target = target_labels[:len(known_exit)]
#         unknown_pred = predictions[len(known_exit):]
#         unknown_target = target_labels[len(known_exit):]
#
#         for i in range(len(known_pred)):
#             if known_pred[i] != -1:
#                 known_known += 1
#             else:
#                 known_unknown += 1
#
#         for i in range(len(unknown_pred)):
#             if unknown_pred[i] != -1:
#                 unknown_known += 1
#             else:
#                 unknown_unknown += 1
#
#         print(known_known, known_unknown, unknown_known, unknown_unknown)
#
#         known_acc = known_known / (known_known + known_unknown)
#         unknown_acc = unknown_unknown / (unknown_known + unknown_unknown)
#
#         known_acc_list.append(known_acc)
#         unknown_acc_list.append(unknown_acc)
#
#     return known_acc_list, unknown_acc_list



# def




if __name__ == '__main__':
    # Load all the npy files
    if (use_5_weight==False) and (use_pp==False):
        epoch_index_list = test_msd_base_epoch
    elif (use_5_weight==True) and (use_pp==False):
        epoch_index_list = test_msd_5_weights_epoch
    else:
        epoch_index_list = test_msd_pp_epoch


    for epoch_index in epoch_index_list:
        print("Processing data for epoch %d" % epoch_index)

        ###################################################
        # Known_known
        ###################################################
        known_known_probs_path = save_path_base + "/" + save_path_sub + "/test/known_known/probs_epoch_" + str(epoch_index) + ".npy"
        known_known_targets_path = save_path_base + "/" + save_path_sub + "/test/known_known/targets_epoch_" + str(epoch_index) + ".npy"
        known_known_original_label_path = save_path_base + "/" + save_path_sub + "/test/known_known/labels_epoch_" + str(epoch_index) + ".npy"
        known_known_rt_path = save_path_base + "/" + save_path_sub + "/test/known_known/rts_epoch_" + str(epoch_index) + ".npy"

        known_known_original_labels = np.load(known_known_original_label_path)
        known_known_target_labels = np.load(known_known_targets_path)
        known_known_probs = np.load(known_known_probs_path)
        known_known_rts = np.load(known_known_rt_path)

        ###################################################
        # Known_unknown
        ###################################################





        ###################################################
        # unknown_unknown
        ###################################################
        unknown_unknown_probs_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown/probs_epoch_" + str(epoch_index) + ".npy"
        unknown_unknown_targets_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown/targets_epoch_" + str(epoch_index) + ".npy"
        unknown_unknown_original_label_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown/labels_epoch_" + str(epoch_index) + ".npy"
        unknown_unknown_rt_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown/rts_epoch_" + str(epoch_index) + ".npy"


        unknown_unknown_original_labels = np.load(unknown_unknown_original_label_path)
        unknown_unknown_target_labels = np.load(unknown_unknown_targets_path)
        unknown_unknown_probs = np.load(unknown_unknown_probs_path)
        unknown_unknown_rts = np.load(unknown_unknown_rt_path)

        # TODO: add process for testing known_unknown
        if test_binary:
            for threshold in threshold_list:
                print("Current threshold: %f" % threshold)

                get_known_exit_stats(original_labels=known_known_original_labels,
                                     target_labels=known_known_target_labels,
                                     probs=known_known_probs,
                                     rts=known_known_rts,
                                     top_1_threshold=threshold)
                print("#" * 30)

                get_known_exit_stats(original_labels=unknown_unknown_original_labels,
                                     target_labels=unknown_unknown_target_labels,
                                     probs=unknown_unknown_probs,
                                     rts=unknown_unknown_rts,
                                     top_1_threshold=threshold)
                print("*" * 50)

        else:
            for threshold in threshold_list:
                print("Current threshold: %f" % threshold)

                get_known_exit_stats(original_labels=known_known_original_labels,
                                     target_labels=known_known_target_labels,
                                     probs=known_known_probs,
                                     rts=known_known_rts,
                                     top_1_threshold=threshold)
                print("#" * 30)

                get_known_exit_stats(original_labels=unknown_unknown_original_labels,
                                     target_labels=unknown_unknown_target_labels,
                                     probs=unknown_unknown_probs,
                                     rts=unknown_unknown_rts,
                                     top_1_threshold=threshold)
                print("*" * 50)
