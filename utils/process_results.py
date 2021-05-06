# Processing the prob npy files and get stats
# Author: Jin Huang
# Updated date: 03/18/2021


import numpy as np
import sys
# import pandas as pd
# import matplotlib.pyplot as plt
import statistics




test_binary = False
model_type = "pp_full" # original/pp_add/pp_full

# test_epoch_list = [140] # for cross entropy
# test_epoch_list = [174] # for ce + pfm
# test_epoch_list = [190] # for ce + pfm + exit
test_epoch_list = [121]

# This is the path that needs to be changed
# save_path_sub = "2021-04-29/cross_entropy_only" # Cross entropy
# save_path_sub = "2021-04-29/cross_entropy_1.0_pfm_1.5" # CE + pfm
# save_path_sub = "2021-04-29/cross_entropy_1.0_pfm_1.0_exit_2.0" # ce + pfm + exit
save_path_sub = "2021-05-02/cross_entropy_1.0_pfm_3.0_exit_2.0"

##############################################################################################
# Normally, no need to change these paths
save_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models"
nb_training_classes = 296

# NPY files paths
original_valid_known_known_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/0225/original/test/valid_known_known_prob_epoch_141.npy"
original_valid_known_unknown_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/0225/original/test/valid_known_unknown_probs_epoch_141.npy"

# pp_mul_valid_known_known_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/0225/pp_loss/test/valid_known_known_prob_epoch_168.npy"
# pp_mul_valid_known_unknown_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/0225/pp_loss/test/valid_known_unknown_probs_epoch_168.npy"

pp_add_valid_known_known_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/0225/pp_loss_add/test/valid_known_known_prob_epoch_111.npy"
pp_add_valid_known_unknown_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/0225/pp_loss_add/test/valid_known_unknown_probs_epoch_111.npy"


def get_known_exit_stats(original_labels,
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

    for i in range(len(original_labels)):
        # Loop thru each sample
        original_label = original_labels[i]

        if original_label < nb_training_classes-1:
            target_label = original_label
        else:
            target_label = -1

        prob = probs[i]
        rt = rts[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]
            pred = np.argmax(one_prob)
            max_prob = np.sort(one_prob)[-1]

            # If this is not the last classifier
            if j != nb_clfs - 1:
                # Only consider top-1 if it is not the last classifier
                if max_prob > top_1_threshold[j]:
                    # First of all, this sample exits
                    exit_count[j] += 1

                    # Also, this sample is predicted as known no matter
                    # whether the pred label is correct or wrong
                    known_as_known_count += 1

                    if pred == target_label:
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

                if max_prob > top_1_threshold[-1]:
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

    for i in range(len(original_labels)):
        # Loop thru each sample
        prob = probs[i]
        rt = rts[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]
            max_prob = np.sort(one_prob)[-1]

            # If this is not the last classifier
            if j != nb_clfs - 1:
                # Only consider top-1 if it is not the last classifier
                if max_prob > top_1_threshold[j]:
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

                if max_prob > top_1_threshold[-1]:
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





def get_thresholds(npy_file_path):
    """
    Get the probability thresholds for 5 exits respectively.

    :param npy_file_path:
    :return:
    """
    prob_clf_0 = []
    prob_clf_1 = []
    prob_clf_2 = []
    prob_clf_3 = []
    prob_clf_4 = []


    # Load npy file and check the shape
    probs = np.load(npy_file_path)
    print(probs.shape) # Shape [nb_samples, nb_clfs, nb_classes]

    # Process each sample
    for i in range(probs.shape[0]):
        one_sample_probs = probs[i, :, :]
        one_sample_probs = np.reshape(one_sample_probs,(probs.shape[1],
                                                        probs.shape[2]))

        # Check each classifier in each sample
        for j in range(one_sample_probs.shape[0]):
            one_clf_probs = one_sample_probs[j, :]

            # Find the max prob
            max_prob = np.max(one_clf_probs)

            if j == 0:
                prob_clf_0.append(max_prob)
            elif j == 1:
                prob_clf_1.append(max_prob)
            elif j == 2:
                prob_clf_2.append(max_prob)
            elif j == 3:
                prob_clf_3.append(max_prob)
            elif j == 4:
                prob_clf_4.append(max_prob)

    # Sort all the probabilities
    # thresh_0 = statistics.median(prob_clf_0)
    # thresh_1 = statistics.median(prob_clf_1)
    # thresh_2 = statistics.median(prob_clf_2)
    # thresh_3 = statistics.median(prob_clf_3)
    # thresh_4 = statistics.median(prob_clf_4)

    thresh_0 = statistics.mean(prob_clf_0)
    thresh_1 = statistics.mean(prob_clf_1)
    thresh_2 = statistics.mean(prob_clf_2)
    thresh_3 = statistics.mean(prob_clf_3)
    thresh_4 = statistics.mean(prob_clf_4)

    # thresh_0 = np.percentile(np.asarray(prob_clf_0), 10)
    # thresh_1 = np.percentile(np.asarray(prob_clf_1), 10)
    # thresh_2 = np.percentile(np.asarray(prob_clf_2), 10)
    # thresh_3 = np.percentile(np.asarray(prob_clf_3), 10)
    # thresh_4 = np.percentile(np.asarray(prob_clf_4), 10)

    return [thresh_0, thresh_1, thresh_2, thresh_3, thresh_4]





if __name__ == '__main__':
    for epoch_index in test_epoch_list:
        print("Processing data for epoch %d" % epoch_index)

        ################################################################
        # Load all the probs, labels and rts
        ################################################################
        # Known_known
        known_known_probs_path = save_path_base + "/" + save_path_sub + "/test/known_known_probs_epoch_" + str(epoch_index) + ".npy"
        known_known_original_label_path = save_path_base + "/" + save_path_sub + "/test/known_known_labels_epoch_" + str(epoch_index) + ".npy"
        known_known_rt_path = save_path_base + "/" + save_path_sub + "/test/known_known_rts_epoch_" + str(epoch_index) + ".npy"

        known_known_original_labels = np.load(known_known_original_label_path)
        known_known_probs = np.load(known_known_probs_path)
        known_known_rts = np.load(known_known_rt_path)

        # Known_unknown
        known_unknown_probs_path = save_path_base + "/" + save_path_sub + "/test/known_unknown_probs_epoch_" + str(epoch_index) + ".npy"
        known_unknown_original_label_path = save_path_base + "/" + save_path_sub + "/test/known_unknown_labels_epoch_" + str(epoch_index) + ".npy"
        known_unknown_rt_path = save_path_base + "/" + save_path_sub + "/test/known_unknown_rts_epoch_" + str(epoch_index) + ".npy"

        known_unknown_original_labels = np.load(known_unknown_original_label_path)
        known_unknown_probs = np.load(known_unknown_probs_path)
        known_unknown_rts = np.load(known_unknown_rt_path)

        # unknown_unknown
        unknown_unknown_probs_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown_probs_epoch_" + str(epoch_index) + ".npy"
        unknown_unknown_original_label_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown_labels_epoch_" + str(epoch_index) + ".npy"
        unknown_unknown_rt_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown_rts_epoch_" + str(epoch_index) + ".npy"

        unknown_unknown_original_labels = np.load(unknown_unknown_original_label_path)
        unknown_unknown_probs = np.load(unknown_unknown_probs_path)
        unknown_unknown_rts = np.load(unknown_unknown_rt_path)


        ################################################################
        # TODO: Get the thresholds for all 3 categories
        ################################################################
        if model_type == "original":
            print("Getting thrshold for original cross entropy")
            known_known_thresh = get_thresholds(npy_file_path=original_valid_known_known_npy_path)
            known_unknown_thresh = get_thresholds(npy_file_path=original_valid_known_unknown_npy_path)

        elif model_type == "pp_add":
            print("Getting thrshold for pp_add")
            known_known_thresh = get_thresholds(npy_file_path=pp_add_valid_known_known_npy_path)
            known_unknown_thresh = get_thresholds(npy_file_path=pp_add_valid_known_unknown_npy_path)

        elif model_type == "pp_full":
            print("Getting thrshold for pp_full")
            known_known_thresh = get_thresholds(npy_file_path=pp_full_valid_known_known_npy_path)
            known_unknown_thresh = get_thresholds(npy_file_path=pp_full_valid_known_unknown_npy_path)

        else:
            print("Something is wrong")


        print("known thresholds")
        print(known_known_thresh)
        print("unknown thresholds")
        print(known_unknown_thresh)


        ################################################################
        # Run test process
        ################################################################
        if test_binary:
            # Todo: this part needs to be rewritten
            pass

            # print("Current threshold: %f" % threshold)
            #
            # get_known_exit_stats(original_labels=known_known_original_labels,
            #                      probs=known_known_probs,
            #                      rts=known_known_rts,
            #                      top_1_threshold=threshold)
            # print("#" * 30)
            #
            # get_unknown_exit_stats(original_labels=unknown_unknown_original_labels,
            #                      probs=unknown_unknown_probs,
            #                      rts=unknown_unknown_rts,
            #                      top_1_threshold=threshold)

        else:
            # TODO: Use 5 thresholds for test process
            # known_known
            print("@" * 40)
            print("Processing known_known samples")
            get_known_exit_stats(original_labels=known_known_original_labels,
                                 probs=known_known_probs,
                                 rts=known_known_rts,
                                 top_1_threshold=known_known_thresh)
            print("@" * 40)

            # known_unknown
            print("Processing known_unknown samples")
            get_unknown_exit_stats(original_labels=known_unknown_original_labels,
                                   probs=known_unknown_probs,
                                   rts=known_unknown_rts,
                                   top_1_threshold=known_unknown_thresh)
            print("@" * 40)

            # unknown_unknown
            print("Processing unknown_unknown samples")
            get_unknown_exit_stats(original_labels=unknown_unknown_original_labels,
                                   probs=unknown_unknown_probs,
                                   rts=unknown_unknown_rts,
                                   top_1_threshold=known_unknown_thresh)


