# Processing the prob npy files and get stats
# Author: Jin Huang
# Updated date: 03/18/2021


import numpy as np
import sys
# import pandas as pd
# import matplotlib.pyplot as plt
import statistics
from scipy.special import softmax




# Cross-entropy seed 0 -- test_ce_00
# test_model_dir = "2022-12-18/cross_entropy_only/seed_0"
# epoch = 174

# Cross-entropy seed 1 -- test_ce_01
# test_model_dir = "2022-12-18/cross_entropy_only/seed_1"
# epoch = 199

# Cross-entropy seed 2 -- test_ce_02
# test_model_dir = "2022-12-18/cross_entropy_only/seed_2"
# epoch = 168

# Cross-entropy seed 3 -- test_ce_03
# test_model_dir = "2022-12-18/cross_entropy_only/seed_3"
# epoch = 182

# Cross-entropy seed 4 -- test_ce_04
# test_model_dir = "2022-12-18/cross_entropy_only/seed_4"
# epoch = 174

#*******************************************************************#
# Cross-entropy + sam seed 0 -- test_sam_00
# test_model_dir = "2022-12-18/cross_entropy_1.0_pfm_1.0/seed_0"
# epoch = 136

# Cross-entropy + sam seed 1 -- test_sam_01
# test_model_dir = "2022-12-18/cross_entropy_1.0_pfm_1.0/seed_1"
# epoch = 138

# Cross-entropy + sam seed 2 -- test_sam_02
# test_model_dir = "2022-12-18/cross_entropy_1.0_pfm_1.0/seed_2"
# epoch = 191

# Cross-entropy + sam seed 3 -- test_sam_03
# test_model_dir = "2022-12-18/cross_entropy_1.0_pfm_1.0/seed_3"
# epoch = 136

# Cross-entropy + sam seed 4 -- test_sam_04
# test_model_dir = "2022-12-18/cross_entropy_1.0_pfm_1.0/seed_4"
# epoch = 136

#*******************************************************************#
# All 3 losses seed 0 -- test_all_loss_00
# test_model_dir = "2022-12-19/cross_entropy_1.0_pfm_1.0_exit_1.0/seed_0"
# epoch = 131

# All 3 losses seed 1 -- test_all_loss_01
# test_model_dir = "2022-12-19/cross_entropy_1.0_pfm_1.0_exit_1.0/seed_1"
# epoch = 146

# All 3 losses seed 2 -- test_all_loss_02
# test_model_dir = "2022-12-19/cross_entropy_1.0_pfm_1.0_exit_1.0/seed_2"
# epoch = 121

# All 3 losses seed 3 -- test_all_loss_03
# test_model_dir = "2022-12-19/cross_entropy_1.0_pfm_1.0_exit_1.0/seed_3"
# epoch = 185

# All 3 losses seed 4 -- test_all_loss_04
# test_model_dir = "2022-12-19/cross_entropy_1.0_pfm_1.0_exit_1.0/seed_4"
# epoch = 160


#*******************************************************************#
# TODO: Psyphy weight 1-0-1 seed 0
# test_model_dir = "2022-12-28/cross_entropy_1.0_pfm_0.0_exit_1.0/seed_0"
# epoch =

# TODO: Psyphy weight 1-0-1 seed 1
# test_model_dir = "2022-12-28/cross_entropy_1.0_pfm_0.0_exit_1.0/seed_1"
# epoch =

# Psyphy weight 1-0-1 seed 2
# test_model_dir = "2022-12-28/cross_entropy_1.0_pfm_0.0_exit_1.0/seed_2"
# epoch = 124

# Psyphy weight 1-0-1 seed 3
# test_model_dir = "2022-12-28/cross_entropy_1.0_pfm_0.0_exit_1.0/seed_3"
# epoch = 190

# Psyphy weight 1-0-1 seed 4
# test_model_dir = "2022-12-28/cross_entropy_1.0_pfm_0.0_exit_1.0/seed_4"
# epoch = 184

#*******************************************************************#
# Psyphy weight 1-1-2 seed 0
# test_model_dir = "2022-12-28/cross_entropy_1.0_pfm_1.0_exit_2.0/seed_0"
# epoch = 175

# Psyphy weight 1-1-2 seed 1
# test_model_dir = "2022-12-28/cross_entropy_1.0_pfm_1.0_exit_2.0/seed_1"
# epoch = 159

# TODO: Psyphy weight 1-1-2 seed 2
# test_model_dir = None

# Psyphy weight 1-1-2 seed 3
# test_model_dir = "2022-12-28/cross_entropy_1.0_pfm_1.0_exit_2.0/seed_3"
# epoch = 182

# TODO:Psyphy weight 1-1-2 seed 4
test_model_dir = "2022-12-28/cross_entropy_1.0_pfm_1.0_exit_2.0/seed_4"
epoch = 117

#*******************************************************************#
# TODO: Psyphy weight 1-2-1 seed 0
# test_model_dir = ""

# TODO: Psyphy weight 1-2-1 seed 1
# test_model_dir = ""

# TODO: Psyphy weight 1-2-1 seed 2
# test_model_dir = ""

# TODO: Psyphy weight 1-2-1 seed 3
# test_model_dir = "2022-12-28/cross_entropy_1.0_pfm_2.0_exit_1.0/seed_3"

# TODO:Psyphy weight 1-2-1 seed 4
# test_model_dir = ""

####################################################################
# Parameters (usually, no need to change these)
####################################################################
percentile = [50]
test_binary = False
nb_training_classes = 293
# save_path_base = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/models/msd_net"
save_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/openset_resnet"

valid_prob_dir = save_path_base + "/" + test_model_dir + "/features"
test_result_dir = save_path_base + "/" + test_model_dir + "/test_results"


####################################################################
# functions
####################################################################
# TODO: Maybe using top-3 and top-5 as well
def get_known_exit_stats(labels,
                         probs,
                         class_threshold,
                         novelty_threshold,
                         nb_clfs=5):
    nb_correct_top_1 = 0
    nb_correct_top_3 = 0
    nb_correct_top_5 = 0

    known_as_unknown_count = 0

    for i in range(len(labels)):
        target_label = labels[i]
        prob = probs[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]

            # TODO: Adding top3 and top5
            top_1 = np.argmax(one_prob)
            top_3 = np.argpartition(one_prob, -3)[-3:]
            top_5 = np.argpartition(one_prob, -5)[-5:]

            # If this is not the last classifier
            if j != nb_clfs - 1:
                # Case 1: At a certain exit, max prob is larger than both
                #         classification threshold and novelty threshold
                if (top_1 > novelty_threshold[j]) and (top_1 > class_threshold[j]):

                    if top_1 == target_label:
                        nb_correct_top_1 += 1
                        nb_correct_top_3 += 1
                        nb_correct_top_5 += 1
                        break

                    elif target_label in top_3:
                        nb_correct_top_3 += 1
                        nb_correct_top_5 += 1
                        break

                    elif target_label in top_5:
                        nb_correct_top_5 += 1
                        break

                    else:
                        continue

                # Case 2: max prob is smaller than novelty threshold or
                #         is smaller than classification threshold, go check next exit
                elif (top_1 < novelty_threshold[j]) or (top_1 < class_threshold[j]):
                    continue

            # If this is the last classifier
            else:
                if (top_1 > novelty_threshold[j]) and (top_1 > class_threshold[j]):
                    if top_1 == target_label:
                        nb_correct_top_1 += 1
                        nb_correct_top_3 += 1
                        nb_correct_top_5 += 1

                    elif target_label in top_3:
                        nb_correct_top_3 += 1
                        nb_correct_top_5 += 1

                    elif target_label in top_5:
                        nb_correct_top_5 += 1

                    else:
                        pass

                elif (top_1 < novelty_threshold[j]):
                    known_as_unknown_count += 1


    acc_top_1 = float(nb_correct_top_1) / (float(len(labels)))
    acc_top_3 = float(nb_correct_top_3) / (float(len(labels)))
    acc_top_5 = float(nb_correct_top_5) / (float(len(labels)))

    unknown_rate = float(known_as_unknown_count) / (float(len(labels)))

    print("Total number of samples: %d" % len(labels))
    print("Accuracy top-1:", acc_top_1)
    print("Accuracy top-3:", acc_top_3)
    print("Accuracy top-5:", acc_top_5)
    print("Known as unknown:", unknown_rate)




def get_unknown_exit_stats(labels,
                           probs,
                           novelty_threshold,
                           nb_clfs=5):

    unknown_as_known_count = 0

    for i in range(len(labels)):
        # Loop thru each sample
        prob = probs[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]
            max_prob = np.sort(one_prob)[-1]

            # print(max_prob)

            # If this is not the last classifier
            if j != nb_clfs - 1:
                # Only consider top-1 if it is not the last classifier
                if max_prob > novelty_threshold[j]:
                    unknown_as_known_count += 1

                    break

                else:
                    # If the max prob is smaller than threshold, check next clf
                    continue

            # If this is the last classifier
            else:
                if max_prob > novelty_threshold[j]:
                    unknown_as_known_count += 1

    print(unknown_as_known_count)
    print(len(labels))

    acc = 1.0 - float(unknown_as_known_count)/(float(len(labels)))

    print("unknown acc:", acc)




def get_thresholds(npy_file_path,
                   percentile):
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

    thresh_0 = np.percentile(np.asarray(prob_clf_0), percentile)
    thresh_1 = np.percentile(np.asarray(prob_clf_1), percentile)
    thresh_2 = np.percentile(np.asarray(prob_clf_2), percentile)
    thresh_3 = np.percentile(np.asarray(prob_clf_3), percentile)
    thresh_4 = np.percentile(np.asarray(prob_clf_4), percentile)

    return [thresh_0, thresh_1, thresh_2, thresh_3, thresh_4]




def calculate_mcc(true_pos,
                  true_neg,
                  false_pos,
                  false_neg):
    """

    :param true_pos:
    :param true_neg:
    :param false_pos:
    :param false_negtive:
    :return:
    """

    return (true_neg*true_pos-false_pos*false_neg)/np.sqrt((true_pos+false_pos)*(true_pos+false_neg)*
                                                           (true_neg+false_pos)*(true_neg+false_neg))




def get_binary_results(known_feature,
                       known_label,
                       unknown_feature,
                       threshold,
                       nb_clfs=5):
    """

    :param original_feature:
    :param aug_feature:
    :param labels:
    :return:
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    correct = 0
    wrong = 0

    # Process known samples
    for i in range(len(known_label)):
        target_label = known_label[i]
        prob = known_feature[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]
            one_target = target_label
            top_1 = np.argmax(one_prob)

            # If this is not the first classifier
            if j != nb_clfs - 1:
                if (top_1 > threshold[j]):
                    if top_1 == one_target:
                        correct += 1
                        true_positive += 1
                        break
                    else:
                        continue

                elif (top_1 < threshold[j]):
                    continue

            # If this is the last classifier
            else:
                if (top_1 > threshold[j]):
                    if top_1 == one_target:
                        correct += 1
                        true_positive += 1

                    else:
                        true_positive += 1

                elif (top_1 < threshold[j]):
                    wrong += 1
                    false_negative += 1

    # Process unknown
    for i in range(len(unknown_feature)):
        prob = unknown_feature[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]
            max_prob = np.sort(one_prob)[-1]

            # If this is not the last classifier
            if j != nb_clfs - 1:
                if max_prob > threshold[j]:
                    false_positive += 1
                    break
                else:
                    continue

            # If this is the last classifier
            else:
                if max_prob > threshold[j]:
                    false_positive += 1
                else:
                    true_negative += 1

    # Calculate all metrics
    precision = float(true_positive) / float(true_positive + false_positive)
    recall = float(true_positive) / float(true_positive + false_negative)
    f1 = (2 * precision * recall) / (precision + recall)
    mcc = calculate_mcc(true_pos=float(true_positive),
                        true_neg=float(true_negative),
                        false_pos=float(false_positive),
                        false_neg=float(false_negative))
    unknown_acc = float(true_negative)/float(true_negative+false_positive)
    known_acc = float(correct)/float(correct+wrong)

    print("True positive: ", true_positive)
    print("True negative: ", true_negative)
    print("False postive: ", false_positive)
    print("False negative: ", false_negative)
    print("known accuracy: ", known_acc)
    print("Unknown accuracy: ", unknown_acc)
    print("F-1 score: ", f1)
    print("MCC score: ", mcc)



if __name__ == '__main__':
    ################################################################
    # Find valid probs
    ################################################################
    valid_known_known_probs_path = valid_prob_dir + "/valid_known_known_epoch_" + str(epoch) + "_probs.npy"
    valid_known_unknown_probs_path = valid_prob_dir + "/valid_known_unknown_epoch_" + str(epoch) + "_probs.npy"

    valid_known_known_probs = np.load(valid_known_known_probs_path)
    valid_known_unknown_probs = np.load(valid_known_unknown_probs_path)

    ################################################################
    # known known is split into 4 parts
    ################################################################
    # known_known (test)
    test_known_known_probs_path_p0 = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_0_probs.npy"
    test_known_known_label_path_p0  = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_0_labels.npy"
    # test_known_known_rt_path_p0  = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_0_rts.npy"
    test_known_known_labels_p0  = np.load(test_known_known_label_path_p0)
    test_known_known_probs_p0  = np.load(test_known_known_probs_path_p0)
    # test_known_known_rts_p0  = np.load(test_known_known_rt_path_p0)

    test_known_known_probs_path_p1 = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_1_probs.npy"
    test_known_known_label_path_p1 = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_1_labels.npy"
    # test_known_known_rt_path_p1 = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_1_rts.npy"
    test_known_known_labels_p1 = np.load(test_known_known_label_path_p1)
    test_known_known_probs_p1 = np.load(test_known_known_probs_path_p1)
    # test_known_known_rts_p1 = np.load(test_known_known_rt_path_p1)

    test_known_known_probs_path_p2 = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_2_probs.npy"
    test_known_known_label_path_p2 = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_2_labels.npy"
    # test_known_known_rt_path_p2 = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_2_rts.npy"
    test_known_known_labels_p2 = np.load(test_known_known_label_path_p2)
    test_known_known_probs_p2 = np.load(test_known_known_probs_path_p2)
    # test_known_known_rts_p2 = np.load(test_known_known_rt_path_p2)

    test_known_known_probs_path_p3 = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_3_probs.npy"
    test_known_known_label_path_p3 = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_3_labels.npy"
    # test_known_known_rt_path_p3 = test_result_dir + "/test_known_known_epoch_" + str(epoch) + "_part_3_rts.npy"
    test_known_known_labels_p3 = np.load(test_known_known_label_path_p3)
    test_known_known_probs_p3 = np.load(test_known_known_probs_path_p3)
    # test_known_known_rts_p3 = np.load(test_known_known_rt_path_p3)

    test_known_known_probs = np.concatenate((test_known_known_probs_p0, test_known_known_probs_p1,
                                             test_known_known_probs_p2, test_known_known_probs_p3), axis=0)
    test_known_known_labels = np.concatenate((test_known_known_labels_p0,test_known_known_labels_p1,
                                             test_known_known_labels_p2,test_known_known_labels_p3),axis=0)
    # test_known_known_rts = np.concatenate((test_known_known_rts_p0, test_known_known_rts_p1,
    #                                       test_known_known_rts_p2, test_known_known_rts_p3),axis=0)

    print(test_known_known_probs.shape)
    print(test_known_known_labels.shape)


    ################################################################
    # Load known unknown and unknown unknown
    ################################################################
    # known_unknown (test)
    # test_known_unknown_probs_path = test_result_dir + "/known_unknown_epoch_" + str(epoch) + "_probs.npy"
    # test_known_unknown_label_path = test_result_dir + "/known_unknown_epoch_" + str(epoch) + "_labels.npy"
    # test_known_unknown_rt_path = test_result_dir + "/known_unknown_epoch_" + str(epoch) + "_rts.npy"
    #
    # test_known_unknown_labels = np.load(test_known_unknown_label_path)
    # test_known_unknown_probs = np.load(test_known_unknown_probs_path)
    # test_known_unknown_rts = np.load(test_known_unknown_rt_path)

    # unknown_unknown (test)
    test_unknown_unknown_probs_path = test_result_dir + "/unknown_unknown_epoch_" + str(epoch) + "_probs.npy"
    test_unknown_unknown_label_path = test_result_dir + "/unknown_unknown_epoch_" + str(epoch) + "_labels.npy"
    # test_unknown_unknown_rt_path = test_result_dir + "/unknown_unknown_epoch_" + str(epoch) + "_rts.npy"

    test_unknown_unknown_labels = np.load(test_unknown_unknown_label_path)
    test_unknown_unknown_probs = np.load(test_unknown_unknown_probs_path)
    # test_unknown_unknown_rts = np.load(test_unknown_unknown_rt_path)


    for one_perct in percentile:
        print("#" * 50)
        print("Current percentile:", one_perct)
        ################################################################
        # TODO: Get the thresholds for all 3 categories
        ################################################################
        known_known_thresh = get_thresholds(npy_file_path=valid_known_known_probs_path,
                                            percentile=one_perct)
        known_unknown_thresh = get_thresholds(npy_file_path=valid_known_unknown_probs_path,
                                              percentile=one_perct)

        print("known thresholds:", known_known_thresh)
        print("unknown thresholds", known_unknown_thresh)

        ################################################################
        # Run test process
        ################################################################
        # known_known
        print("@" * 40)
        print("Processing known_known samples")
        get_known_exit_stats(labels=test_known_known_labels,
                             probs=test_known_known_probs,
                             class_threshold=known_known_thresh,
                             novelty_threshold=known_unknown_thresh)

        # unknown_unknown
        print("@" * 40)
        print("Processing unknown_unknown samples")
        # get_unknown_exit_stats(labels=test_unknown_unknown_labels,
        #                        probs=test_unknown_unknown_probs,
        #                        novelty_threshold=known_unknown_thresh)

        get_unknown_exit_stats(labels=test_unknown_unknown_labels,
                               probs=test_unknown_unknown_probs,
                               novelty_threshold=known_known_thresh)

        get_binary_results(known_feature=test_known_known_probs,
                           known_label=test_known_known_labels,
                           unknown_feature=test_unknown_unknown_probs,
                           threshold=known_known_thresh)




