# Processing the prob npy files and get stats
# Author: Jin Huang
# Updated date: 03/18/2021


import numpy as np
import sys
# import pandas as pd
# import matplotlib.pyplot as plt
import statistics





#TODO: change this for each model
percentile = [50]

# save_path_sub = "2022-01-13/cross_entropy_only_unknown_ratio_1.0/seed_0"
# epoch = 144

# save_path_sub = "2022-01-13/cross_entropy_1.0_pfm_1.0_unknown_ratio_1.0/seed_0"
# epoch = 193

# save_path_sub = "2022-01-13/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_0"
# epoch = 162

save_path_sub = "2022-01-10/cross_entropy_1.0_pfm_3.0_exit_3.0_unknown_ratio_1.0/seed_0"
epoch = 107

####################################################################
# Paths (usually, no need to change these)
####################################################################
test_binary = False
nb_training_classes = 294
save_path_base = "/scratch365/jhuang24/sail-on/models/msd_net"

valid_prob_dir = save_path_base + "/" + save_path_sub + "/features"
test_result_dir = save_path_base + "/" + save_path_sub + "/test_results"


####################################################################
# functions
####################################################################
# TODO: Maybe using top-3 and top-5 as well
def get_known_exit_stats(labels,
                         probs,
                         rts,
                         top_1_threshold,
                         nb_clfs=5):

    known_as_known_count = 0
    known_as_unknown_count = 0

    nb_correct = 0
    nb_wrong = 0

    exit_rt = []
    exit_count = [0] * nb_clfs

    for i in range(len(labels)):
        original_label = labels[i]

        print("*" * 30)

        target_label = original_label + 1

        prob = probs[i]
        rt = rts[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]
            pred = np.argmax(one_prob)
            max_prob = np.sort(one_prob)[-1]

            print("GT:", target_label)
            print("prediction", pred)

            # If this is not the last classifier
            if j != nb_clfs - 1:
                # Only consider top-1 if it is not the last classifier
                if (max_prob > top_1_threshold[j]) and (pred == target_label):
                    exit_count[j] += 1
                    known_as_known_count += 1
                    nb_correct += 1

                    break

                else:
                    continue

            # If this is the last classifier
            else:
                exit_count[j] += 1

                if max_prob > top_1_threshold[-1]:
                    known_as_known_count += 1

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

    print("Total number of samples: %d" % len(labels))
    print("Known as known: %f" % (float(known_as_known_count)/float(len(labels))))
    print("Known as unknown: %f" % (float(known_as_unknown_count)/float(len(labels))))
    print("Number of right prediction: %d" % nb_correct)
    print("Number of wrong prediction: %d" % nb_wrong)
    print("Accuracy: %4f" % acc)
    print("Known exit counts:")
    print(exit_count)

    exit_count_percentage = []
    for one_count in exit_count:
        one_percentage = float(one_count)/float(len(labels))
        exit_count_percentage.append(round(one_percentage*100, 2))

    print(exit_count_percentage)

    # Deal with RTs
    exit_rt_np = np.asarray(exit_rt)

    print("Known RT avg:")
    print(np.mean(exit_rt_np))
    print("Known RT median:")
    print(np.median(exit_rt_np))




def get_unknown_exit_stats(labels,
                           probs,
                           rts,
                           top_1_threshold,
                           nb_clfs=5):

    unknown_as_known_count = 0
    unknown_as_unknown_count = 0

    nb_correct = 0
    nb_wrong = 0
    exit_rt = []
    exit_count = [0] * nb_clfs

    for i in range(len(labels)):
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

    print("Total number of samples: %d" % len(labels))
    print("Unknown predicted as unknown: %d" % unknown_as_unknown_count)
    print("Unknown predicted as known: %d" % unknown_as_known_count)
    print("Number of right prediction: %d" % nb_correct)
    print("Number of wrong prediction: %d" % nb_wrong)
    print("Accuracy: %4f" % acc)
    print("Unknown exit counts:")
    print(exit_count)

    exit_count_percentage = []
    for one_count in exit_count:
        one_percentage = float(one_count) / float(len(labels))
        exit_count_percentage.append(round(one_percentage*100, 2))

    print(exit_count_percentage)

    # Deal with RTs
    exit_rt_np = np.asarray(exit_rt)

    print("Unknown RT avg:")
    print(np.mean(exit_rt_np))
    print("Unknown RT median:")
    print(np.median(exit_rt_np))





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
    test_known_known_probs_path_p0 = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_0_probs.npy"
    test_known_known_label_path_p0  = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_0_labels.npy"
    test_known_known_rt_path_p0  = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_0_rts.npy"
    test_known_known_labels_p0  = np.load(test_known_known_label_path_p0)
    test_known_known_probs_p0  = np.load(test_known_known_probs_path_p0)
    test_known_known_rts_p0  = np.load(test_known_known_rt_path_p0)

    test_known_known_probs_path_p1 = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_1_probs.npy"
    test_known_known_label_path_p1 = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_1_labels.npy"
    test_known_known_rt_path_p1 = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_1_rts.npy"
    test_known_known_labels_p1 = np.load(test_known_known_label_path_p1)
    test_known_known_probs_p1 = np.load(test_known_known_probs_path_p1)
    test_known_known_rts_p1 = np.load(test_known_known_rt_path_p1)

    test_known_known_probs_path_p2 = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_2_probs.npy"
    test_known_known_label_path_p2 = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_2_labels.npy"
    test_known_known_rt_path_p2 = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_2_rts.npy"
    test_known_known_labels_p2 = np.load(test_known_known_label_path_p2)
    test_known_known_probs_p2 = np.load(test_known_known_probs_path_p2)
    test_known_known_rts_p2 = np.load(test_known_known_rt_path_p2)

    test_known_known_probs_path_p3 = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_3_probs.npy"
    test_known_known_label_path_p3 = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_3_labels.npy"
    test_known_known_rt_path_p3 = test_result_dir + "/known_known_epoch_" + str(epoch) + "_part_3_rts.npy"
    test_known_known_labels_p3 = np.load(test_known_known_label_path_p3)
    test_known_known_probs_p3 = np.load(test_known_known_probs_path_p3)
    test_known_known_rts_p3 = np.load(test_known_known_rt_path_p3)

    test_known_known_probs = np.concatenate((test_known_known_probs_p0, test_known_known_probs_p1,
                                             test_known_known_probs_p2, test_known_known_probs_p3), axis=0)
    test_known_known_labels = np.concatenate((test_known_known_labels_p0,test_known_known_labels_p1,
                                             test_known_known_labels_p2,test_known_known_labels_p3),axis=0)
    test_known_known_rts = np.concatenate((test_known_known_rts_p0, test_known_known_rts_p1,
                                          test_known_known_rts_p2, test_known_known_rts_p3),axis=0)


    ################################################################
    # Load known unknown and unknown unknown
    ################################################################
    # known_unknown (test)
    test_known_unknown_probs_path = test_result_dir + "/known_unknown_epoch_" + str(epoch) + "_probs.npy"
    test_known_unknown_label_path = test_result_dir + "/known_unknown_epoch_" + str(epoch) + "_labels.npy"
    test_known_unknown_rt_path = test_result_dir + "/known_unknown_epoch_" + str(epoch) + "_rts.npy"

    test_known_unknown_labels = np.load(test_known_unknown_label_path)
    test_known_unknown_probs = np.load(test_known_unknown_probs_path)
    test_known_unknown_rts = np.load(test_known_unknown_rt_path)

    # unknown_unknown (test)
    test_unknown_unknown_probs_path = test_result_dir + "/unknown_unknown_epoch_" + str(epoch) + "_probs.npy"
    test_unknown_unknown_label_path = test_result_dir + "/unknown_unknown_epoch_" + str(epoch) + "_labels.npy"
    test_unknown_unknown_rt_path = test_result_dir + "/unknown_unknown_epoch_" + str(epoch) + "_rts.npy"

    test_unknown_unknown_labels = np.load(test_unknown_unknown_label_path)
    test_unknown_unknown_probs = np.load(test_unknown_unknown_probs_path)
    test_unknown_unknown_rts = np.load(test_unknown_unknown_rt_path)


    # print(np.unique(test_known_known_labels))
    # print(np.unique(test_known_unknown_labels))
    # print(np.unique(test_unknown_unknown_labels))
    # sys.exit()


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
        # print("@" * 40)
        print("Processing known_known samples")
        get_known_exit_stats(labels=test_known_known_labels,
                             probs=test_known_known_probs,
                             rts=test_known_known_rts,
                             top_1_threshold=known_known_thresh)

        # known_unknown
        # print("@" * 40)
        # print("Processing known_unknown samples")
        # get_known_exit_stats(labels=test_known_unknown_labels,
        #                        probs=test_known_unknown_probs,
        #                        rts=test_known_unknown_rts,
        #                        top_1_threshold=known_unknown_thresh)

        # # unknown_unknown
        # print("@" * 40)
        # print("Processing unknown_unknown samples")
        # get_unknown_exit_stats(labels=test_unknown_unknown_labels,
        #                        probs=test_unknown_unknown_probs,
        #                        rts=test_unknown_unknown_rts,
        #                        top_1_threshold=known_unknown_thresh)


