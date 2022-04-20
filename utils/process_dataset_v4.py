# Processing the prob npy files and get stats
# Author: Jin Huang
# Updated date: 04/12/2021


import numpy as np
import sys
import statistics



####################################################################
# Model and data paths
####################################################################
# # TODO: Cross-entropy seed 0 -- test_ce_00
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_0"
# model = "cross_entropy_only"
# epoch = 147

# TODO: Cross-entropy + sam seed 0 -- test_sam_00
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_0"
# model = "cross_entropy_pfm"
# epoch = 175

# TODO: All 3 losses seed 0 -- test_all_loss_00
test_model_dir = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_0"
model = "psyphy"
epoch = 156



####################################################################
# Parameters (usually, no need to change these)
####################################################################
percentile = [50]
nb_training_classes = 293

model_path_base = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/models/msd_net"
result_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/test_uccs"

valid_prob_dir = model_path_base + "/" + test_model_dir + "/features"
test_result_dir = result_path_base + "/" + model


####################################################################
# functions
####################################################################
def get_known_exit_stats(labels,
                         probs,
                         class_threshold,
                         novelty_threshold,
                         nb_clfs=5):
    """

    :param labels:
    :param probs:
    :param class_threshold:
    :param novelty_threshold:
    :param nb_clfs:
    :return:
    """
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
    """

    :param labels:
    :param probs:
    :param novelty_threshold:
    :param nb_clfs:
    :return:
    """

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




if __name__ == '__main__':
    ################################################################
    # Find valid probs
    ################################################################
    valid_known_known_probs_path = valid_prob_dir + "/valid_known_known_epoch_" + str(epoch) + "_probs.npy"
    valid_known_unknown_probs_path = valid_prob_dir + "/valid_known_unknown_epoch_" + str(epoch) + "_probs.npy"

    valid_known_known_probs = np.load(valid_known_known_probs_path)
    valid_known_unknown_probs = np.load(valid_known_unknown_probs_path)

    ################################################################
    # TODO: Update UCCS paths
    ################################################################
    # Data for test 07
    known_subject_07_prob = np.load(test_result_dir + "/known_subject_07_epoch_" + str(epoch) + "_probs.npy")
    known_subject_07_label = np.load(test_result_dir + "/known_subject_07_epoch_" + str(epoch) + "_labels.npy")
    unknown_subject_07_prob = np.load(test_result_dir + "/unknown_subject_07_epoch_" + str(epoch) + "_probs.npy")
    unknown_subject_07_label = np.load(test_result_dir + "/unknown_subject_07_epoch_" + str(epoch) + "_labels.npy")

    known_object_07_prob = np.load(test_result_dir + "/known_object_07_epoch_" + str(epoch) + "_probs.npy")
    known_object_07_label = np.load(test_result_dir + "/known_object_07_epoch_" + str(epoch) + "_labels.npy")
    unknown_object_07_prob = np.load(test_result_dir + "/unknown_object_07_epoch_" + str(epoch) + "_probs.npy")
    unknown_object_07_label = np.load(test_result_dir + "/unknown_object_07_epoch_" + str(epoch) + "_labels.npy")

    # Data for test 08
    known_subject_08_prob = np.load(test_result_dir + "/known_subject_08_epoch_" + str(epoch) + "_probs.npy")
    known_subject_08_label = np.load(test_result_dir + "/known_subject_08_epoch_" + str(epoch) + "_labels.npy")
    unknown_subject_08_prob = np.load(test_result_dir + "/unknown_subject_08_epoch_" + str(epoch) + "_probs.npy")
    unknown_subject_08_label = np.load(test_result_dir + "/unknown_subject_08_epoch_" + str(epoch) + "_labels.npy")

    known_object_08_prob = np.load(test_result_dir + "/known_object_08_epoch_" + str(epoch) + "_probs.npy")
    known_object_08_label = np.load(test_result_dir + "/known_object_08_epoch_" + str(epoch) + "_labels.npy")
    unknown_object_08_prob = np.load(test_result_dir + "/unknown_object_08_epoch_" + str(epoch) + "_probs.npy")
    unknown_object_08_label = np.load(test_result_dir + "/unknown_object_08_epoch_" + str(epoch) + "_labels.npy")



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
        # test 07
        print("@" * 40)
        print("Processing known_subject_07")
        get_known_exit_stats(labels=known_subject_07_label,
                             probs=known_subject_07_prob,
                             class_threshold=known_known_thresh,
                             novelty_threshold=known_unknown_thresh)

        print("@" * 40)
        print("Processing known_object_07")
        get_known_exit_stats(labels=known_object_07_label,
                             probs=known_object_07_prob,
                             class_threshold=known_known_thresh,
                             novelty_threshold=known_unknown_thresh)

        print("@" * 40)
        print("Processing unknown_subject_07")
        get_unknown_exit_stats(labels=unknown_subject_07_label,
                              probs=unknown_subject_07_prob,
                              novelty_threshold=known_known_thresh)

        print("@" * 40)
        print("Processing unknown_object_07")
        get_unknown_exit_stats(labels=unknown_object_07_label,
                               probs=unknown_object_07_prob,
                               novelty_threshold=known_known_thresh)

        # Test 08
        print("&" * 60)
        print("Processing known_subject_08")
        get_known_exit_stats(labels=known_subject_08_label,
                             probs=known_subject_08_prob,
                             class_threshold=known_known_thresh,
                             novelty_threshold=known_unknown_thresh)

        print("@" * 40)
        print("Processing known_object_08")
        get_known_exit_stats(labels=known_object_08_label,
                             probs=known_object_08_prob,
                             class_threshold=known_known_thresh,
                             novelty_threshold=known_unknown_thresh)

        print("@" * 40)
        print("Processing unknown_subject_08")
        get_unknown_exit_stats(labels=unknown_subject_08_label,
                               probs=unknown_subject_08_prob,
                               novelty_threshold=known_known_thresh)

        print("@" * 40)
        print("Processing unknown_object_08")
        get_unknown_exit_stats(labels=unknown_object_08_label,
                               probs=unknown_object_08_prob,
                               novelty_threshold=known_known_thresh)




