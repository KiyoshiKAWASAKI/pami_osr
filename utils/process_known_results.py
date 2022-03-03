# Processing the prob npy files and get stats
# Author: Jin Huang
# Updated date: 03/18/2021


import numpy as np
import sys
import statistics




# TODO: Cross-entropy seed 0 -- test_00
# save_path_sub = "2022-02-13/known_only_cross_entropy/seed_0"
# epoch = 147

# TODO: Cross-entropy seed 1 -- test_01
# save_path_sub = "2022-02-13/known_only_cross_entropy/seed_1"
# epoch = 181

# TODO: Cross-entropy seed 2 -- test_02
# save_path_sub = "2022-02-13/known_only_cross_entropy/seed_2"
# epoch = 195

# TODO: Cross-entropy seed 3 -- test_03
# save_path_sub = "2022-02-13/known_only_cross_entropy/seed_3"
# epoch = 142

# TODO: Cross-entropy seed 4 -- test_04
# save_path_sub = "2022-02-13/known_only_cross_entropy/seed_4"
# epoch = 120


# TODO: Sam seed 0 -- test_sam_0
# save_path_sub = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_0"
# epoch = 175

# TODO: Sam seed 1 -- test_sam_1
# save_path_sub = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_1"
# epoch = 105

# TODO: Sam seed 2 -- test_sam_2
# save_path_sub = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_2"
# epoch = 159

# TODO: Sam seed 3 -- test_sam_3
# save_path_sub = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_3"
# epoch = 103

# TODO: Sam seed 4 -- test_sam_4
# save_path_sub = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_4"
# epoch = 193


# TODO: PP seed 0 -- test_pp0
# save_path_sub = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0_exit_1.0/seed_0"
# epoch = 189

# TODO: PP seed 1 -- test_pp1
# save_path_sub = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0_exit_1.0/seed_1"
# epoch = 113

# TODO: PP seed 2 -- test_pp2
# save_path_sub = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0_exit_1.0/seed_2"
# epoch = 148

# TODO: PP seed 3 -- test_pp3
# save_path_sub = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0_exit_1.0/seed_3"
# epoch = 121

# TODO: PP seed 4 -- test_pp4
save_path_sub = "2022-02-17/known_only_cross_entropy_1.0_pfm_1.0_exit_1.0/seed_4"
epoch = 186


####################################################################
# Paths (usually, no need to change these)
####################################################################
test_known = False
test_unknown = True

percentile = [50]

test_binary = False
nb_training_classes = 293
save_path_base = "/scratch365/jhuang24/sail-on/models/msd_net"

valid_prob_dir = save_path_base + "/" + save_path_sub + "/features"
test_result_dir = save_path_base + "/" + save_path_sub + "/test_results"

print(test_result_dir)

####################################################################
# functions
####################################################################
def get_known_acc(labels,
                  probs,
                  nb_clfs=5):

    nb_correct_top_1 = [0] * nb_clfs
    nb_correct_top_3 = [0] * nb_clfs
    nb_correct_top_5 = [0] * nb_clfs

    for i in range(len(labels)):
        # print ("*" * 50)
        target_label = labels[i]
        prob = probs[i]

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]

            # TODO: Adding top3 and top5
            top_1 = np.argmax(one_prob)
            top_3 = np.argpartition(one_prob, -3)[-3:]
            top_5 = np.argpartition(one_prob, -5)[-5:]


            # TODO: Check for top-1, top-3 and top-5 separately
            if top_1 == target_label:
                nb_correct_top_1[j] += 1

            if target_label in top_3:
                nb_correct_top_3[j] += 1

            if target_label in top_5:
                nb_correct_top_5[j] += 1


    acc_top_1 = np.asarray(nb_correct_top_1)/(float(len(labels)))
    acc_top_3 = np.asarray(nb_correct_top_3)/(float(len(labels)))
    acc_top_5 = np.asarray(nb_correct_top_5)/(float(len(labels)))

    print("Total number of samples: %d" % len(labels))
    print("Accuracy top-1:",  acc_top_1)
    print("Accuracy top-3:",  acc_top_3)
    print("Accuracy top-5:",  acc_top_5)




def cal_accuracy_sep(labels,
                     probs,
                     nb_classes_without_rt=253,
                     nb_clfs=5):
    """
    Calculate the accuracy for samples with RT and without RT separately

    :return:
    """
    nb_correct_top_1_no_rt = [0] * nb_clfs
    nb_correct_top_3_no_rt = [0] * nb_clfs
    nb_correct_top_5_no_rt = [0] * nb_clfs

    nb_correct_top_1_rt = [0] * nb_clfs
    nb_correct_top_3_rt = [0] * nb_clfs
    nb_correct_top_5_rt = [0] * nb_clfs

    nb_no_rt_count = 0
    nb_rt_count = 0

    for i in range(len(labels)):
        target_label = labels[i]
        prob = probs[i]

        if target_label < nb_classes_without_rt-1:
            nb_no_rt_count +=1
            rt = False
        else:
            nb_rt_count += 1
            rt = True

        # check each classifier in order and decide when to exit
        for j in range(nb_clfs):
            one_prob = prob[j]

            # TODO: Adding top3 and top5
            top_1 = np.argmax(one_prob)
            top_3 = np.argpartition(one_prob, -3)[-3:]
            top_5 = np.argpartition(one_prob, -5)[-5:]


            # TODO: Check for top-1, top-3 and top-5 separately
            if top_1 == target_label:
                if rt:
                    nb_correct_top_1_rt[j] += 1
                else:
                    nb_correct_top_1_no_rt[j] += 1

            if target_label in top_3:
                if rt:
                    nb_correct_top_3_rt[j] += 1
                else:
                    nb_correct_top_3_no_rt[j] += 1

            if target_label in top_5:
                if rt:
                    nb_correct_top_5_rt[j] += 1
                else:
                    nb_correct_top_5_no_rt[j] += 1

    acc_top_1_no_rt = np.asarray(nb_correct_top_1_no_rt)/(float(nb_no_rt_count))
    acc_top_3_no_rt = np.asarray(nb_correct_top_3_no_rt)/(float(nb_no_rt_count))
    acc_top_5_no_rt = np.asarray(nb_correct_top_5_no_rt)/(float(nb_no_rt_count))

    acc_top_1_rt = np.asarray(nb_correct_top_1_rt) / (float(nb_rt_count))
    acc_top_3_rt = np.asarray(nb_correct_top_3_rt) / (float(nb_rt_count))
    acc_top_5_rt = np.asarray(nb_correct_top_5_rt) / (float(nb_rt_count))

    print("Total number of samples: %d" % len(labels))

    print("Accuracy top-1 (no RT):",  acc_top_1_no_rt)
    print("Accuracy top-3 (no RT):",  acc_top_3_no_rt)
    print("Accuracy top-5 (no RT):",  acc_top_5_no_rt)

    print("Accuracy top-1 (RT):", acc_top_1_rt)
    print("Accuracy top-3 (RT):", acc_top_3_rt)
    print("Accuracy top-5 (RT):", acc_top_5_rt)




def get_unknown_acc(labels,
                    probs,
                    top_1_threshold,
                    nb_clfs=5):

    nb_correct = 0
    nb_wrong = 0


    for i in range(len(labels)):
        # print("*" * 40)
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
                if max_prob > top_1_threshold[j]:
                    nb_wrong += 1
                    break

                else:
                    # If the max prob is smaller than threshold, check next clf
                    continue

            # If this is the last classifier
            else:
                if max_prob > top_1_threshold[-1]:
                    nb_wrong += 1

                else:
                    nb_correct += 1

    acc = float(nb_correct)/(float(nb_correct)+float(nb_wrong))

    print("Total number of samples: %d" % len(labels))
    print("Unknown predicted as unknown: %d" % nb_correct)
    print("Unknown predicted as known: %d" % nb_wrong)
    print("Accuracy: %4f" % acc)






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
    valid_known_unknown_probs_path = valid_prob_dir + "/valid_known_unknown_epoch_" + str(epoch) + "_probs.npy"
    valid_known_unknown_probs = np.load(valid_known_unknown_probs_path)

    valid_known_known_probs_path = valid_prob_dir + "/valid_known_known_epoch_" + str(epoch) + "_probs.npy"
    valid_known_known_probs = np.load(valid_known_known_probs_path)

    train_known_known_probs_path = valid_prob_dir + "/train_known_known_epoch_" + str(epoch) + "_probs.npy"
    train_known_known_probs = np.load(train_known_known_probs_path)


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

    print(np.unique(test_known_known_labels))

    test_unknown_unknown_probs_path = test_result_dir + "/unknown_unknown_epoch_" + str(epoch) + "_probs.npy"
    test_unknown_unknown_label_path = test_result_dir + "/unknown_unknown_epoch_" + str(epoch) + "_labels.npy"
    test_unknown_unknown_rt_path = test_result_dir + "/unknown_unknown_epoch_" + str(epoch) + "_rts.npy"
    test_unknown_unknown_labels = np.load(test_unknown_unknown_label_path)
    test_unknown_unknown_probs = np.load(test_unknown_unknown_probs_path)
    test_unknown_unknown_rts = np.load(test_unknown_unknown_rt_path)


    for one_perct in percentile:
        print("#" * 50)
        print("Current percentile:", one_perct)
        ################################################################
        # TODO: Get the thresholds for all 3 categories
        ################################################################
        known_known_thresh = get_thresholds(npy_file_path=valid_known_known_probs_path,
                                            percentile=one_perct)

        # known_known_thresh = get_thresholds(npy_file_path=train_known_known_probs_path,
        #                                     percentile=one_perct)

        print("thresholds:", known_known_thresh)

        # known_unknown_thresh = get_thresholds(npy_file_path=valid_known_unknown_probs_path,
        #                                       percentile=one_perct)


        ################################################################
        # Run test process
        ################################################################
        # known_known
        print("@" * 40)

        if test_known:
            print("Processing known_known samples")
            get_known_acc(labels=test_known_known_labels,
                                 probs=test_known_known_probs)

            cal_accuracy_sep(labels=test_known_known_labels,
                             probs=test_known_known_probs,
                             nb_classes_without_rt=253,
                             nb_clfs=5)

        if test_unknown:
            print("Processing unknown samples")
            get_unknown_acc(labels=test_unknown_unknown_labels,
                            probs=test_unknown_unknown_probs,
                            top_1_threshold=known_known_thresh,
                            nb_clfs=5)
