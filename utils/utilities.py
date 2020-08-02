# Initial version date:
# 07/30/2020: Getting the threshold for rejecting novelty
# Author: Jin Huang


import numpy as np
import statistics
import sys



prob_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_07232020/07_29_threshold/0730_probs.npy"
label_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_07232020/07_29_threshold/0730_targets.npy"


def get_novelty_thresh(prob_file_path,
                       label_file_path,
                       top_k,
                       nb_block):

    """

    :param prob_file_path:
    :param label_file_path:
    :return:
    """

    probs = np.load(prob_file_path)[:, :, :369]
    labels = np.load(label_file_path)

    print(probs.shape) # (96237, 5, 413)
    print(labels.shape) # (96237,)

    max_probs = []
    top_k_probs = []

    for i in range(probs.shape[0]):
        one_prob = probs[i, :, :]

        # Top-1 prob
        max_p = np.max(one_prob)
        max_probs.append(max_p)

        # Top-5 prob: need to extract 5th from each block
        for j in range(one_prob.shape[0]):
            sub_prob = one_prob[j, :]

            top_k_p = np.sort(sub_prob)[-top_k]
            top_k_probs.append(top_k_p)

    # Get the prob of top-1 and top-k
    median_1 = statistics.median(max_probs)
    avg_1 = statistics.mean(max_probs)

    median_k = statistics.median(top_k_probs)
    avg_k = statistics.mean(top_k_probs)

    print(median_1)
    print(avg_1)
    print(median_k)
    print(avg_k)




def get_eval_from_csv(result_csv_path):
    """

    :param result_csv_path:
    :return:
    """




if __name__ == "__main__":
    get_novelty_thresh(prob_file_path=prob_npy_path,
                       label_file_path=label_npy_path,
                       top_k=5,
                       nb_block=5)