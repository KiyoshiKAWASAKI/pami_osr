# Initial version date:
# 07/30/2020: Getting the threshold for rejecting novelty
# Author: Jin Huang


import numpy as np



prob_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_07232020/07_29_threshold/0730_probs.npy"
label_npy_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_07232020/07_29_threshold/0730_targets.npy"


def get_novelty_thresh(prob_file_path,
                       label_file_path):

    """

    :param prob_file_path:
    :param label_file_path:
    :return:
    """

    probs = np.load(prob_file_path)
    labels = np.load(label_file_path)

    print(probs.shape)
    print(labels)



if __name__ == "__main__":
    get_novelty_thresh(prob_file_path=prob_npy_path,
                       label_file_path=label_npy_path)