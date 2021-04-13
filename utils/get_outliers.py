# Get the outlier for RTs

import numpy as np
import os
import sys
# import statistics



base_dir = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/0225/get_outliers"
cate = ["train_known_known", "train_known_unknown", "valid_known_known", "valid_known_unknown"]
models = ["original", "pp_add", "pp_mul"]




def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]




def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh




def get_outliers(base_dir,
                 categories,
                 models,
                 nb_itr=1000):
    """

    :param base_dir:
    :param categories:
    :param models:
    :return:
    """
    for one_category in categories:
        for one_model in models:
            one_model_dir = os.path.join(base_dir, one_category, one_model)
            print("Processing %s" % one_model_dir)

            rt_exit_0 = []
            rt_exit_1 = []
            rt_exit_2 = []
            rt_exit_3 = []
            rt_exit_4 = []

            for i in range(nb_itr):
                one_npy = np.load(one_model_dir + "/rt_itr_" + str(i) + ".npy")

                rt_exit_0 += one_npy[:, 0].tolist()
                rt_exit_1 += one_npy[:, 1].tolist()
                rt_exit_2 += one_npy[:, 2].tolist()
                rt_exit_3 += one_npy[:, 3].tolist()
                rt_exit_4 += one_npy[:, 4].tolist()

            rt_exit_0.sort()
            rt_exit_1.sort()
            rt_exit_2.sort()
            rt_exit_3.sort()
            rt_exit_4.sort()

            rt_exit_0_np = np.asarray(rt_exit_0)
            rt_exit_1_np = np.asarray(rt_exit_1)
            rt_exit_2_np = np.asarray(rt_exit_2)
            rt_exit_3_np = np.asarray(rt_exit_3)
            rt_exit_4_np = np.asarray(rt_exit_4)

            outlier_exit_0_mad = mad_based_outlier(points=rt_exit_0_np)
            outlier_exit_1_mad = mad_based_outlier(points=rt_exit_1_np)
            outlier_exit_2_mad = mad_based_outlier(points=rt_exit_2_np)
            outlier_exit_3_mad = mad_based_outlier(points=rt_exit_3_np)
            outlier_exit_4_mad = mad_based_outlier(points=rt_exit_4_np)

            _, counts_0 = np.unique(outlier_exit_0_mad, return_counts=True)
            _, counts_1 = np.unique(outlier_exit_1_mad, return_counts=True)
            _, counts_2 = np.unique(outlier_exit_2_mad, return_counts=True)
            _, counts_3 = np.unique(outlier_exit_3_mad, return_counts=True)
            _, counts_4 = np.unique(outlier_exit_4_mad, return_counts=True)


            # Print all the results
            print("&" * 60)
            print("Stats for exit 1")
            print("RT MAX - %f" % rt_exit_0[-1])
            print("3 largest RTs:")
            print(rt_exit_0[-3:])
            print("RT median - %f" % np.median(rt_exit_0_np))
            print("RT 95 percentile - %f" % np.percentile(rt_exit_0_np, 95))
            print("RT 99 percentile - %f" % np.percentile(rt_exit_0_np, 99))
            print("RT 99.99 percentile - %f" % np.percentile(rt_exit_0_np, 99.99))
            print("%d outliers by MAD, taking %f percent of the data" %
                  (counts_0[1], (float(counts_0[1])/float(len(rt_exit_0)))))

            print("&" * 60)
            print("Stats for exit 2")
            print("RT MAX - %f" % rt_exit_1[-1])
            print("3 largest RTs:")
            print(rt_exit_1[-3:])
            print("RT median - %f" % np.median(rt_exit_1_np))
            print("RT 95 percentile - %f" % np.percentile(rt_exit_1_np, 95))
            print("RT 99 percentile - %f" % np.percentile(rt_exit_1_np, 99))
            print("RT 99.99 percentile - %f" % np.percentile(rt_exit_1_np, 99.99))
            print("%d outliers by MAD, taking %f percent of the data" %
                  (counts_1[1], (float(counts_1[1])/float(len(rt_exit_1)))))

            print("&" * 60)
            print("Stats for exit 3")
            print("RT MAX - %f" % rt_exit_2[-1])
            print("3 largest RTs:")
            print(rt_exit_2[-3:])
            print("RT median - %f" % np.median(rt_exit_2_np))
            print("RT 95 percentile - %f" % np.percentile(rt_exit_2_np, 95))
            print("RT 99 percentile - %f" % np.percentile(rt_exit_2_np, 99))
            print("RT 99.99 percentile - %f" % np.percentile(rt_exit_2_np, 99.99))
            print("%d outliers by MAD, taking %f percent of the data" %
                  (counts_2[1], (float(counts_2[1])/float(len(rt_exit_2)))))

            print("&" * 60)
            print("Stats for exit 4")
            print("3 largest RTs:")
            print(rt_exit_3[-3:])
            print("RT MAX - %f" % rt_exit_3[-1])
            print("RT median - %f" % np.median(rt_exit_3_np))
            print("RT 95 percentile - %f" % np.percentile(rt_exit_3_np, 95))
            print("RT 99 percentile - %f" % np.percentile(rt_exit_3_np, 99))
            print("RT 99.99 percentile - %f" % np.percentile(rt_exit_3_np, 99.99))
            print("%d outliers by MAD, taking %f percent of the data" %
                  (counts_3[1], (float(counts_3[1])/float(len(rt_exit_3)))))

            print("&" * 60)
            print("Stats for exit 5")
            print("3 largest RTs:")
            print(rt_exit_4[-3:])
            print("RT MAX - %f" % rt_exit_4[-1])
            print("RT median - %f" % np.median(rt_exit_4_np))
            print("RT 95 percentile - %f" % np.percentile(rt_exit_4_np, 95))
            print("RT 99 percentile - %f" % np.percentile(rt_exit_4_np, 99))
            print("RT 99.99 percentile - %f" % np.percentile(rt_exit_4_np, 99.99))
            print("%d outliers by MAD, taking %f percent of the data" %
                  (counts_4[1], (float(counts_4[1])/float(len(rt_exit_4)))))

            sys.exit()


if __name__ == '__main__':
    get_outliers(base_dir=base_dir,
                 categories=cate,
                 models=models)