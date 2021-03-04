# Plot losses and accuracies
# Initial version date: 03/03/2021
# Author: Jin Huang


import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys
import os

base_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models"


original_csv_path = base_path + "/" + "0225/original/results.csv"
pp_mul_csv_path = base_path + "/" + "0225/pp_loss/results.csv"
pp_add_csv_path = base_path + "/" + "0225/pp_loss_add/results.csv"

save_loss_fig_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/0225/loss.png"
save_acc_fig_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/0225/acc.png"


def plot_fig(original_csv_path,
             pp_mul_csv_path,
             pp_add_csv_path,
             save_figure_path,
             plot_type,
             figure_size,
             font_size=16):
    """

    :param original_csv_path:
    :param pp_mul_csv_path:
    :param pp_add_csv_path:
    :param save_loss_figure_path:
    :param save_acc_figure_path:
    :param loss_figure_name:
    :param acc_figure_name:
    :param plot_type: "loss" or "acc"
    :param figure_size:
    :param font_size:
    :return:
    """

    # Load all the results together
    df_original = pd.read_csv(original_csv_path, delimiter=",", index_col=False)
    df_pp_mul = pd.read_csv(pp_mul_csv_path, delimiter=",", index_col=False)
    df_pp_add = pd.read_csv(pp_add_csv_path, delimiter=",", index_col=False)

    df_original.columns = df_original.columns.str.strip()
    df_pp_mul.columns = df_pp_mul.columns.str.strip()
    df_pp_add.columns = df_pp_add.columns.str.strip()

    # Get the losses for 3 models
    train_loss_original = df_original[["train_loss"]]
    valid_loss_original = df_original[["valid_loss"]]

    train_loss_pp_mul = df_pp_mul[["train_loss"]]
    valid_loss_pp_mul = df_pp_mul[["valid_loss"]]

    train_loss_pp_add = df_pp_add[["train_loss"]]
    valid_loss_pp_add = df_pp_add[["valid_loss"]]

    # Get the accuracies
    train_acc_original_top_1 = df_original[["train_acc_top1"]]
    train_acc_original_top_3 = df_original[["train_acc_top3"]]
    train_acc_original_top_5 = df_original[["train_acc_top5"]]
    valid_acc_original_top_1 = df_original[["valid_acc_top1"]]
    valid_acc_original_top_3 = df_original[["valid_acc_top3"]]
    valid_acc_original_top_5 = df_original[["valid_acc_top5"]]

    train_acc_pp_mul_top_1 = df_pp_mul[["train_acc_top1"]]
    train_acc_pp_mul_top_3 = df_pp_mul[["train_acc_top3"]]
    train_acc_pp_mul_top_5 = df_pp_mul[["train_acc_top5"]]
    valid_acc_pp_mul_top_1 = df_pp_mul[["valid_acc_top1"]]
    valid_acc_pp_mul_top_3 = df_pp_mul[["valid_acc_top3"]]
    valid_acc_pp_mul_top_5 = df_pp_mul[["valid_acc_top5"]]

    train_acc_pp_add_top_1 = df_pp_add[["train_acc_top1"]]
    train_acc_pp_add_top_3 = df_pp_add[["train_acc_top3"]]
    train_acc_pp_add_top_5 = df_pp_add[["train_acc_top5"]]
    valid_acc_pp_add_top_1 = df_pp_add[["valid_acc_top1"]]
    valid_acc_pp_add_top_3 = df_pp_add[["valid_acc_top3"]]
    valid_acc_pp_add_top_5 = df_pp_add[["valid_acc_top5"]]

    fig = plt.figure(figsize=figure_size)

    if plot_type == "loss":
        plt.plot(train_loss_original, label="Training Loss (Original)")
        plt.plot(valid_loss_original, label="Validation Loss (Original)")
        plt.plot(train_loss_pp_mul, label="Training Loss (Psyphy-mul)")
        plt.plot(valid_loss_pp_mul, label="Validation Loss (Psyphy-mul)")
        plt.plot(train_loss_pp_add, label="Training Loss (Psyphy-add)")
        plt.plot(valid_loss_pp_add, label="Validation Loss (Psyphy-add)")

        fig.suptitle("Losses", fontsize=font_size)
        plt.xlabel('Number of Epochs', fontsize=font_size)
        plt.ylabel('Loss', fontsize=font_size)

    elif plot_type == "acc":
        plt.plot(train_acc_original_top_1, label="train_acc_original_top_1")
        plt.plot(train_acc_original_top_3, label="train_acc_original_top_3")
        plt.plot(train_acc_original_top_5, label="train_acc_original_top_5")
        plt.plot(valid_acc_original_top_1, label="valid_acc_original_top_1")
        plt.plot(valid_acc_original_top_3, label="valid_acc_original_top_3")
        plt.plot(valid_acc_original_top_5, label="valid_acc_original_top_5")

        plt.plot(train_acc_pp_mul_top_1, label="train_acc_pp_mul_top_1")
        plt.plot(train_acc_pp_mul_top_3, label="train_acc_pp_mul_top_3")
        plt.plot(train_acc_pp_mul_top_5, label="train_acc_pp_mul_top_5")
        plt.plot(valid_acc_pp_mul_top_1, label="valid_acc_pp_mul_top_1")
        plt.plot(valid_acc_pp_mul_top_3, label="valid_acc_pp_mul_top_3")
        plt.plot(valid_acc_pp_mul_top_5, label="valid_acc_pp_mul_top_5")

        plt.plot(train_acc_pp_add_top_1, label="train_acc_pp_add_top_1")
        plt.plot(train_acc_pp_add_top_3, label="train_acc_pp_add_top_3")
        plt.plot(train_acc_pp_add_top_5, label="train_acc_pp_add_top_5")
        plt.plot(valid_acc_pp_add_top_1, label="valid_acc_pp_add_top_1")
        plt.plot(valid_acc_pp_add_top_3, label="valid_acc_pp_add_top_3")
        plt.plot(valid_acc_pp_add_top_5, label="valid_acc_pp_add_top_5")

        fig.suptitle("Accuracy", fontsize=font_size)
        plt.xlabel('Number of Epochs', fontsize=font_size)
        plt.ylabel('Accuracy(%)', fontsize=font_size)

    plt.legend()
    plt.show()

    fig.savefig(save_figure_path)
    print("Figure saved to: %s" % save_figure_path)



if __name__ == '__main__':
    # plot_fig(original_csv_path=original_csv_path,
    #          pp_mul_csv_path=pp_mul_csv_path,
    #          pp_add_csv_path=pp_add_csv_path,
    #          save_figure_path=save_loss_fig_path,
    #          plot_type="loss",
    #          figure_size=(12, 6))

    plot_fig(original_csv_path=original_csv_path,
             pp_mul_csv_path=pp_mul_csv_path,
             pp_add_csv_path=pp_add_csv_path,
             save_figure_path=save_acc_fig_path,
             plot_type="acc",
             figure_size=(20, 12))