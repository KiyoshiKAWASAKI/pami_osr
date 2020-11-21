import os
import time
import torch
from torchvision import datasets, transforms
from models import efficient_dense_net
import numpy as np
from timeit import default_timer as timer



nb_training_classes = 334
thresh_top_1 = 0.90

save_probs_path = ""
save_targets_path = ""
save_original_label_path = ""
save_rt_path = ""

save_probs_path_unknown = ""
save_targets_path_unknown = ""
save_original_label_path_unknown = ""
save_rt_path_unknown = ""


def test_with_novelty(test_loader,
                      model,
                      test_unknown):
    """

    :param val_loader:
    :param model:
    :param criterion:
    :return:
    """

    # Set the model to evaluation mode
    model.eval()

    # Define the softmax - do softmax to each block.
    sm = torch.nn.Softmax(dim=2)

    full_original_label_list = []
    full_prob_list = []
    full_target_list = []
    full_rt_list = []

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            print("*" * 50)

            rts = []
            input = input.cuda()
            target = target.cuda(async=True)

            # Save original labels to the list
            original_label_list = np.array(target.cpu().tolist())
            for label in original_label_list:
                full_original_label_list.append(label)

            # Check the target labels: keep or change
            if test_unknown:
                for k in range(len(target)):
                    target[k] = -1

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # Get the model outputs and RTs
            print("Timer started.")
            start =timer()
            output, end_time = model(input_var)

            # Save the RTs
            for end in end_time:
                rts.append(end-start)
            full_rt_list.append(rts)

            # extract the probability and apply our threshold
            prob = sm(torch.stack(output).to()) # Shape is [block, batch, class]
            prob_list = np.array(prob.cpu().tolist())
            max_prob = np.max(prob_list)

            # decide whether to do classification or reject
            # When the probability is larger than our threshold
            if max_prob >= thresh_top_1:
                print("Max top-1 probability is %f, larger than threshold %f" % (max_prob, thresh_top_1))

                # Get top-5 predictions from 5 classifiers
                pred_label_list = []
                for j in range(len(output)):
                    _, pred = output[j].data.topk(5, 1, True, True) # pred is a tensor
                    pred_label_list.append(pred.tolist())

                # Update the evaluation metrics for one sample
                # Top-1 and top-5: if any of the 5 classifiers makes a right prediction, consider correct
                # top_5_list = pred_label_list
                top_1_list = []

                for l in pred_label_list:
                    top_1_list.append(l[0][0])


                if target.tolist()[0] in top_1_list:
                    pred_label = target.tolist()[0]
                else:
                    pred_label = top_1_list[-1]

            # When the probability is smaller than our threshold
            else:
                pred_label = -1

            prob_list = np.reshape(prob_list,
                                    (prob_list.shape[1],
                                     prob_list.shape[0],
                                     prob_list.shape[2]))
            target_list = np.array(target.cpu().tolist())

            for one_prob in prob_list.tolist():
                full_prob_list.append(one_prob)
            for one_target in target_list.tolist():
                full_target_list.append(one_target)

            if not isinstance(output, list):
                output = [output]


        full_prob_list_np = np.array(full_prob_list)
        full_target_list_np = np.array(full_target_list)
        full_rt_list_np = np.array(full_rt_list)
        full_original_label_list_np = np.array(full_original_label_list)

        if test_unknown:
            print("Saving probabilities to %s" % save_probs_path_unknown)
            np.save(save_probs_path_unknown, full_prob_list_np)
            print("Saving target labels to %s" % save_targets_path_unknown)
            np.save(save_targets_path_unknown, full_target_list_np)
            print("Saving original labels to %s" % save_original_label_path_unknown)
            np.save(save_original_label_path_unknown, full_original_label_list_np)
            print("Saving RTs to %s" % save_rt_path_unknown)
            np.save(save_rt_path_unknown, full_rt_list_np)

        else:
            print("Saving probabilities to %s" % save_probs_path)
            np.save(save_probs_path, full_prob_list_np)
            print("Saving target labels to %s" % save_targets_path)
            np.save(save_targets_path, full_target_list_np)
            print("Saving original labels to %s" % save_original_label_path)
            np.save(save_original_label_path, full_original_label_list_np)
            print("Saving RTs to %s" % save_rt_path)
            np.save(save_rt_path, full_rt_list_np)
