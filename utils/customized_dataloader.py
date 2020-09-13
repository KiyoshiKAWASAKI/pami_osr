import json

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset
# from torchvision import datasets as datasets
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np

import random

def collate(batch):
    PADDING_CONSTANT = 0

    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim1 = dim1 + (dim0 - (dim1 % dim0))
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    # label_lengths = []
    psychs = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img
        l = batch[i]['gt_label']
        psych = batch[i]["rt"]
        all_labels.append(l)
        # label_lengths.append(len(l))

        if psych is not None:
            # print(psych)
            # print(((200-psych)/len(l)))
            # print("-----------")
            psych = (423)-psych
            if psych < 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print((psych))

            psychs.append((psych)*100000)
        else:
            psychs.append(0)
    # all_labels = np.concatenate(all_labels)
    # label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2])
    line_imgs = torch.from_numpy(line_imgs)
    labels = torch.from_numpy(np.array(all_labels).astype(np.int32))
    # label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "psychs": psychs,
        "gt_label": [b['gt_label'] for b in batch]
    }




class msd_net_dataset(Dataset):
    def __init__(self,
                 json_path,
                 img_height=32,
                 augmentation=False):

        # Everything was in one JSON
        with open(json_path) as f:
            data = json.load(f)
        print("Json file loaded!")

        self.img_height = img_height
        self.data = data
        # print("*" * 20)
        # print(self.data["0"])
        # print("*" * 20)
        self.augmentation = augmentation
        self.randomWeights = None

        for i in range(20):
            print(self.data[str(i)])



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print("*" * 20)
        print(idx)
        print(self.data[str(idx)])
        print("*" * 20)
        item = self.data[str(idx)]

        img = cv2.imread(item["img_path"])
        img_category = item["category"]
        print("Loaded one image: %s" % item["img_path"])

        if self.randomWeights is None:
            print("Checking whether an RT exists for this image...")
            try:
                rt = item["RT"]
                print("RT exists")
            except:
                print("RT does not exist")
                rt = None
        else:
            # TODO: should we apply random psyphy weights??
            pass
            # print("WARNING!!! RANDOM PSYCHOMETRIC WEIGHTS ARE BEING USED")
            # psych = self.randomWeights[idx]
            # print psych

        if img is None:
            print("Warning: image is None:", item["img_path"])
            return None

        # TODO: check the image size for image_net and change this part
        percent = float(self.img_height) / img.shape[0]
        img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

        # TODO: Adding multiple data augmentation methods
        if self.augmentation:
            pass
            # img = grid_distortion.warp_image(img, h_mesh_std=5, w_mesh_std=10)

        # img = img.astype(np.float32)
        # img = img / 128.0 - 1.0

        gt = item["label"]
        # gt_label = string_utils.str2label(gt, self.char_to_idx)

        return {
            "line_img": img,
            "gt_label": gt,
            "rt": rt,
            "category": img_category
        }

