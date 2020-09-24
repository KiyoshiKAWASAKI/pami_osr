import json

import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset
# from torchvision import datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

from collections import defaultdict
import os
import numpy as np
from PIL import Image
import sys
import random

def collate(batch):
    try:
        PADDING_CONSTANT = 0

        batch = [b for b in batch if b is not None]
        #These all should be the same size or error
        assert len(set([b["img"].shape[0] for b in batch])) == 1
        assert len(set([b["img"].shape[2] for b in batch])) == 1

        # TODO: what is dim 0, 1, 2??
        """
        dim0: channel
        dim1: ??
        dim2: hight?
        """
        dim0 = batch[0]["img"].shape[0]
        dim1 = max([b["img"].shape[1] for b in batch])
        dim1 = dim1 + (dim0 - (dim1 % dim0))
        dim2 = batch[0]["img"].shape[2]

        # print(batch)

        all_labels = []
        psychs = []

        input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
        # input_batch = batch

        for i in range(len(batch)):
            b_img = batch[i]["img"]
            input_batch[i,:,:b_img.shape[1],:] = b_img
            l = batch[i]["gt_label"]
            psych = batch[i]["rt"]
            cate = batch[i]["category"]
            all_labels.append(l)

            # TODO: Leave the scale factor alone for now
            if psych is not None:
                # psych = (423)-psych
                # if psych < 0:
                #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #     print((psych))

                # psychs.append((psych)*100000)
                psychs.append(psych)
            else:
                psychs.append(0)

        # line_imgs = input_batch.transpose([0,3,1,2])
        # line_imgs = torch.from_numpy(line_imgs)
        line_imgs = torch.from_numpy(input_batch)
        labels = torch.from_numpy(np.array(all_labels).astype(np.int32))

        return {"imgs": line_imgs,
                "labels": labels,
                "rts": psychs,
                "category": cate}

    except Exception as e:
        print(e)




class msd_net_dataset(Dataset):
    def __init__(self,
                 json_path,
                 transform,
                 img_height=32,
                 augmentation=False):

        with open(json_path) as f:
            data = json.load(f)
        print("Json file loaded: %s" % json_path)

        self.img_height = img_height
        self.data = data
        self.transform = transform
        self.augmentation = augmentation
        self.random_weight = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[str(idx)]
        except KeyError:
            item = self.data[str(idx+1)]

        # Open the image and do normalization and augmentation
        img = Image.open(item["img_path"])
        img = img.convert('RGB')
        # print(img.size)
        try:
            img = self.transform(img)

        except Exception as e:
            print("@" * 20)
            print(e)
            print("@" * 20)
            print(idx)
            print(self.data[str(idx)])
            sys.exit(0)

        # Deal with reaction times
        if self.random_weight is None:
            # print("Checking whether an RT exists for this image...")
            if item["RT"] != None:
                rt = item["RT"]
                # print("RT exists")
            else:
                # print("RT does not exist")
                rt = None
        # No random weights for reaction time
        else:
            pass

        return {
            "img": img,
            "gt_label": item["label"],
            "rt": rt,
            "category": item["category"]
        }

