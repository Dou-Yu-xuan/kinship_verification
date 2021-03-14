from collections import defaultdict
from glob import glob
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import product
from torchvision import transforms
from PIL import Image
import torch
import random


def get_train_transforms():
    return transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([131.0912, 103.8827, 91.4953],
                                                     [1, 1, 1])
                                ])

def get_validation_transforms():
    return transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([131.0912, 103.8827, 91.4953],
                                                     [1, 1, 1])
                                ])

# TODO: this class should return both train and validation set, to properly divide sample number in them
class FIWDataset(Dataset):

    def __init__(self, train_families, csv_file, root_dir, train=True):

        self.train_df = pd.read_csv(csv_file)
        self.train_families = train_families
        self.root_dir = root_dir

        if train:
            self.transform = get_train_transforms()
        else:
            self.transform = get_validation_transforms()


        self.person2image = defaultdict(list)

        all_images = glob(self.root_dir + "*/*/*.jpg")
        all_images = [x.replace('\\', '/') for x in all_images]

        images = [x for x in all_images if x.split("/")[-3] in train_families]

        for x in images:
            self.person2image[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

        relationships = list(zip(self.train_df.p1.values.tolist(), self.train_df.p2.values.tolist()))

        self.image_pairs = []

        # TODO: do we need to train on all possible combinations of images of 2 persons? possibly we can take some part of these combinations
        for relation in relationships:
            member1_imgs, member2_imgs = self.person2image[relation[0]], self.person2image[relation[1]]

            self.image_pairs += list(product(member1_imgs, member2_imgs))

        for i in range(len(self.image_pairs)):
            img_pair = list(self.image_pairs[i])
            img_pair.append(1)
            self.image_pairs[i] = img_pair



        num_positive = len(self.image_pairs)

        print("Number of positive samples: ", num_positive)

        num_negative = 0
        while num_negative < num_positive:
            # sample two pairs
            two_pairs = random.sample(relationships, 2)

            # change their families ids
            f1_id = two_pairs[0][0].split("/")
            f2_id = two_pairs[1][0].split("/")

            # skip, in case if we randomly select two relations from the same family
            if f1_id[0] == f2_id[0]:
                continue

            two_pairs[0] = (f2_id[0] + "/" + f1_id[1], two_pairs[0][1])
            two_pairs[1] = (f1_id[0] + "/" + f2_id[1], two_pairs[1][1])


            # get all possible combinations of the images of each pair
            for pair in two_pairs:
                # in case if there are no such images, just skip
                try:
                    member1_imgs, member2_imgs = self.person2image[pair[0]], self.person2image[pair[1]]
                except Exception:
                    continue
                combinations = list(product(member1_imgs, member2_imgs))

                for i in range(len(combinations)):
                    img_pair = list(combinations[i])
                    # add negative label
                    img_pair.append(0)
                    combinations[i] = img_pair

                self.image_pairs += combinations

                num_negative += len(combinations)

        print("Number of positive samples: ", num_negative)




    def __len__(self):
        return len(self.image_pairs)


    def __getitem__(self, item):
        img_pair = self.image_pairs[item]

        img1_path, img2_path, l = img_pair

        image1 = Image.open(img1_path).convert('RGB')
        image2 = Image.open(img1_path).convert('RGB')

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        label = np.array([l], dtype=float)
        return image1, image2, label

