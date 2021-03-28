import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from vggface import VGGFace
from dataset import FIWDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torchvision import transforms



def get_validation_transforms():
    return transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([131.0912 / 255, 103.8827 / 255, 91.4953 / 255],
                                                     [1, 1, 1])
                                ])

def test(model):

    ROOT_PATH = "/home/franchukp/kinship_verification/ustc-nelslip_siamese/FG2020-kinship-master/Track1/input/test/test"
    test_df = pd.read_csv("sample_submission.csv")

    img_pairs = test_df.img_pair.tolist()

    img_pairs = [img_pair.split("-") for img_pair in img_pairs]

    for img_pair in img_pairs:
        img_pair[0] = os.path.join(ROOT_PATH, img_pair[0])
        img_pair[1] = os.path.join(ROOT_PATH, img_pair[1])

    # model = VGGFace(desc='resnet50', desc_out_shape=512, neurons_num=[2048, 512])
    #
    # weights_path = "weights/vggface_resnet50_epoch=40.pth"
    #
    # # load weights, pretrained on VGGFace2 dataset
    # state_dict = torch.load(weights_path)


    # with torch.no_grad():
    #     model.descriptor.classifier.weight.data = state_dict['descriptor.classifier.weight']
    #     model.descriptor.classifier.bias.data = state_dict['descriptor.classifier.bias']
    #
    #     model.fc1.weight.data = state_dict['fc1.weight']
    #     model.fc1.bias.data = state_dict['fc1.bias']
    #
    #     model.fc2.weight.data = state_dict['fc2.weight']
    #     model.fc2.bias.data = state_dict['fc2.bias']
    #

    # model.load_state_dict(state_dict)

    # get available hardware and move model to it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    predictions = []

    sigm = nn.Sigmoid()

    for i in tqdm(range(len(img_pairs))):
        img1_path, img2_path = img_pairs[i]

        image1 = Image.open(img1_path).convert('RGB')
        image2 = Image.open(img1_path).convert('RGB')

        transform = get_validation_transforms()

        x1 = transform(image1)
        x2 = transform(image2)

        # add 4th dimension to the tensor, aka add a batch of size 1
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)

        x1 = x1.to(device)
        x2 = x2.to(device)

        with torch.no_grad():
            outputs = model(x1, x2)

        outputs = sigm(outputs.float())

        predictions.append(outputs.cpu().numpy()[0][0])

    return predictions



def prepare_submission(model):
    predictions = test(model)

    df = pd.read_csv("sample_submission.csv")
    df.is_related = predictions

    df.to_csv("submission.csv", index=False)


# if __name__ == "__main__":
#     prepare_submission()
