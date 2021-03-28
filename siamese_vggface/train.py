import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from vggface import VGGFace
from dataset import FIWDataset
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from test import prepare_submission

NUM_EPOCHS = 50
VAL_FRACTION = 0.2

random.seed(227)
torch.manual_seed(227)

ROOT_PATH = "/home/franchukp/kinship_verification/ustc-nelslip_siamese/FG2020-kinship-master/Track1/input"


def train():
    ############## DATA PREPARATION ##############

    # # split on train and val set
    families = list(os.listdir(os.path.join(ROOT_PATH, "train/test-public-faces/test-public-faces/")))

    val_set_size = int(len(families) * VAL_FRACTION)

    val_families = random.sample(families, val_set_size)
    train_families = [fam for fam in families if fam not in val_families]

    train_dataset = FIWDataset(train_families=train_families,
                               csv_file=os.path.join(ROOT_PATH, "train_relationships.csv"),
                               root_dir=os.path.join(ROOT_PATH, "train/test-public-faces/test-public-faces/"),
                               )

    train_dataloader = DataLoader(train_dataset, batch_size=16,
                                  shuffle=True, num_workers=2)

    val_dataset = FIWDataset(train_families=val_families, train=False,
                             csv_file=os.path.join(ROOT_PATH, "train_relationships.csv"),
                             root_dir=os.path.join(ROOT_PATH, "train/test-public-faces/test-public-faces/"),
                             )

    val_dataloader = DataLoader(val_dataset, batch_size=16,
                                shuffle=True, num_workers=2)

    print("Train families number: ", len(train_families))
    print("Validation families number: ", len(val_families))

    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(val_dataset))

    ############## MODEL & TRAINING PREPARATION ##############
    model = VGGFace(desc='resnet_vggface')

    # get available hardware and move model to it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(model)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print(name, param)
        total_params += param
    print("Total parameters: ", total_params)

    sigm = nn.Sigmoid()

    criterion = nn.BCEWithLogitsLoss()
    # params_to_optimize = list(filter(lambda p : (p.requires_grad == True), model.parameters()))
    # print(params_to_optimize)

    params_to_optimize = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_optimize.append(param)
            print("Trainable parameter:", name)

    best_acc = 0

    optimizer = optim.Adam(params_to_optimize, lr=0.0003)

    date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open("training_results.txt", "a") as f:
        f.write("Date: {}\n".format(date))

    for epoch in range(NUM_EPOCHS):
        print("Epoch {}\n".format(epoch + 1))
        model.train()

        running_loss = 0
        running_corrects = 0
        all_outputs = []
        for batch in tqdm(train_dataloader):
            x1, x2, labels = batch

            # move data to the corresponding hardware (CPU or GPU)
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x1, x2)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # multiply by the size of batch
            running_loss += loss.item() * x1.size(0)

            predictions = sigm(outputs.float())

            for i in range(len(predictions)):
                predictions[i] = int(predictions[i] > 0.5)

            running_corrects += torch.sum(predictions == labels.data)

        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects / len(train_dataset)

        running_loss = 0
        running_corrects = 0

        print("| train loss: {} | train acc: {} |\n".format(train_loss, train_acc))

        with open("training_results.txt", "a") as f:
            f.write("{} | {} |".format(train_loss, train_acc))

        model.eval()

        for batch in tqdm(val_dataloader):
            x1, x2, labels = batch

            # move data to the corresponding hardware (CPU or GPU)
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            # forward
            with torch.no_grad():
                outputs = model(x1, x2)

            loss = criterion(outputs.float(), labels.float())

            # multiply by the size of batch
            running_loss += loss.item() * x1.size(0)

            predictions = sigm(outputs.float())

            for i in range(len(predictions)):
                predictions[i] = int(predictions[i] > 0.5)

            running_corrects += torch.sum(predictions == labels.data)

        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects / len(val_dataset)

        print("| val loss: {} | val acc: {} |\n".format(val_loss, val_acc))
        with open("training_results.txt", "a") as f:
            f.write(" {} | {}\n".format(val_loss, val_acc))

        if epoch % 4 == 0:
            torch.save(model.state_dict(), "weights/vggface_resnet50_epoch={}.pth".format(epoch))

        if train_acc > best_acc:
            print("New best model: ", epoch)
            best_acc = train_acc
            best_model = model

    return best_model


if __name__ == "__main__":
    model = train()
    prepare_submission(model)
