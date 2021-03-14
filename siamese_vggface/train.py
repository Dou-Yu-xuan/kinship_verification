import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from vggface import VGGFace
from dataset import FIWDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# import sklearn.metrics.roc_auc_score as roc_auc

NUM_EPOCHS = 5
VAL_FRACTION = 0.2

# random.seed(228)
# torch.manual_seed(228)


ROOT_PATH = "/home/franchukp/kinship_verification/ustc-nelslip_siamese/FG2020-kinship-master/Track1/input"



def train():
    ############## DATA PREPARATION ##############


    # split on train and val set
    families = list(os.listdir(os.path.join(ROOT_PATH, "train/train-faces/")))
    val_set_size = int(len(families) * VAL_FRACTION)

    val_families = random.sample(families, val_set_size)
    train_families = [fam for fam in families if fam not in val_families]



    train_dataset = FIWDataset(train_families=train_families,
                         csv_file=os.path.join(ROOT_PATH, "train_relationships.csv"),
                         root_dir=os.path.join(ROOT_PATH, "train/train-faces/"),
                        )

    train_dataloader = DataLoader(train_dataset, batch_size=8,
                            shuffle=True, num_workers=2)

    val_dataset = FIWDataset(train_families=val_families,
                                    csv_file=os.path.join(ROOT_PATH, "train_relationships.csv"),
                                    root_dir=os.path.join(ROOT_PATH, "train/train-faces/"),
                                    )

    val_dataloader = DataLoader(val_dataset, batch_size=8,
                            shuffle=True, num_workers=2)

    print(len(train_families))
    print(len(val_families))

    print(len(train_dataset))
    print(len(val_dataset))

    ############## MODEL & TRAINING PREPARATION ##############
    model = VGGFace(desc='resnet50', desc_out_shape=8631, neurons_num=[2048, 512])

    # get available hardware and move model to it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        print("Epoch {}\n".format(epoch + 1))
        model.train()

        running_loss = 0
        all_outputs = []

        # TODO: change on tqdm.notebook
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
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()

            # multiply by the size of batch
            running_loss += loss.item() * x1.size(0)
            # all_outputs += outputs


        epoch_loss = running_loss / len(train_dataset)
        # epoch_roc_auc = roc_auc([1 for i in range(len(all_outputs))], all_outputs)

        running_loss = 0
        all_outputs = 0

        print("| train loss: {} |\n".format(epoch_loss))


        model.eval()

        # TODO: change on tqdm.notebook
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
            # all_outputs += outputs

        epoch_loss = running_loss / len(val_dataset)
        # epoch_roc_auc = roc_auc([1 for i in range(len(all_outputs))], all_outputs)

        running_loss = 0
        all_outputs = 0

        print("| val loss: {} |\n".format(epoch_loss))

        if epoch % 3 == 0:
            torch.save(model.state_dict(), "vggface_resnet50_epoch={}.pth".format(epoch))

if __name__ == "__main__":
    train()

