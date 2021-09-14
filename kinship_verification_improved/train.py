import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import pandas as pd

from tqdm import tqdm
import random

from dataset import KinDataSet
from models.siamese_network import SiameseNetwork
from utils import plot_confusion_matrix



NUM_EPOCHS = 50
VAL_FRACTION = 0.2
BINARIZATION_THRESHOLD = 0.5

FIW_ROOT_FOLDER = "FIDs_NEW/"

writer = SummaryWriter("logs/baseline_v2")


def train():
    ############## DATA PREPARATION ##############

    # TODO: implement the data loading for the image data

    fold1_df = pd.read_csv("data/FIW_fold1.csv")
    fold2_df = pd.read_csv("data/FIW_fold2.csv")
    fold3_df = pd.read_csv("data/FIW_fold3.csv")
    fold4_df = pd.read_csv("data/FIW_fold4.csv")
    fold5_df = pd.read_csv("data/FIW_fold5.csv")

    # concatenate 4 dataset folds into one single, leaving fifth fold for the testing
    df = pd.concat([fold1_df, fold2_df, fold3_df, fold5_df], ignore_index=True)

    idx = df.index.values.tolist()
    val_size = int(df.shape[0] * VAL_FRACTION)

    random.shuffle(idx)

    # split data on the train and validation subsets
    validation_idx = idx[:val_size]
    train_idx = idx[val_size:]

    train_df = df.loc[train_idx]
    validation_df = df.loc[validation_idx]

    test_df = fold4_df

    # TODO: Experiment with batch size, possibly increase to 32 or 64, or even more
    train_dataset = KinDataSet(df=train_df,
                               embeddings_file="fiw_vggface_embeddings.h5",
                               root_folder=FIW_ROOT_FOLDER)

    train_dataloader = DataLoader(train_dataset, batch_size=64,
                                  shuffle=True, num_workers=2)

    validation_dataset = KinDataSet(df=validation_df,
                                    embeddings_file="fiw_vggface_embeddings.h5",
                                    root_folder=FIW_ROOT_FOLDER)

    validation_dataloader = DataLoader(validation_dataset, batch_size=64,
                                       shuffle=True, num_workers=2)

    test_dataset = KinDataSet(df=test_df,
                              embeddings_file="fiw_vggface_embeddings.h5",
                              root_folder=FIW_ROOT_FOLDER)

    test_dataloader = DataLoader(test_dataset, batch_size=64,
                                 shuffle=True, num_workers=2)

    ############## MODEL & TRAINING PREPARATION ##############
    model = SiameseNetwork()

    # get available hardware and move model to it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)

    sigm = nn.Sigmoid()

    criterion = nn.BCEWithLogitsLoss()

    # TODO: tune learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    ############## TRAINING ##############

    for epoch in range(NUM_EPOCHS):
        print("Epoch {}\n".format(epoch + 1))
        model.train()

        running_loss = 0
        running_corrects = 0
        roc_auc_total = 0

        all_predictions, all_labels = [], []

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
                predictions[i] = int(predictions[i] > BINARIZATION_THRESHOLD)

            running_corrects += torch.sum(predictions == labels.data)

            predictions = predictions.cpu().detach().numpy().flatten().tolist()
            labels = labels.cpu().detach().numpy().flatten().tolist()

            roc_auc = roc_auc_score(labels, predictions)
            roc_auc_total += roc_auc

            if all_predictions is None:
                all_predictions = predictions
            else:
                all_predictions.extend(predictions)

            if all_labels is None:
                all_labels = labels
            else:
                all_labels.extend(labels)

        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects / len(train_dataset)
        train_roc_auc = roc_auc_total / len(train_dataloader)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("ROC AUC/train", train_roc_auc, epoch)

        conf_mat = confusion_matrix(all_labels, all_predictions)

        figure = plot_confusion_matrix(conf_mat, class_names=["False", "True"])

        writer.add_figure('Confusion matrix/train', figure, epoch)

        running_loss = 0
        running_corrects = 0
        roc_auc_total = 0
        all_predictions, all_labels = [], []

        print("| train loss: {} | train acc: {} | train ROC AUC: {} |\n".format(train_loss, train_acc, train_roc_auc))

        ############## VALIDATION ##############

        model.eval()

        for batch in tqdm(validation_dataloader):
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
                predictions[i] = int(predictions[i] > BINARIZATION_THRESHOLD)

            running_corrects += torch.sum(predictions == labels.data)

            predictions = predictions.cpu().detach().numpy().flatten().tolist()
            labels = labels.cpu().detach().numpy().flatten().tolist()

            roc_auc = roc_auc_score(labels, predictions)
            roc_auc_total += roc_auc

            if all_predictions is None:
                all_predictions = predictions
            else:
                all_predictions.extend(predictions)

            if all_labels is None:
                all_labels = labels
            else:
                all_labels.extend(labels)

        val_roc_auc = roc_auc_total / len(validation_dataloader)

        val_loss = running_loss / len(validation_dataset)
        val_acc = running_corrects / len(validation_dataset)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("ROC AUC/val", val_roc_auc, epoch)

        conf_mat = confusion_matrix(all_labels, all_predictions)

        figure = plot_confusion_matrix(conf_mat, class_names=["False", "True"])

        writer.add_figure('Confusion matrix/val', figure, epoch)

        running_corrects = 0
        roc_auc_total = 0
        all_predictions, all_labels = [], []

        print("| val loss: {} | val acc: {} | val ROC AUC: {} |\n".format(val_loss, val_acc, val_roc_auc))

        if (epoch + 1 < 10) and (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(),
                       "weights/baseline_epoch={}_val_acc={:.4f}_val_auc={:.2f}.pth".format(epoch + 1, val_acc,
                                                                                            val_roc_auc))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       "weights/baseline_epoch={}_val_acc={:.4f}_val_auc={:.2f}.pth".format(epoch + 1, val_acc,
                                                                                            val_roc_auc))

        ############## TESTING ##############

        model.eval()

        for batch in tqdm(test_dataloader):
            x1, x2, labels = batch

            # move data to the corresponding hardware (CPU or GPU)
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            # forward
            with torch.no_grad():
                outputs = model(x1, x2)

            predictions = sigm(outputs.float())

            for i in range(len(predictions)):
                predictions[i] = int(predictions[i] > 0.5)

            running_corrects += torch.sum(predictions == labels.data)

            predictions = predictions.cpu().detach().numpy().flatten().tolist()
            labels = labels.cpu().detach().numpy().flatten().tolist()

            roc_auc = roc_auc_score(labels, predictions)
            roc_auc_total += roc_auc

            if all_predictions is None:
                all_predictions = predictions
            else:
                all_predictions.extend(predictions)

            if all_labels is None:
                all_labels = labels
            else:
                all_labels.extend(labels)

        test_acc = running_corrects / len(test_dataset)
        test_roc_auc = roc_auc_total / len(test_dataloader)

        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("ROC AUC/test", test_roc_auc, epoch)

        print("All labels: ", all_labels)
        print("All predictions: ", all_predictions)

        conf_mat = confusion_matrix(all_labels, all_predictions)

        figure = plot_confusion_matrix(conf_mat, class_names=["False", "True"])

        writer.add_figure('Confusion matrix/test', figure, epoch)

        print("| test acc: {} | test ROC AUC: {} |\n".format(test_acc, test_roc_auc))

    return model


if __name__ == "__main__":
    train()
    writer.flush()
    writer.close()
