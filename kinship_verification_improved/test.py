import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

import pandas as pd
from tqdm import tqdm

from dataset import KinDataSet
from models.siamese_network import SiameseNetwork

FIW_ROOT_FOLDER = "FIDs_NEW/"


def test(df_path, embedding_file_path, weights_path):
    test_df = pd.read_csv(df_path)

    test_dataset = KinDataSet(df=test_df,
                              embeddings_file=embedding_file_path,
                              root_folder=FIW_ROOT_FOLDER)

    test_dataloader = DataLoader(test_dataset, batch_size=64,
                                 shuffle=True, num_workers=2)

    model = SiameseNetwork()

    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    sigm = nn.Sigmoid()

    running_corrects = 0
    roc_auc_total = 0

    all_predictions, all_labels = [], []

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

        roc_auc = roc_auc_score(labels.data.cpu(), predictions.cpu())
        roc_auc_total += roc_auc

        predictions = predictions.cpu().detach().numpy().flatten().tolist()
        labels = labels.cpu().detach().numpy().flatten().tolist()

        if all_predictions is None:
            all_predictions = predictions
        else:
            all_predictions.extend(predictions)

        if all_labels is None:
            all_labels = labels
        else:
            all_labels.extend(labels)

    test_acc = running_corrects / len(test_dataset) * 100
    test_roc_auc = roc_auc_total / len(test_dataloader)

    print("Test accuracy: ", test_acc)
    print("Test ROC AUC score: ", test_roc_auc)


if __name__ == "__main__":
    # FIW dataset
    test(df_path="data/FIW_fold4.csv",
         embedding_file_path="fiw_vggface_embeddings.h5",
         weights_path="weights/baseline_epoch=99_val_acc=0.9802.pth")

    # CornellKin dataset
    # test(df_path="CornellKin/CornellKin_test.csv",
    #      embedding_file_path="cornellkin_vggface_embeddings.h5",
    #      weights_path="weights/baseline_epoch=99_val_acc=0.9802.pth")
