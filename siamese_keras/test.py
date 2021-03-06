import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

#TODO: create baseline model file, and put model definition into it
from train import baseline_model
from dataset import read_img, read_img_vgg
import os

from pathlib import Path
from utils import prepare_images


data_folder = '../FIW_dataset/FIDs_NEW/'
test_df_path = "../FIW_dataset/test.csv"

IMG_SIZE_FN = (160, 160)
IMG_SIZE_VGG = (224, 224)
IMG_SIZE_ARCF = (112, 112)
IMG_SIZE_DF = (152, 152)
IMG_SIZE_DID = (55, 47)
IMG_SIZE_OF = (96, 96)

def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def convert_to_binary(lst, threshold=0.5):
    for i in range(len(lst)):
        if lst[i] >= threshold:
            lst[i] = 1
        else:
            lst[i] = 0
    return lst

def test():
    model = baseline_model()

    file_path = "checkpoints/arcface_openface_baseline.h5"
    model.load_weights(file_path)

    test_df = pd.read_csv(test_df_path)

    predictions = []
    labels = test_df.label.tolist()

    for batch in tqdm(chunker(list(zip(test_df.p1.values.tolist(), test_df.p2.values.tolist(), test_df.label.values.tolist())))):
        X1 = [x[0] for x in batch]
        X1_FN = np.array([read_img(data_folder + x, IMG_SIZE_FN, train=False) for x in X1])
        X1_VGG = np.array([read_img_vgg(data_folder + x, train=False) for x in X1])

        X2 = [x[1] for x in batch]
        X2_FN = np.array([read_img(data_folder + x, IMG_SIZE_FN, train=False) for x in X2])
        X2_VGG = np.array([read_img_vgg(data_folder + x, train=False) for x in X2])

        pred = model.predict([X1_FN, X2_FN, X1_VGG, X2_VGG]).ravel().tolist()

        # X1 = np.array([read_img(os.path.join(data_folder, x[0])) for x in batch])
        # X2 = np.array([read_img(os.path.join(data_folder, x[1])) for x in batch])

        # X1 = np.array([read_img(os.path.join(data_folder, x[0]), IMG_SIZE_DID, train=False) for x in batch])
        # X2 = np.array([read_img(os.path.join(data_folder, x[1]), IMG_SIZE_DID, train=False) for x in batch])
        # #
        # pred = model.predict([X1, X2]).ravel().tolist()

        predictions += pred



    auc_score = roc_auc_score(labels, predictions)
    print("ROC AUC score: ", auc_score)

    test_df["prediction"] = predictions

    # Accuracy score
    predictions = convert_to_binary(predictions)
    accuracy = accuracy_score(labels, predictions)
    print("Accuracy: ", accuracy)


    test_df.to_csv("predictions/predictions_arcface_openface_baseline.csv", index=False)


if __name__ == "__main__":
    # prepare_images(Path("../test_dataset/FIDs_NEW/"), Path("../test_dataset/FIDs_NEW_prepared/"))
    test()

