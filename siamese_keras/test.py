import numpy as np
import pandas as pd

from tqdm import tqdm

#TODO: create baseline model file, and put model definition into it
#TODO: create utils.py file
from train import baseline_model
from dataset import read_img


test_path = "../kinship_dataset/test/"

submission = pd.read_csv('../kinship_dataset/sample_submission.csv')

val_famillies_list = ["F09","F04","F08","F06", "F02"]


def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# def test():
#     model = baseline_model()
#     preds_for_sub = np.zeros(submission.shape[0])
#
#     for i in tqdm(range(len(val_famillies_list) - 1)):
#         file_path = f"checkpoints/vgg_face_{i}.h5"
#         model.load_weights(file_path)
#         # Get the predictions
#         predictions = []
#
#         for batch in tqdm(chunker(submission.img_pair.values)):
#             X1 = [x.split("-")[0] for x in batch]
#             X1 = np.array([read_img(test_path + x) for x in X1])
#
#             X2 = [x.split("-")[1] for x in batch]
#             X2 = np.array([read_img(test_path + x) for x in X2])
#
#             pred = model.predict([X1, X2]).ravel().tolist()
#             predictions += pred
#
#         preds_for_sub += np.array(predictions) / (len(val_famillies_list) - 1)
#
#
#     submission['is_related'] = preds_for_sub
#     submission.to_csv("submission.csv", index=False)



def test():
    model = baseline_model()

    file_path = f"checkpoints/baseline/vgg_face.h5"
    model.load_weights(file_path)

    predictions = []

    for batch in tqdm(chunker(submission.img_pair.values)):
        X1 = [x.split("-")[0] for x in batch]
        X1 = np.array([read_img(test_path + x, train=False) for x in X1])

        X2 = [x.split("-")[1] for x in batch]
        X2 = np.array([read_img(test_path + x, train=False) for x in X2])

        pred = model.predict([X1, X2]).ravel().tolist()
        predictions += pred

    submission['is_related'] = predictions
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    test()
