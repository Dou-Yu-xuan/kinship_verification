from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.data_utils import get_file
from random import choice, sample
from collections import defaultdict
from glob import glob
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import tensorflow as tf
from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate, ToGray, MedianBlur
)


# transforms = Compose([
#             HorizontalFlip(),
#             # RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2)
#             # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0)
#             HueSaturationValue(hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=0),
#             # ToGray()
#             MedianBlur(blur_limit=5),
#         ])





def preprocess_input(x, data_format=None, version=1, train=True):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    # if train:
    #     x_temp = x_temp.astype('float32')
    #     x_temp = transforms(image=x_temp)['image']

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError



    return x_temp

def get_train_val(familly_name):
    train_file_path = "../kinship_dataset/train_relationships.csv"
    train_folders_path = "../kinship_dataset/test-public-faces/"
    val_famillies = familly_name

    all_images = glob(train_folders_path + "*/*/*.jpg")
    all_images=[x.replace('\\','/') for x in all_images]
    train_images = [x for x in all_images if val_famillies not in x]
    val_images = [x for x in all_images if val_famillies in x]

    train_person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)

    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)
#     relationships = pd.read_excel(train_file_path)
    relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values.tolist(), relationships.p2.values.tolist()))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

    train = [x for x in relationships if val_famillies not in x[0]]
    val = [x for x in relationships if val_famillies in x[0]]
    return train, val, train_person_to_images_map, val_person_to_images_map


def read_img(path, train=True):
    img = image.load_img(path, target_size=(224, 224))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2, train=train)

def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        yield [X1, X2], labels

