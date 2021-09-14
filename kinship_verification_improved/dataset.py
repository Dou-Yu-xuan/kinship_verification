from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image

from torch.utils.data import Dataset

import numpy as np
import h5py

IMG_SIZE_VGG = (224, 224)


def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

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


def read_img_vgg(path, IMG_SIZE=IMG_SIZE_VGG):
    img = image.load_img(path, target_size=IMG_SIZE)
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)


class KinDataSet(Dataset):
    def __init__(self, df, embeddings_file, root_folder):
        self.df = df
        self.root_folder = root_folder

        f = h5py.File(embeddings_file, "r")
        embeddings = f['embeddings'][...]
        paths = f['path'][...]

        embeddings_dict = {path.decode(): embedding for path, embedding in zip(paths, embeddings)}
        self.embeddings_dict = embeddings_dict

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        img_pair = self.df.iloc[[item]]

        img1_path = self.root_folder + img_pair.p1.tolist()[0]
        img2_path = self.root_folder + img_pair.p2.tolist()[0]
        l = img_pair.label.tolist()[0]

        embedding1 = self.embeddings_dict[img1_path]
        embedding2 = self.embeddings_dict[img2_path]

        label = np.array([l], dtype=float)
        return embedding1, embedding2, label


