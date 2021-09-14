from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import numpy as np

from glob import glob
from tqdm import tqdm
import h5py

from models.vggface import VGGFace
from dataset import read_img_vgg


def feature_extractor():
    input = Input(shape=(224, 224, 3))

    vggface = VGGFace(model='resnet50', include_top=False)

    out = vggface(input)

    model = Model(input, out)

    return model


def extract_features(root_dir, embeddings_file_path):
    # get pathes to all images in the dataset
    all_images = glob(root_dir + "*/*/*.jpg")
    all_images = [x.replace('\\', '/') for x in all_images]

    # dt = h5py.special_dtype(vlen=str) # for h5py 3 version
    dt = h5py.string_dtype(encoding='ascii')  # for h5py 2 version
    f = h5py.File(embeddings_file_path, "w")

    f.create_dataset("embeddings", (len(all_images), 2048), dtype='f')
    f.create_dataset("path", (len(all_images),), dtype=dt)

    model = feature_extractor()

    for i in tqdm(range(len(all_images))):
        image_path = all_images[i]

        image = read_img_vgg(image_path)  # read & preprocess image

        image = np.array([image]).astype('float32')

        embedding = model.predict(image)  # feed image through the feature extractor to get the embeddings

        f["embeddings"][i] = embedding.flatten()
        f["path"][i] = image_path

    f.close()


if __name__ == "__main__":
    extract_features(root_dir="FIDs_NEW/", embeddings_file_path="fiw_vggface_embeddings.h5")  # FIW dataset

    # extract_features(root_dir="CornellKin/",
    #                  embeddings_file_path="cornellkin_vggface_embeddings.h5")                   # CornellKin dataset
