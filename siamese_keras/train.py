from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract,Add,  Reshape, Lambda, Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras
import pandas as pd
import numpy as np
from dataset import read_img, read_img_vgg
from sklearn.metrics import roc_auc_score, accuracy_score

import os

from pathlib import Path
from utils import signed_sqrt, prepare_images, prepare_images_ellipse
from dataset import get_train_val, gen, fiw_data_generator

from vggface import VGGFace
from facenet import FaceNet
from arcface import ArcFace
from deepid import DeepID
from facenet_deepface import FaceNetDF
from openface import OpenFace

IMG_SIZE_FN = (160, 160)
IMG_SIZE_VGG = (224, 224)
IMG_SIZE_ARCF = (112, 112)
IMG_SIZE_DF = (152, 152)
IMG_SIZE_DID = (55, 47)
IMG_SIZE_OF = (96, 96)

train_file_path = "../FIW_dataset/train_FIW.csv"
val_file_path = "../FIW_dataset/val_FIW.csv"
images_root_dir = "../FIW_dataset/FIDs_NEW/"

test_file_path = "../FIW_dataset/test.csv"

train_df = pd.read_csv(train_file_path)
val_df = pd.read_csv(val_file_path)


auc = tf.keras.metrics.AUC()

logdir = "logs/" + "vggface_facenet_flip_contrast_brightness_saturation"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def convert_to_binary(lst, threshold=0.5):
    for i in range(len(lst)):
        if lst[i] >= threshold:
            lst[i] = 1
        else:
            lst[i] = 0
    return lst

# class TestCallback(tensorflow.keras.callbacks.Callback):
#     def __init__(self, data_folder, csv_file):
#         self.test_df = pd.read_csv(csv_file)
#         self.data_folder = data_folder
#
#     def on_epoch_end(self, epoch, logs={}):
#         if epoch % 10 == 0:
#             predictions = []
#             labels = self.test_df.label.tolist()
#
#             for batch in chunker(list(zip(self.test_df.p1.values, self.test_df.p2.values, self.test_df.label.values))):
#                 X1 = [x[0] for x in batch]
#                 X1_FN = np.array([read_img(self.data_folder + x, IMG_SIZE_FN) for x in X1])
#                 X1_VGG = np.array([read_img_vgg(self.data_folder + x) for x in X1])
#
#                 X2 = [x[1] for x in batch]
#                 X2_FN = np.array([read_img(self.data_folder + x, IMG_SIZE_FN) for x in X2])
#                 X2_VGG = np.array([read_img_vgg(self.data_folder + x) for x in X2])
#
#                 pred = self.model.predict([X1_FN, X2_FN, X1_VGG, X2_VGG]).ravel().tolist()
#
#                 predictions += pred
#
#                 # X1 = np.array([read_img_of(os.path.join(self.data_folder, x[0])) for x in batch])
#                 # X2 = np.array([read_img_of(os.path.join(self.data_folder, x[1])) for x in batch])
#                 #
#                 # pred = self.model.predict([X1, X2]).ravel().tolist()
#                 #
#                 # predictions += pred
#
#             auc_score = roc_auc_score(labels, predictions)
#
#             predictions = convert_to_binary(predictions)
#             accuracy = accuracy_score(labels, predictions)
#
#             print("Epoch: {} | Test ROC AUC: {} | Test Accuracy: {}".format(epoch, auc_score, accuracy))


def baseline_model():
    input_1 = Input(shape=(IMG_SIZE_FN[0], IMG_SIZE_FN[1], 3))
    input_2 = Input(shape=(IMG_SIZE_FN[0], IMG_SIZE_FN[1], 3))
    input_3 = Input(shape=(IMG_SIZE_VGG[0], IMG_SIZE_VGG[1], 3))
    input_4 = Input(shape=(IMG_SIZE_VGG[0], IMG_SIZE_VGG[1], 3))


    model_vgg = VGGFace(model='resnet50', include_top=False)
    model_facenet = FaceNet()


    x1 = model_facenet(input_1)
    x2 = model_facenet(input_2)
    x3 = model_vgg(input_3) # (1, 2048)
    x4 = model_vgg(input_4) # (1, 2048)

    x1 = Reshape((1, 1, 128))(x1)
    x2 = Reshape((1, 1, 128))(x2)
    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)]) # (1, 256)
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)]) # (1, 256)

    ##################### Combination #7 FIW ###############

    # x1_x2_sum = Add()([x1, x2])  # x1 + x2
    # x1_x2_diff = Subtract()([x1, x2])  # x1 - x2
    # x1_x2_product = Multiply()([x1, x2])  # x1 * x2
    #
    # x1_sqrt = Lambda(lambda tensor: signed_sqrt(tensor))(x1)
    # x2_sqrt = Lambda(lambda tensor: signed_sqrt(tensor))(x2)
    #
    # x1_sqrt_x2_sqrt_sum = Add()([x1_sqrt, x2_sqrt])
    #
    # x1_square = Multiply()([x1, x1])
    # x2_square = Multiply()([x2, x2])
    # x1_sq_x2_sq_sum = Add()([x1_square, x2_square])
    #
    #
    # x3_x4_sum = Add()([x3, x4])  # x3 + x4
    # x3_x4_diff = Subtract()([x3, x4])  # x3 - x4
    # x3_x4_product = Multiply()([x3, x4])  # x3 * x4
    #
    # x3_sqrt = Lambda(lambda tensor: signed_sqrt(tensor))(x3)
    # x4_sqrt = Lambda(lambda tensor: signed_sqrt(tensor))(x4)
    #
    # x3_sqrt_x4_sqrt_sum = Add()([x3_sqrt, x4_sqrt])
    #
    # x3_square = Multiply()([x3, x3])
    # x4_square = Multiply()([x4, x4])
    # x3_sq_x4_sq_sum = Add()([x3_square, x4_square])
    #
    #
    #
    # x3_x4_sum = Conv2D(128, [1, 1])(x3_x4_sum)  # (1, 128)
    # x3_x4_diff = Conv2D(128, [1, 1])(x3_x4_diff)  # (1, 128)
    # x3_x4_product = Conv2D(128, [1, 1])(x3_x4_product)  # (1, 128)
    # x3_sqrt_x4_sqrt_sum = Conv2D(128, [1, 1])(x3_sqrt_x4_sqrt_sum)
    # x3_sq_x4_sq_sum = Conv2D(128, [1, 1])(x3_sq_x4_sq_sum)
    #
    # x = Concatenate(axis=-1)(
    #     [Flatten()(x3_x4_sum), x1_x2_sum,
    #      Flatten()(x3_x4_diff), x1_x2_diff,
    #      Flatten()(x3_x4_product), x1_x2_product,
    #      Flatten()(x3_sqrt_x4_sqrt_sum), x1_sqrt_x2_sqrt_sum,
    #      Flatten()(x3_sq_x4_sq_sum), x1_sq_x2_sq_sum
    #      ])


    ########################################################

    ################### Combination #6 FIW ##################

    # x1_square = Multiply()([x1, x1])
    # x2_square = Multiply()([x2, x2])
    # x1_sq_x2_sq_diff = Subtract()([x1_square, x2_square])
    #
    # x1_x2_diff = Subtract()([x1, x2])
    # x1_x2_diff_sq = Multiply()([x1_x2_diff, x1_x2_diff])
    #
    # x1_x2_product = Multiply()([x1, x2])
    #
    # x1_x2_sum = Add()([x1, x2])  # x1 + x2
    #
    # x1_sqrt = Lambda(lambda tensor: signed_sqrt(tensor))(x1)
    # x2_sqrt = Lambda(lambda tensor: signed_sqrt(tensor))(x2)
    #
    # x1_sqrt_x2_sqrt_diff = Subtract()([x1_sqrt, x2_sqrt])
    # x1_sqrt_x2_sqrt_sum = Add()([x1_sqrt, x2_sqrt])
    #
    #
    # x3_square = Multiply()([x3, x3])
    # x4_square = Multiply()([x4, x4])
    # x3_sq_x4_sq_diff = Subtract()([x3_square, x4_square])
    #
    # x3_x4_diff = Subtract()([x3, x4])
    # x3_x4_diff_sq = Multiply()([x3_x4_diff, x3_x4_diff])
    #
    # x3_x4_product = Multiply()([x3, x4])
    #
    # x3_x4_sum = Add()([x3, x4])  # x1 + x2
    #
    # x3_sqrt = Lambda(lambda tensor: signed_sqrt(tensor))(x3)
    # x4_sqrt = Lambda(lambda tensor: signed_sqrt(tensor))(x4)
    #
    # x3_sqrt_x4_sqrt_diff = Subtract()([x3_sqrt, x4_sqrt])
    # x3_sqrt_x4_sqrt_sum = Add()([x3_sqrt, x4_sqrt])
    #
    # x3_sq_x4_sq_diff = Conv2D(128, [1, 1])(x3_sq_x4_sq_diff)
    # x3_x4_diff_sq = Conv2D(128, [1, 1])(x3_x4_diff_sq)
    # x3_x4_product = Conv2D(128, [1, 1])(x3_x4_product)
    # x3_x4_sum = Conv2D(128, [1, 1])(x3_x4_sum)
    # x3_x4_diff = Conv2D(128, [1, 1])(x3_x4_diff)
    # x3_sqrt_x4_sqrt_diff = Conv2D(128, [1, 1])(x3_sqrt_x4_sqrt_diff)
    # x3_sqrt_x4_sqrt_sum = Conv2D(128, [1, 1])(x3_sqrt_x4_sqrt_sum)
    #
    #
    # x = Concatenate(axis=-1)([
    #     Flatten()(x3_sq_x4_sq_diff), x1_sq_x2_sq_diff,
    #     Flatten()(x3_x4_diff_sq), x1_x2_diff_sq,
    #     Flatten()(x3_x4_product), x1_x2_product,
    #     Flatten()(x3_x4_sum), x1_x2_sum,
    #     Flatten()(x3_x4_diff), x1_x2_diff,
    #     Flatten()(x3_sqrt_x4_sqrt_sum), x1_sqrt_x2_sqrt_sum,
    #     Flatten()(x3_sqrt_x4_sqrt_diff), x1_sqrt_x2_sqrt_diff
    # ])


    #################### Combination #5 FIW ################

    # x3 = Conv2D(128, [1, 1])(x3)
    # x4 = Conv2D(128, [1, 1])(x4)
    #
    #
    # x = Concatenate(axis=-1)(
    #     [Flatten()(x4), x2,
    #      Flatten()(x3), x1])

    ########################################################

    ################### Combination #4 FIW ##################

    # x1_x2_sum = Add()([x1, x2])  # x1 + x2
    # x1_x2_diff = Subtract()([x1, x2])  # x1 - x2
    #
    #
    # x3_x4_sum = Add()([x3, x4])  # x3 + x4
    # x3_x4_diff = Subtract()([x3, x4])  # x3 - x4
    #
    #
    # x3_x4_sum = Conv2D(128, [1, 1])(x3_x4_sum)  # (1, 128)
    # x3_x4_diff = Conv2D(128, [1, 1])(x3_x4_diff)  # (1, 128)
    #
    #
    # x = Concatenate(axis=-1)(
    #     [Flatten()(x3_x4_sum), x1_x2_sum,
    #      Flatten()(x3_x4_diff), x1_x2_diff])

    ###########################################################

    ############# Combination #3 FIW ################

    x1_x2_sum = Add()([x1, x2])  # x1 + x2
    x1_x2_diff = Subtract()([x1, x2])  # x1 - x2
    x1_x2_product = Multiply()([x1, x2])  # x1 * x2

    x3_x4_sum = Add()([x3, x4])  # x3 + x4
    x3_x4_diff = Subtract()([x3, x4])  # x3 - x4
    x3_x4_product = Multiply()([x3, x4])  # x3 * x4

    x3_x4_sum = Conv2D(128, [1, 1])(x3_x4_sum)  # (1, 128)
    x3_x4_diff = Conv2D(128, [1, 1])(x3_x4_diff)  # (1, 128)
    x3_x4_product = Conv2D(128, [1, 1])(x3_x4_product)  # (1, 128)

    x = Concatenate(axis=-1)(
        [Flatten()(x3_x4_sum), x1_x2_sum,
         Flatten()(x3_x4_diff), x1_x2_diff,
         Flatten()(x3_x4_product), x1_x2_product])

    ##########################################

    ################## Combination #2 FIW ########################

    # x1_square = Multiply()([x1, x1])
    # x2_square = Multiply()([x2, x2])
    # x1_sq_x2_sq_diff = Subtract()([x1_square, x2_square])
    #
    # x1_x2_diff = Subtract()([x1, x2])
    # x1_x2_diff_sq = Multiply()([x1_x2_diff, x1_x2_diff])
    #
    #
    # x3_square = Multiply()([x3, x3])
    # x4_square = Multiply()([x4, x4])
    # x3_sq_x4_sq_diff = Subtract()([x3_square, x4_square])
    #
    # x3_x4_diff = Subtract()([x3, x4])
    # x3_x4_diff_sq = Multiply()([x3_x4_diff, x3_x4_diff])
    #
    #
    # x3_sq_x4_sq_diff = Conv2D(128, [1, 1])(x3_sq_x4_sq_diff)
    # x3_x4_diff_sq = Conv2D(128, [1, 1])(x3_x4_diff_sq)
    #
    # x = Concatenate(axis=-1)([
    #     Flatten()(x3_sq_x4_sq_diff), x1_sq_x2_sq_diff,
    #     Flatten()(x3_x4_diff_sq), x1_x2_diff_sq
    # ])

    ################################################################

    ################## Combination #1 FIW ###################

    # x1_square = Multiply()([x1, x1])
    # x2_square = Multiply()([x2, x2])
    # x1_sq_x2_sq_diff = Subtract()([x1_square, x2_square])
    #
    # x1_x2_diff = Subtract()([x1, x2])
    # x1_x2_diff_sq = Multiply()([x1_x2_diff, x1_x2_diff])
    #
    # x1_x2_product = Multiply()([x1, x2])
    #
    # x3_square = Multiply()([x3, x3])
    # x4_square = Multiply()([x4, x4])
    # x3_sq_x4_sq_diff = Subtract()([x3_square, x4_square])
    #
    # x3_x4_diff = Subtract()([x3, x4])
    # x3_x4_diff_sq = Multiply()([x3_x4_diff, x3_x4_diff])
    #
    # x3_x4_product = Multiply()([x3, x4])
    #
    # x3_sq_x4_sq_diff = Conv2D(128, [1, 1])(x3_sq_x4_sq_diff)
    # x3_x4_diff_sq = Conv2D(128, [1, 1])(x3_x4_diff_sq)
    # x3_x4_product = Conv2D(128, [1, 1])(x3_x4_product)
    #
    # x = Concatenate(axis=-1)([
    #     Flatten()(x3_sq_x4_sq_diff), x1_sq_x2_sq_diff,
    #     Flatten()(x3_x4_diff_sq), x1_x2_diff_sq,
    #     Flatten()(x3_x4_product), x1_x2_product
    # ])


    #############################################


    ######## Combination 0 #################
    # x = Dense(128, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # out = Dense(1, activation="sigmoid")(x)

    ######### Combination 1 ################

    # x = Dense(128, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # x = Dense(64, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # out = Dense(1, activation="sigmoid")(x)

    ######### Combination 2 ################

    # x = Dense(256, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # x = Dense(128, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # x = Dense(64, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # out = Dense(1, activation="sigmoid")(x)

    ######### Combination 3 ################

    # x = Dense(512, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # x = Dense(256, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # x = Dense(128, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # x = Dense(64, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # out = Dense(1, activation="sigmoid")(x)

    ######### Combination 4 ################
    #
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.02)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.02)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.02)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.02)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.02)(x)
    out = Dense(1, activation="sigmoid")(x)

    ######## Combination 8 #################
    # x = Dense(64, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2, input_3, input_4], out)




    # input_1 = Input(shape=(224, 224, 3))
    # input_2 = Input(shape=(224, 224, 3))


    # base_model = VGGFace(model='resnet50', include_top=False)
    # for x in base_model.layers[:-3]:
    #     x.trainable = True

    # x1 = base_model(input_1)
    # x2 = base_model(input_2)


    # x1 = GlobalMaxPool2D()(x1)
    # x2 = GlobalAvgPool2D()(x2)

    # x3 = Subtract()([x1, x2])
    # x3 = Multiply()([x3, x3])
    #
    # x1_ = Multiply()([x1, x1])
    # x2_ = Multiply()([x2, x2])
    # x4 = Subtract()([x1_, x2_])
    #
    # x5 = Multiply()([x1, x2])
    #
    # x = Concatenate(axis=-1)([x3, x4, x5])
    # # # #     x = Dense(512, activation="relu")(x)
    # # # #     x = Dropout(0.03)(x)
    # x = Dense(128, activation="relu")(x)
    # x = Dropout(0.02)(x)
    # out = Dense(1, activation="sigmoid")(x)
    #
    # model = Model([input_1, input_2], out)
    #
    model.compile(loss="binary_crossentropy", metrics=['acc', auc], optimizer=Adam(0.00001))
    #
    model.summary()

    return model


def train():
    model = baseline_model()
    file_path = "checkpoints/vggface_facenet_flip_contrast_brightness_saturation.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.3, patience=30, verbose=1)
    callbacks_list = [checkpoint, reduce_on_plateau, tensorboard_callback]

    history = model.fit_generator(fiw_data_generator(images_root_dir, train_file_path, batch_size=16),
                                  use_multiprocessing=False,
                                  validation_data=fiw_data_generator(images_root_dir, val_file_path, batch_size=16),
                                  epochs=50, verbose=2,
                                  workers=1, callbacks=callbacks_list,
                                  steps_per_epoch=train_df.shape[0] // 16, validation_steps=val_df.shape[0] // 16)



if __name__ == "__main__":
    # prepare_images(Path("../FIW_dataset/FIDs_NEW/"), Path("../FIW_dataset/FIDs_NEW_square_crop/"))
    # prepare_images_ellipse(Path("../FIW_dataset/FIDs_NEW/"), Path("../FIW_dataset/FIDs_NEW_ellipse_crop/"))
    train()

