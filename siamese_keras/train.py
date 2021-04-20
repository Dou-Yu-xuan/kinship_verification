from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract,Add,  Reshape, Lambda, Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras

from utils import signed_sqrt
from dataset import get_train_val, gen
from vggface import VGGFace
from facenet import FaceNet


# val_famillies_list = ["F09","F04","F08","F06", "F02"]

auc = tf.keras.metrics.AUC()

logdir = "logs/scalars/" + "vggface_facenet"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

IMG_SIZE_FN = 160
IMG_SIZE_VGG = 224

def baseline_model():
    input_1 = Input(shape=(IMG_SIZE_FN, IMG_SIZE_FN, 3))
    input_2 = Input(shape=(IMG_SIZE_FN, IMG_SIZE_FN, 3))
    input_3 = Input(shape=(IMG_SIZE_VGG, IMG_SIZE_VGG, 3))
    input_4 = Input(shape=(IMG_SIZE_VGG, IMG_SIZE_VGG, 3))

    model_vgg = VGGFace(model='resnet50', include_top=False)
    model_fn = FaceNet()

    for x in model_vgg.layers[:-3]:
        x.trainable = True

    x1 = model_fn(input_1)
    x2 = model_fn(input_2)
    x3 = model_vgg(input_3)
    x4 = model_vgg(input_4)

    x1 = Reshape((1, 1, 128))(x1)
    x2 = Reshape((1, 1, 128))(x2)
    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x1t = Lambda(lambda tensor: K.square(tensor))(x1)
    x2t = Lambda(lambda tensor: K.square(tensor))(x2)
    x3t = Lambda(lambda tensor: K.square(tensor))(x3)
    x4t = Lambda(lambda tensor: K.square(tensor))(x4)

    merged_add_fn = Add()([x1, x2])
    merged_add_vgg = Add()([x3, x4])
    merged_sub1_fn = Subtract()([x1, x2])
    merged_sub1_vgg = Subtract()([x3, x4])
    merged_sub2_fn = Subtract()([x2, x1])
    merged_sub2_vgg = Subtract()([x4, x3])
    merged_mul1_fn = Multiply()([x1, x2])
    merged_mul1_vgg = Multiply()([x3, x4])
    merged_sq1_fn = Add()([x1t, x2t])
    merged_sq1_vgg = Add()([x3t, x4t])
    merged_sqrt_fn = Lambda(lambda tensor: signed_sqrt(tensor))(merged_mul1_fn)
    merged_sqrt_vgg = Lambda(lambda tensor: signed_sqrt(tensor))(merged_mul1_vgg)

    merged_add_vgg = Conv2D(128, [1, 1])(merged_add_vgg)
    merged_sub1_vgg = Conv2D(128, [1, 1])(merged_sub1_vgg)
    merged_sub2_vgg = Conv2D(128, [1, 1])(merged_sub2_vgg)
    merged_mul1_vgg = Conv2D(128, [1, 1])(merged_mul1_vgg)
    merged_sq1_vgg = Conv2D(128, [1, 1])(merged_sq1_vgg)
    merged_sqrt_vgg = Conv2D(128, [1, 1])(merged_sqrt_vgg)

    merged = Concatenate(axis=-1)(
        [Flatten()(merged_add_vgg), (merged_add_fn), Flatten()(merged_sub1_vgg), (merged_sub1_fn),
         Flatten()(merged_sub2_vgg), (merged_sub2_fn), Flatten()(merged_mul1_vgg), (merged_mul1_fn),
         Flatten()(merged_sq1_vgg), (merged_sq1_fn), Flatten()(merged_sqrt_vgg), (merged_sqrt_fn)])

    merged = Dense(100, activation="relu")(merged)
    merged = Dropout(0.1)(merged)
    merged = Dense(25, activation="relu")(merged)
    merged = Dropout(0.1)(merged)
    out = Dense(1, activation="sigmoid")(merged)

    model = Model([input_1, input_2, input_3, input_4], out)

    # x1 = base_model(input_1)
    # x2 = base_model(input_2)
    #
    # x1 = GlobalMaxPool2D()(x1)
    # x2 = GlobalAvgPool2D()(x2)
    #
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
    # #     x = Dense(512, activation="relu")(x)
    # #     x = Dropout(0.03)(x)
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

    train, val, train_person_to_images_map, val_person_to_images_map = get_train_val("F09")
    print(len(train), len(val), len(train_person_to_images_map), len(val_person_to_images_map))
    file_path = "checkpoints/vggface_facenet.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.3, patience=30, verbose=1)
    callbacks_list = [checkpoint, reduce_on_plateau, tensorboard_callback]

    history = model.fit_generator(gen(train, train_person_to_images_map, batch_size=16),
                                  use_multiprocessing=False,
                                  validation_data=gen(val, val_person_to_images_map, batch_size=16),
                                  epochs=150, verbose=2,
                                  workers=1, callbacks=callbacks_list,
                                  steps_per_epoch=200, validation_steps=100)



if __name__ == "__main__":
    train()

