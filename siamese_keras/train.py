from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract,Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras


from dataset import get_train_val, gen
from vggface import VGGFace


# val_famillies_list = ["F09","F04","F08","F06", "F02"]

auc = tf.keras.metrics.AUC()

logdir = "logs/scalars/" + "baseline"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = VGGFace(model='resnet50', include_top=False)

    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    x1 = GlobalMaxPool2D()(x1)
    x2 = GlobalAvgPool2D()(x2)

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])

    x5 = Multiply()([x1, x2])

    x = Concatenate(axis=-1)([x3, x4, x5])
    #     x = Dense(512, activation="relu")(x)
    #     x = Dropout(0.03)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.02)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc', auc], optimizer=Adam(0.00001))

    model.summary()

    return model


def train():
    model = baseline_model()

    train, val, train_person_to_images_map, val_person_to_images_map = get_train_val("F09")
    print(len(train), len(val), len(train_person_to_images_map), len(val_person_to_images_map))
    file_path = "checkpoints/data_augm/vgg_face.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.3, patience=30, verbose=1)
    callbacks_list = [checkpoint, reduce_on_plateau, tensorboard_callback]

    history = model.fit_generator(gen(train, train_person_to_images_map, batch_size=16),
                                  use_multiprocessing=False,
                                  validation_data=gen(val, val_person_to_images_map, batch_size=16),
                                  epochs=50, verbose=2,
                                  workers=1, callbacks=callbacks_list,
                                  steps_per_epoch=200, validation_steps=100)



if __name__ == "__main__":
    train()

