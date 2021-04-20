from tensorflow.keras.models import load_model


def FaceNet():
    model_path = 'weights/facenet_keras.h5'
    model = load_model(model_path)

    for layer in model.layers[:-3]:
        layer.trainable = True

    return model

