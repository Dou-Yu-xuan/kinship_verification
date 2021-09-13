import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras

import os
from insightface import model_zoo
from insightface.utils.face_align import norm_crop
from tqdm import tqdm
import cv2
import numpy as np

import dlib

predictor_path = "../weights/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)



def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed


def signed_sqrt(x):
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)



# source https://github.com/vuvko/fitw2020
def ensure_path(cur):
    if not cur.exists():
        os.makedirs(str(cur))
    return cur


def prepare_images(root_dir, output_dir):
    whitelist_dir = 'MID'
    detector = model_zoo.get_model('retinaface_r50_v1')
    detector.prepare(ctx_id=-1, nms=0.4)

    for family_path in tqdm(root_dir.iterdir()):
        for person_path in family_path.iterdir():
            if not person_path.is_dir() or not person_path.name.startswith(whitelist_dir):
                continue
            output_person = ensure_path(output_dir / person_path.relative_to(root_dir))
            for img_path in person_path.iterdir():
                img = cv2.imread(str(img_path))
                bbox, landmarks = detector.detect(img, threshold=0.5, scale=1.0)
                output_path = output_person / img_path.name
                if len(landmarks) < 1:
                    print('smth wrong with {}'.format(img_path))
                    continue
                warped_img = norm_crop(img, landmarks[0])
                cv2.imwrite(str(output_path), warped_img)


def crop_ellipse(image, center, axesLength):
    mask = np.zeros(image.shape[:2], np.uint8)

    angle = 0

    startAngle = 0

    endAngle = 360

    color = (255, 255, 255)

    thickness = -1

    mask = cv2.ellipse(mask, center, axesLength,
                       angle, startAngle, endAngle, color, thickness)

    res = cv2.bitwise_and(image, image, mask=mask)

    return res

# TODO: fix issues with not normal images (just part of face is visible, face is not horizontal), possibly with pre-aligning
def detect_n_crop(image, detector):
    dets = detector(image, 1)

    if len(dets) < 1:
        print("no face detected")
        return

    shape = predictor(image, dets[0])

    target_landmarks = [29, 1, 15, 8]  # nose, left and right edge of the cheeks, chin
    points = []
    for j in target_landmarks:
        points.append((shape.part(j).x,
                       shape.part(j).y))
    points = np.array(points, dtype=np.int32)

    width = max(abs(points[0][0] - points[1][0]), abs(points[0][0] - points[2][0])) * 0.9

    height = abs(points[0][1] - points[3][1])

    center = (points[0][0], points[0][1])

    res = crop_ellipse(image, center, (int(width), int(height)))

    return res


def prepare_images_ellipse(root_dir, output_dir):
    whitelist_dir = 'MID'

    for family_path in tqdm(root_dir.iterdir()):
        for person_path in family_path.iterdir():
            if not person_path.is_dir() or not person_path.name.startswith(whitelist_dir):
                continue
            output_person = ensure_path(output_dir / person_path.relative_to(root_dir))
            for img_path in person_path.iterdir():
                img = cv2.imread(str(img_path))
                output_path = output_person / img_path.name
                cropped = detect_n_crop(img, detector)
                if cropped is None:
                    print('smth wrong with {}'.format(img_path))
                    cv2.imwrite(str(output_path), img)
                else:
                    cv2.imwrite(str(output_path), cropped)
