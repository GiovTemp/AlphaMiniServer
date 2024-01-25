import cv2
import numpy as np
import dlib
import pandas as pd

def detect_landmarks(image_path):
    # Load the pre-trained face detector model
    detector_face = dlib.get_frontal_face_detector()
    predictor_path = "models/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    # Load an image
    image = cv2.imread(image_path)

    # Convert the image to grayscale (required by Dlib)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector_face(gray)

    landmarks = predictor(gray, faces[0])
    landmarks_coordinates = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

    return landmarks_coordinates


def landmarks_combination_df(landmarks):
    columns_names = [f'Landmark_x_{i}' for i in range(68)]
    columns_names1 = [f'Landmark_y_{i}' for i in range(68)]

    all_columns_names = columns_names + columns_names1

    vec_comb_lin_x = []
    vec_comb_lin_y = []

    # Dividi l'array in coordinate x e y
    land_x = landmarks[:, 0]  # Estrae tutte le colonne 0 (coordinate x)
    land_y = landmarks[:, 1]  # Estrae tutte le colonne 1 (coordinate y)

    for i in range(68):
        sum_x = 0
        sum_y = 0

        for j in range(68):
            sum_x = sum_x + land_x[i] * (1 / land_x[j])
            sum_y = sum_y + land_y[i] * (1 / land_y[j])

        vec_comb_lin_x.append(sum_x)
        vec_comb_lin_y.append(sum_y)

    all_landmark = vec_comb_lin_x + vec_comb_lin_y

    return pd.DataFrame([all_landmark], columns=all_columns_names)