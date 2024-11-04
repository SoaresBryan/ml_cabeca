import numpy as np

def extract_features(face_landmarks):
    pontos = face_landmarks.landmark
    pontos_indices = [1, 33, 61, 199, 263, 291, 17, 57]
    features = []
    for idx in pontos_indices:
        ponto = pontos[idx]
        features.extend([ponto.x, ponto.y, ponto.z])
    return features
