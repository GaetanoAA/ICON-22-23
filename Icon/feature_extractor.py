import numpy as np
import cv2


class FeatureExtractor:
    @staticmethod
    def extract_features(images):
        features = []

        for image in images:
            color_mean = np.mean(image, axis=(0, 1))
            color_std = np.std(image, axis=(0, 1))

            # Calcolo dell'istogramma dei colori nel dominio HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hist_hsv, _ = np.histogram(hsv_image[:, :, 0], bins=8, range=(0, 180))
            hist_hsv = hist_hsv / np.sum(hist_hsv)  # Normalizzazione

            # Calcolo dell'istogramma dei colori nel dominio LAB
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            hist_lab, _ = np.histogram(lab_image[:, :, 1:], bins=8, range=(-127, 128))
            hist_lab = hist_lab / np.sum(hist_lab)  # Normalizzazione

            # Compressione dell'immagine e estrazione del colore dominante
            compressed_image = cv2.resize(image, (1, 1))
            dominant_color = compressed_image[0, 0]

            feature = np.concatenate((color_mean, color_std, hist_hsv, hist_lab, dominant_color))
            features.append(feature)

        return np.array(features)
