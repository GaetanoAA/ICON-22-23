import os
import cv2
import numpy as np
from PIL import Image
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def extract_dominant_blue(image_path):

    image = Image.open(image_path)
    image = image.convert('RGB')
    image_array = np.array(image)

    # Compressione dell'immagine
    compressed_image = cv2.resize(image_array, (1, 1))

    # Estrazione del colore dominante (blu)
    dominant_blue = compressed_image[0, 0, 2]

    return dominant_blue


class BayesianNetwork:
    def __init__(self):
        self.cpd_uninfected = None
        self.model = BayesianModel([('blue', 'parasitized'), ('blue', 'uninfected')])
        self.cpd_parasitized = TabularCPD('parasitized', 2, [[0.5, 0.5], [0.5, 0.5]], evidence=['blue'],
                                          evidence_card=[2])
        self.cpd_uninfected = TabularCPD('uninfected', 2, [[0.5, 0.5], [0.5, 0.5]], evidence=['blue'],
                                         evidence_card=[2])

    def build_model(self):

        self.model.add_cpds(self.cpd_parasitized, self.cpd_uninfected)

    def calculate_probabilities_from_blue_channel(self, folder_path):

        count_above = 0
        count_below = 0
        total = 0  # Inizializza total a zero

        # Scorrere i file all'interno della cartella e delle sottocartelle
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                # Verifica se il file è un'immagine (puoi aggiungere ulteriori controlli se necessario)
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    file_path = os.path.join(root, file_name)
                    blue_value = extract_dominant_blue(file_path)  # Estrae il valore blu dominante
                    # print(file_path, blue_value)
                    if blue_value > 160:
                        count_above += 1
                    else:
                        count_below += 1
                    total += 1  # Incrementa total per ogni immagine valida

        # Controlla se total è zero per evitare la divisione per zero
        if total == 0:
            probability_above = 0
            probability_below = 0
        else:
            probability_above = count_above / total
            probability_below = count_below / total

        self.cpd_parasitized.values = [[probability_above], [probability_below]]
        self.cpd_uninfected.values = [[1 - probability_above], [1 - probability_below]]

        return probability_above, probability_below

