import os
import cv2
from feature_extractor import FeatureExtractor


# Funzione per ottenere il nome dell'immagine senza estensione
def get_image_name(file_name):
    return os.path.splitext(file_name)[0]


# Funzione per creare la base di conoscenza Prolog dalle etichette e dalle features delle immagini
def create_prolog_kb(folder_path, kb_file_path, feature_extractor):
    kb = ""
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(('.jpg', '.png')):
                image_name = get_image_name(file_name)
                label = os.path.basename(root)

                # Carica l'immagine e estrai le features
                image_path = os.path.join(root, file_name)
                image = cv2.imread(image_path)
                features = feature_extractor.extract_features([image])[0]

                # Aggiungiamo le features alla base di conoscenza Prolog
                kb += f"immagine('{image_name}', '{label}', {list(features)}).\n"

    kb += "\n"

    # Aggiungiamo regole e predicati alla base di conoscenza Prolog
    kb += ":- dynamic immagine/3.  % Fatto dinamico per permettere l'aggiunta di nuovi fatti\n\n"
    kb += "% Regole per il riconoscimento delle immagini infette e non infette\n"
    kb += "parasitized(X) :- immagine(X, 'Parasitized', _).\n"
    kb += "uninfected(X) :- immagine(X, 'Uninfected', _).\n\n"
    kb += "above_threshold(B) :-\n"
    kb += "    immagine(_, _, L),\n"
    kb += "    L = [_ | Rest],\n"
    kb += "    last(Rest, B),\n"
    kb += "    B > 160.0.\n\n"
    kb += "below_threshold(B) :-\n"
    kb += "    immagine(_, _, L),\n"
    kb += "    L = [_ | Rest],\n"
    kb += "    last(Rest, B),\n"
    kb += "    B < 160.0.\n\n"
    kb += "% Regola: count_parasitized(N) - Conta il numero di immagini infette\n"
    kb += "count_parasitized(N) :- findall(_, immagine(_, 'Parasitized', _), List), length(List, N).\n\n"
    kb += "% Regola: count_uninfected(N) - Conta il numero di immagini non infette\n"
    kb += "count_uninfected(N) :- findall(_, immagine(_, 'Uninfected', _), List), length(List, N).\n\n"
    kb += "% Regola: random_parasitized(X) - Seleziona casualmente un'immagine infetta\n"
    kb += "random_parasitized(X) :- immagine(X, 'Parasitized', _), random(X).\n\n"
    kb += "% Regola: find_image_by_name(X, Name) - Trova un'immagine per nome\n"
    kb += "find_image_by_name(X, Name) :- immagine(X, Name, _).\n\n"
    kb += "% Regola: find_image_by_real_name(X, L, Data) - Trova un'immagine per nome reale e dati\n"
    kb += "find_image_by_real_name(X, L, Data) :- immagine(_, X, Data), immagine(_, L, Data).\n"

    # Salva la base di conoscenza Prolog nel file specificato
    with open(kb_file_path, "w") as kb_file:
        kb_file.write(kb)

    print("Base di conoscenza Prolog creata e salvata correttamente.")


if __name__ == "__main__":
    # Specifica il percorso della cartella in cui sono presenti le immagini
    folder_path = "cell_images"

    # Specifica il percorso e il nome del file per salvare la base di conoscenza Prolog
    kb_file_path = "KB.pl"

    # Creazione della base di conoscenza Prolog dal dataset di immagini
    feature_extractor = FeatureExtractor()
    create_prolog_kb(folder_path, kb_file_path, feature_extractor)
