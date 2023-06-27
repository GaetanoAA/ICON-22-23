from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from image_loader import ImageLoader
from feature_extractor import FeatureExtractor
from ensemble_model import EnsembleModel
from prolog_reasoner import PrologReasoner
from classifier_evaluator import ClassifierEvaluator
from Bayesian_network import BayesianNetwork

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Caricamento delle immagini
folder_path = "cell_images"
sample_size = 200  # 27558 numero max
image_loader = ImageLoader(folder_path)
images, labels = image_loader.load_images_and_labels(sample_size)

# Estrazione delle feature
feature_extractor = FeatureExtractor()
features = feature_extractor.extract_features(images)

# Codifica delle etichette
label_encoder = LabelEncoder()
label_encoder.fit(labels)
labels_encoded = label_encoder.transform(labels)

# Divisione dei dati in train set e test set
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Creazione del modello di ensemble
ensemble_model = EnsembleModel()
train_accuracy, test_accuracy, metrics = ensemble_model.evaluate(X_train, y_train, X_test, y_test)

# Reasoning con Prolog
kb_file_path = "kb.pl"
threshold = 170
# Reasoning con Prolog
prolog_reasoner = PrologReasoner(kb_file_path, threshold)
count_parasitized, count_uninfected, macchie_violacee, no_macchie_violacee = prolog_reasoner.perform_reasoning()

# Decodifica delle etichette per valutazione del classificatore
y_test_decoded = label_encoder.inverse_transform(y_test)
classifier_evaluator = ClassifierEvaluator(label_encoder)
y_scores = ensemble_model.predict_proba(X_test)[:, 1]
auc, fpr, tpr = classifier_evaluator.evaluate_roc_auc(y_test_decoded, y_scores)


print("\nTraining Accuracy (Ensemble): {:.2f}%".format(train_accuracy * 100))
print("Test Accuracy (Ensemble): {:.2f}%".format(test_accuracy * 100))


for clf_name, clf_metrics in metrics.items():
    print(f"\nAlgorithm: {clf_name}")
    print("Training Accuracy: {:.2f}%".format(clf_metrics['Training Accuracy'] * 100))
    print("Test Accuracy: {:.2f}%".format(clf_metrics['Test Accuracy'] * 100))
    print("\nConfusion Matrix:")
    print(clf_metrics['Confusion Matrix'])
    print("\nClassification Report:")
    print(clf_metrics['Classification Report'])

print("\nNumber of Parasitized Images:", count_parasitized)
print("Number of Uninfected Images:", count_uninfected)

# Creazione dell'istanza della classe BayesianNetwork
model = BayesianNetwork()
model.build_model()

# Esempio di utilizzo con una cartella
folder_path = 'cell_images'  # Sostituisci 'path_to_folder' con il percorso completo della cartella
probability_above, probability_below = model.calculate_probabilities_from_blue_channel(folder_path)
print('\nProbability (Blue > 160):', probability_above)
print('Probability (Blue < 160):', probability_below)

print("\nImmagini con macchie violacee:")
for i, solution in enumerate(macchie_violacee):
    if i >= 5:
        break
    print(solution["X"])

print("\nImmagini senza macchie violacee:")
for i, solution in enumerate(no_macchie_violacee):
    if i >= 5:
        break
    print(solution["X"])

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()
