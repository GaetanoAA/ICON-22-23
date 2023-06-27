from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.exceptions import NotFittedError


class ClassifierEvaluator:
    def __init__(self, label_encoder):
        """
        Inizializza un oggetto ClassifierEvaluator.

        Args:
            label_encoder: Istanza del label encoder utilizzato per codificare le etichette.

        Returns:
            None
        """
        self.label_encoder = label_encoder

    def evaluate_roc_auc(self, y_test, y_scores):

        try:
            y_test_binary = self.label_encoder.transform(y_test)
            auc = roc_auc_score(y_test_binary, y_scores)
            fpr, tpr, _ = roc_curve(y_test_binary, y_scores)
            return auc, fpr, tpr
        except NotFittedError:
            raise NotFittedError(
                "L'istanza del LabelEncoder non è stata addestrata ancora. Chiamare 'fit' con gli argomenti "
                "appropriati prima di utilizzare questo stimatore")
