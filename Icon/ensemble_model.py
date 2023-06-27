from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class EnsembleModel:
    def __init__(self):
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('svm', SVC(kernel='rbf', probability=True)),
                ('dt', DecisionTreeClassifier()),
                ('rf', RandomForestClassifier()),
                ('gb', GradientBoostingClassifier()),
                ('ab', AdaBoostClassifier()),
                ('nb', GaussianNB()),
                ('knn', KNeighborsClassifier()),
            ],
            voting='soft'
        )

    def train(self, X_train, y_train):
        self.ensemble_model.fit(X_train, y_train)

    def predict_proba(self, X_test):
        return self.ensemble_model.predict_proba(X_test)

    def evaluate(self, X_train, y_train, X_test, y_test):
        self.train(X_train, y_train)
        train_accuracy = self.ensemble_model.score(X_train, y_train)
        test_accuracy = self.ensemble_model.score(X_test, y_test)
        y_pred = self.ensemble_model.predict(X_test)

        metrics = {}

        for clf_name, clf in self.ensemble_model.named_estimators_.items():
            clf_train_accuracy = clf.score(X_train, y_train)
            clf_test_accuracy = clf.score(X_test, y_test)
            clf_y_pred = clf.predict(X_test)
            clf_confusion = confusion_matrix(y_test, clf_y_pred)
            clf_report = classification_report(y_test, clf_y_pred)

            clf_metrics = {
                'Training Accuracy': clf_train_accuracy,
                'Test Accuracy': clf_test_accuracy,
                'Confusion Matrix': clf_confusion,
                'Classification Report': clf_report
            }

            metrics[clf_name] = clf_metrics

        return train_accuracy, test_accuracy, metrics
