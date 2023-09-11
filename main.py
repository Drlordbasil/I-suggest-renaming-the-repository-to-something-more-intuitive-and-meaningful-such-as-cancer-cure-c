import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


class CancerDataLoader:
    def load_data(self, file_path):
        """
        Load the cancer dataset from a given file path.

        Args:
        - file_path: str, the path of the dataset file.

        Returns:
        - X: numpy array of shape (n_samples, n_features), the features matrix.
        - y: numpy array of shape (n_samples,), the target values.
        """
        data = pd.read_csv(file_path)
        X = data.drop('target', axis=1).values
        y = data['target'].values
        return X, y


class CancerDataPreprocessor:
    def preprocess_data(self, X, y):
        """
        Preprocess the cancer dataset.

        Args:
        - X: numpy array of shape (n_samples, n_features), the features matrix.
        - y: numpy array of shape (n_samples,), the target values.

        Returns:
        - X_train: numpy array of shape (n_train_samples, n_features), the preprocessed training features matrix.
        - X_test: numpy array of shape (n_test_samples, n_features), the preprocessed testing features matrix.
        - y_train: numpy array of shape (n_train_samples,), the training target values.
        - y_test: numpy array of shape (n_test_samples,), the testing target values.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Perform any necessary preprocessing steps
        # (e.g., feature scaling, missing value imputation, etc.)

        return X_train, X_test, y_train, y_test


class CancerModelTrainer:
    def train_model(self, X_train, y_train):
        """
        Train a classifier model using the training data.

        Args:
        - X_train: numpy array of shape (n_train_samples, n_features), the training features matrix.
        - y_train: numpy array of shape (n_train_samples,), the training target values.

        Returns:
        - classifier: trained classifier model.
        """
        classifier = SVC()
        classifier.fit(X_train, y_train)
        return classifier


class CancerModelEvaluator:
    def evaluate_model(self, classifier, X_test, y_test):
        """
        Evaluate the classifier model using the test data.

        Args:
        - classifier: trained classifier model.
        - X_test: numpy array of shape (n_test_samples, n_features), the testing features matrix.
        - y_test: numpy array of shape (n_test_samples,), the testing target values.

        Returns:
        - report: str, the classification report.
        - cm: numpy array of shape (n_classes, n_classes), the confusion matrix.
        """
        y_pred = classifier.predict(X_test)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return report, cm


class CancerModelVisualizer:
    def visualize_results(self, report, cm):
        """
        Visualize the evaluation results.

        Args:
        - report: str, the classification report.
        - cm: numpy array of shape (n_classes, n_classes), the confusion matrix.
        """
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Benign', 'Malignant'])
        plt.yticks([0, 1], ['Benign', 'Malignant'])
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Save the plot to a file instead of showing it
        plt.savefig('confusion_matrix.png')

        # Print the classification report to a file instead of console
        with open('classification_report.txt', 'w') as f:
            print(report, file=f)


class CancerResearchProgram:
    def __init__(self):
        self.data_loader = CancerDataLoader()
        self.data_preprocessor = CancerDataPreprocessor()
        self.model_trainer = CancerModelTrainer()
        self.model_evaluator = CancerModelEvaluator()
        self.model_visualizer = CancerModelVisualizer()

    def run(self, file_path):
        # Load the data
        X, y = self.data_loader.load_data(file_path)

        # Preprocess the data
        X_train, X_test, y_train, y_test = self.data_preprocessor.preprocess_data(
            X, y)

        # Train the model
        classifier = self.model_trainer.train_model(X_train, y_train)

        # Evaluate the model
        report, cm = self.model_evaluator.evaluate_model(
            classifier, X_test, y_test)

        # Visualize the results
        self.model_visualizer.visualize_results(report, cm)


# Define the file path of the cancer dataset
file_path = "cancer_dataset.csv"

# Create an instance of the CancerResearchProgram and run it
program = CancerResearchProgram()
program.run(file_path)
