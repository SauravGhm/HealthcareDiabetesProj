from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    """
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    return lr_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, file_path)