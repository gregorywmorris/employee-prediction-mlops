from zenml.steps import step
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@step
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    print("Evaluation Metrics:", metrics)
    return metrics

