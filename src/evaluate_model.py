from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

def evaluate_classifier(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print("Evaluation Metrics")
    print("------------------")
    print(f"Accuracy : {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall   : {recall}\n")

    print("Classification Report:\n")
    print(classification_report(y_test, predictions, zero_division=0))