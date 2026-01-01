import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import load_and_clean_data
from feature_engineering import prepare_features
from train_model import train_classifier
from evaluate_model import evaluate_classifier

DATA_PATH = "data/Dataset.csv"

def main():
    print("\nTask 3: Cuisine Classification\n")

    print("Loading and preprocessing data...")
    df = load_and_clean_data(DATA_PATH)

    X_train, X_test, y_train, y_test = prepare_features(df)

    print("Training cuisine classification model...")
    model = train_classifier(X_train, y_train)

    print("Evaluating model...\n")
    evaluate_classifier(model, X_test, y_test)

if __name__ == "__main__":
    main()