from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def prepare_features(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    cuisine_features = tfidf.fit_transform(df["Cuisines"])

    numeric_features = df[
        ["Average Cost for two", "Aggregate rating", "Votes"]
    ].values

    X = hstack([cuisine_features, numeric_features])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Primary Cuisine"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
