# 🍽️ Cuisine Classification System

## 🎯 Objective
The objective of this project is to develop a **machine learning classification model**
that predicts the **primary cuisine of a restaurant** using real-world restaurant data.

---

## 📊 Dataset Description
The dataset contains restaurant-related attributes such as:
- Cuisines
- Average Cost for Two
- Aggregate Rating
- Votes

For classification, the **primary cuisine** (the first cuisine listed) is considered
as the target label.

---

## 🛠️ Project Workflow

### 1. Data Preprocessing
- Removed records with missing cuisine information
- Handled missing numerical values using median or default values
- Extracted the primary cuisine from multiple cuisine entries
- Removed extremely rare cuisine classes to reduce class imbalance

### 2. Feature Engineering
- Applied **TF-IDF vectorization** on cuisine text data
- Combined text-based features with numerical features
- Encoded cuisine labels
- Used stratified train-test split to preserve class distribution

### 3. Model Training
- Implemented **Random Forest Classifier**
- Chosen for its ability to handle non-linear relationships and class imbalance

### 4. Model Evaluation
The model was evaluated using:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- Classification Report

---

## 📈 Results

- **Accuracy:** ~82%
- **Weighted Precision:** ~81%
- **Weighted Recall:** ~82%

The model performs well for frequently occurring cuisines.
Lower performance for rare cuisines is expected due to limited training samples.

---

## 📁 Folder Structure

```
Cuisine_Classification/
│
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Dataset.csv
│
└── src/
    ├── data_preprocessing.py
    ├── feature_engineering.py
    ├── train_model.py
    └── evaluate_model.py
```

---

## ▶️ How to Run the Project

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

---

## 🔍 Observations
Cuisine classification is challenging due to:
- Large number of cuisine categories
- Class imbalance
- Similar or overlapping cuisine names

Despite these challenges, the model demonstrates a complete
end-to-end machine learning pipeline.

---

## 🧾 Conclusion
This project successfully implements a cuisine classification system using real-world
restaurant data. It demonstrates practical skills in data preprocessing, feature
engineering, model training, and evaluation, making it suitable for academic and
portfolio purposes.

---

## 👤 Author
**Sanjay**