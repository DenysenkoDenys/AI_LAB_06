import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
data = pd.read_csv(url)

print("Перші рядки набору даних:")
print(data.head(3))

target_col = "train_class"
feature_cols = ["origin", "destination", "train_type", "price"]

data_clean = data.dropna(subset=[target_col] + feature_cols).copy()

data_clean["price_group"] = pd.qcut(data_clean["price"], q=4, duplicates="drop")

X = data_clean[["origin", "destination", "train_type", "price_group"]]
y = data_clean[target_col]

encoder = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"),
     ["origin", "destination", "train_type", "price_group"])
])

model = Pipeline([
    ("preprocess", encoder),
    ("naive_bayes", MultinomialNB())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== Матриця неточностей (Confusion Matrix) ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Детальний звіт класифікації ===")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
