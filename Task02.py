import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

data = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"],
]

df = pd.DataFrame(data, columns=["Outlook", "Temperature", "Humidity", "Wind", "Play"])

X = df[["Outlook", "Temperature", "Humidity", "Wind"]]
y = df["Play"]

enc = OrdinalEncoder()
X_enc = enc.fit_transform(X)
clf = CategoricalNB()
clf.fit(X_enc, y)

x9 = pd.DataFrame([["Sunny", "Mild", "Normal", "Strong"]], columns=X.columns)
x9_enc = enc.transform(x9)

pred = clf.predict(x9_enc)[0]
proba = clf.predict_proba(x9_enc)[0]

print("Вихідні дані: Outlook=Sunny, Humidity=Normal, Wind=Strong")
print(f"Прогноз: {pred}")
print(f"Ймовірності класів {list(clf.classes_)}: {proba.round(3)}")
