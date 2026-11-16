import kagglehub
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

path = kagglehub.dataset_download("yasserh/titanic-dataset")

files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(files[0])

df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True, dtype=int)

X = df.drop("Survived", axis=1)
y = df["Survived"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Titanic Confusion Matrix")
plt.show()