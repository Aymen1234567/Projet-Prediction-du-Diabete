# logistic.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score, roc_curve
import joblib
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score





#  Charger le dataset nettoyé et normaliser
df = pd.read_csv("../notebooks/diabetes_clean.csv")

#  Séparer features et cible
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

#  séparer train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=32
)

#  Entraîner Regression Logistique
logreg = LogisticRegression(max_iter=1000,solver='lbfgs')

cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
print("Mean accuracy:", cv_scores.mean())

logreg.fit(X_train, y_train)

#  Prédictions et évaluation
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
ra=classification_report(y_test, y_pred)
print("précision :", accuracy)
print("matrice de  confusion:\n", cm)
print(classification_report(y_test, y_pred))




#  Sauvegarder le modèle
joblib.dump(logreg, "../models/logreg_diabetes_model.pkl")
print("Modèle Regression Logistique sauvegardé ✅")

#  Sauvegarder les résultats
with open("../results/accuracy_logreg.txt", "w") as f:
    f.write(f"précision: {accuracy}\n")
    f.write(f"matrice de confusion :\n{cm}\n")
    f.write(f"rapport: {accuracy}\n")
