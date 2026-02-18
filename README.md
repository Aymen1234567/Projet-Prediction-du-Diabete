# Projet Prédiction du Diabète – Pima Indians Diabetes Dataset

Ce projet a pour objectif de **prédire la présence ou non du diabète** chez des femmes en utilisant le dataset **Pima Indians Diabetes**.  

Il contient principalement :

- `notebooks/01_data_exploration.ipynb` → exploration, nettoyage et standardisation  
- `scripts/logistic.py` → entraînement et sauvegarde d’un modèle de régression logistique

---

## 📂 Structure du projet

```text
projet-diabete/
├── data/
│   └── diabetes.csv               # Dataset original
├── notebooks/
│   └── 01_data_exploration.ipynb  # Nettoyage + standardisation
├── scripts/
│   └── logistic.py                # Entraînement modèle + évaluation
├── models/
│   └── logreg_diabetes_model.pkl  # Modèle sauvegardé
├── results/
│   └── accuracy_logreg.txt        # Résultats bruts
└── README.md




1️⃣ Exploration et nettoyage des données (01_data_exploration.ipynb)
Étapes réalisées

Chargement du dataset original

Visualisation des corrélations (heatmap)

Correction des valeurs médicalement impossibles :

Variable	0 = réaliste ?	Pourquoi remplacer par la médiane ?	Valeurs extrêmes attendues
Glucose	Non	< 40–50 → coma hypoglycémique ; 0 = mesure ratée	70–200 mg/dL
BloodPressure	Non	< 50–60 → choc ; 0 = mesure impossible	80–140 mmHg
SkinThickness	Non (très rare)	Pli cutané < 7–8 mm très exceptionnel	10–40 mm
Insulin	Rare	Souvent non mesuré → valeur manquante codée en 0	2–300 µU/mL
BMI	Non	IMC < 10 → famine sévère incompatible	18–45

Les valeurs 0 ont été remplacées par la médiane de chaque colonne pour garder la robustesse face aux outliers.

Gestion des outliers extrêmes (clipping au 1% / 99%) sur :

Insulin (> 500–600 très rare)

Pregnancies (> 13–14 improbable)

BMI (> 60–65 très rare)

Standardisation : moyenne = 0, écart-type = 1 pour toutes les variables explicatives

Sauvegarde du dataset nettoyé → diabetes_clean.csv

2️⃣ Entraînement du modèle (logistic.py)
Étapes réalisées

Chargement du dataset nettoyé (diabetes_clean.csv)

Séparation train/test (80/20) – random_state=42 pour reproductibilité

Entraînement d’une Régression Logistique (solver='lbfgs', max_iter=1000)

Évaluation sur le test set :

Accuracy

Matrice de confusion

Sauvegarde du modèle → models/logreg_diabetes_model.pkl

Sauvegarde des métriques → results/accuracy_logreg.txt

Résultats typiques

Accuracy ≈ 75–80 %

Matrice de confusion typique :

[[80–90  15–25]  # Non-diabétique (0)
 [20–30  25–40]] # Diabétique (1)


Le modèle est souvent meilleur pour détecter les non-diabétiques que les diabétiques en raison du déséquilibre des classes (~65% non-diabète / 35% diabète).

⚙️ Reproduire le projet

Installer les dépendances :

pip install pandas scikit-learn joblib seaborn matplotlib


Lancer le notebook d’exploration :

jupyter notebook notebooks/01_data_exploration.ipynb


Lancer l’entraînement du modèle :

python scripts/logistic.py


Les résultats et le modèle entraîné seront sauvegardés dans results/ et models/.
