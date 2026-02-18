# Projet Prédiction du Diabète – Pima Indians Diabetes Dataset

Ce projet vise à prédire la présence ou non du diabète chez des femmes 
Il contient deux fichiers principaux :
- `notebooks/01_data_exploration.ipynb` → exploration, nettoyage et standardisation
- `scripts/logistic.py` → entraînement et sauvegarde d’un modèle de régression logistique

## Structure du projet
projet-diabete/
├── data/
│   └── diabetes.csv               # dataset original
├── notebooks/
│   └── 01_data_exploration.ipynb  # nettoyage + standardisation
├── scripts/
│   └── logistic.py                # entraînement modèle + évaluation
├── models/
│   └── logreg_diabetes_model.pkl  # modèle sauvegardé
├── results/
│   └── accuracy_logreg.txt        # résultats bruts
└── README.md



## 1. Exploration et Nettoyage des données (01_data_exploration.ipynb)

### Ce qui a été fait

- Chargement du dataset original
- Visualisation des corrélations (heatmap)
- **Correction des valeurs impossibles médicalement** :
  - Glucose = 0 → impossible (hypoglycémie sévère incompatible avec la vie)
  - BloodPressure = 0 → impossible (arrêt cardiaque)
  - SkinThickness = 0 → très improbable (pli cutané mesurable)
  - Insulin = 0 → très rare / souvent valeur manquante
  - BMI = 0 → impossible (personne n’a un IMC de 0)

→ Toutes ces valeurs ont été remplacées par la **médiane** de la colonne (plus robuste que la moyenne face aux outliers).

- Gestion des outliers extrêmes (clipping au 1% / 99%) sur :
  - Insulin (valeurs > 500–600 très rares)
  - Pregnancies (> 13–14 grossesses très improbable)
  - BMI (> 60–65 très rare)

- **Standardisation** (moyenne = 0, écart-type = 1) de toutes les variables explicatives
- Sauvegarde du dataset propre → `diabetes_clean.csv`

### Point de vue médical rapide (explications simples)

| Variable              | Valeur 0 = réaliste ? | Pourquoi on remplace par la médiane ?                          | Valeurs extrêmes attendues ? |
|-----------------------|------------------------|------------------------------------------------------------------|-------------------------------|
| Glucose               | Non                   | < 40–50 → coma hypoglycémique. 0 = mesure ratée ou oubliée     | 70–200 mg/dL (jeûne)         |
| BloodPressure         | Non                   | < 50–60 → choc. 0 = mesure impossible                           | 80–140 mmHg                  |
| SkinThickness         | Non (très rare)       | Pli cutané < 7–8 mm très exceptionnel                           | 10–40 mm                     |
| Insulin               | Rare                  | Souvent non mesuré → valeur manquante codée en 0                | 2–300 µU/mL (jeûne)          |
| BMI                   | Non                   | IMC < 10 → famine sévère incompatible avec la grossesse         | 18–45 (grossesse)            |

→ Remplacer par la médiane est une bonne approximation quand on ne peut pas récupérer la vraie valeur.

## 2. Entraînement du modèle (logistic.py)

### Ce qui a été fait

- Chargement du dataset nettoyé (`diabetes_clean.csv`)
- Séparation train/test (80/20) – random_state=42 pour reproductibilité
- Entraînement d’une **Régression Logistique** (solver lbfgs, max_iter=1000)
- Évaluation sur le test set :
  - Accuracy
  - Matrice de confusion
- Sauvegarde du modèle → `../models/logreg_diabetes_model.pkl`
- Sauvegarde des métriques → `../results/accuracy_logreg.txt`

### Résultats attendus (exemple typique sur ce dataset)

- Accuracy ≈ **75–80 %** (assez bon pour une régression logistique simple)
- Matrice de confusion typique :
- [[80–90  15–25]     ← Non-diabétique (0)
[20–30  25–40]]    ← Diabétique    (1)



→ Le modèle est souvent meilleur pour détecter les non-diabétiques que les diabétiques (déséquilibre des classes : ~65% non-diabète / 35% diabète).

## Comment reproduire le projet

1. Installer les dépendances
 ```bash
 pip install pandas scikit-learn joblib seaborn matplotlib

Lancer le notebook d’exploration: jupyter notebook notebooks/01_data_exploration.ipynb
Lancer l’entraînement du modèle: python scripts/logistic.py
