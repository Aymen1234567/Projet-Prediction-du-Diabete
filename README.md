# Projet Prédiction du Diabète – Pima Indians Diabetes Dataset

Ce projet a pour objectif de **prédire la présence ou non du diabète** chez des femmes en utilisant le dataset **Pima Indians Diabetes**.

Il contient principalement :
- `notebooks/01_data_exploration.ipynb` → exploration, nettoyage et standardisation
- `scripts/logistic.py` → entraînement et sauvegarde d'un modèle de régression logistique

---

## 📂 Structure du projet

```text
projet-diabete/
├── data/
│   └── diabetes.csv                  # Dataset original
├── notebooks/
│   └── 01_data_exploration.ipynb     # Nettoyage + standardisation
├── scripts/
│   └── logistic.py                   # Entraînement modèle + évaluation
├── models/
│   └── logreg_diabetes_model.pkl     # Modèle sauvegardé
├── results/
│   └── accuracy_logreg.txt           # Résultats bruts
└── README.md
```

---

## 1️⃣ Exploration et nettoyage des données (`01_data_exploration.ipynb`)

### Étapes réalisées

1. Chargement du dataset original
2. Visualisation des corrélations (heatmap)
3. Correction des valeurs médicalement impossibles :

| Variable        | 0 = réaliste ? | Pourquoi remplacer par la médiane ?                         | Valeurs attendues  |
|-----------------|----------------|-------------------------------------------------------------|--------------------|
| `Glucose`       | Non            | < 40–50 → coma hypoglycémique ; 0 = mesure ratée           | 70–200 mg/dL       |
| `BloodPressure` | Non            | < 50–60 → état de choc ; 0 = mesure impossible             | 80–140 mmHg        |
| `SkinThickness` | Non (très rare)| Pli cutané < 7–8 mm très exceptionnel                      | 10–40 mm           |
| `Insulin`       | Rare           | Souvent non mesuré → valeur manquante codée en 0           | 2–300 µU/mL        |
| `BMI`           | Non            | IMC < 10 → famine sévère incompatible avec les données     | 18–45              |

> Les valeurs à 0 ont été remplacées par la **médiane** de chaque colonne, pour une meilleure robustesse face aux outliers.

4. Gestion des outliers extrêmes (clipping au 1er / 99e percentile) sur :
   - `Insulin` (> 500–600, très rare)
   - `Pregnancies` (> 13–14, improbable)
   - `BMI` (> 60–65, très rare)

5. **Standardisation** : moyenne = 0, écart-type = 1 pour toutes les variables explicatives
6. Sauvegarde du dataset nettoyé → `diabetes_clean.csv`

---

## 2️⃣ Entraînement du modèle (`logistic.py`)

### Étapes réalisées

1. Chargement du dataset nettoyé (`diabetes_clean.csv`)
2. Séparation train/test (80/20) avec `random_state=42` pour la reproductibilité
3. Entraînement d'une **régression logistique** (`solver='lbfgs'`, `max_iter=1000`)
4. Évaluation sur le jeu de test :
   - Accuracy
   - Matrice de confusion
5. Sauvegarde du modèle → `models/logreg_diabetes_model.pkl`
6. Sauvegarde des métriques → `results/accuracy_logreg.txt`

### Résultats typiques

- **Accuracy ≈ 75–80 %**
- Matrice de confusion typique :

```
[[80–90   15–25]   # Vrais négatifs / Faux positifs  (non-diabétique)
 [20–30   25–40]]  # Faux négatifs  / Vrais positifs  (diabétique)
```

> Le modèle détecte mieux les **non-diabétiques** que les diabétiques, en raison du déséquilibre des classes (~65 % négatifs / ~35 % positifs).

---

## ⚙️ Reproduire le projet

### 1. Installer les dépendances

```bash
pip install pandas scikit-learn joblib seaborn matplotlib
```

### 2. Lancer le notebook d'exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 3. Entraîner le modèle

```bash
python scripts/logistic.py
```

Le modèle entraîné sera sauvegardé dans `models/logreg_diabetes_model.pkl` et les métriques dans `results/accuracy_logreg.txt`.

---

## 📦 Dépendances

| Package        | Usage                              |
|----------------|------------------------------------|
| `pandas`       | Manipulation des données           |
| `scikit-learn` | Modélisation et évaluation         |
| `joblib`       | Sauvegarde du modèle               |
| `seaborn`      | Visualisation (heatmap)            |
| `matplotlib`   | Visualisation (graphiques)         |
