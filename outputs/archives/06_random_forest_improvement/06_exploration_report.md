# 06 — Exploration pour améliorer le Random Forest

## Résultats observés

Validation externe : variation d’AP OOF agrégée de -0.018 ; amélioration sur 3/5 plis. Les scores agrégés mélangent des modèles ajustés dans des plis différents ; consulter aussi les scores appariés par pli.

| model | policy | average_precision | roc_auc | precision | recall | f2 | threshold | fp | tp | flagged_fraction |
|---|---|---|---|---|---|---|---|---|---|---|
| Reference | 0.5 | 0.541 | 0.799 | 0.767 | 0.167 | 0.198 | 0.500 | 10 | 33 | 0.035 |
| Reference | F2_interne | 0.541 | 0.799 | 0.307 | 0.747 | 0.581 | variable par pli | 334 | 148 | 0.397 |
| Optimise | 0.5 | 0.523 | 0.794 | 0.714 | 0.177 | 0.208 | 0.500 | 14 | 35 | 0.040 |
| Optimise | F2_interne | 0.523 | 0.794 | 0.310 | 0.788 | 0.602 | variable par pli | 347 | 156 | 0.414 |

| metric | bootstrap_low | bootstrap_high |
|---|---|---|
| delta_ap | -0.053 | 0.014 |
| delta_auc | -0.020 | 0.012 |

Le gain de classement n’est pas confirmé : l’intervalle conditionnel de différence d’AP contient zéro.

Test historique : AP 0.507 → 0.532, ROC-AUC 0.819 → 0.820. Au seuil F2 choisi sur entraînement (0.147), rappel 84.0%, précision 31.6%, 42 départs détectés et 91 faux positifs.

| model | policy | average_precision | roc_auc | precision | recall | f2 | threshold | fp | tp | flagged_fraction |
|---|---|---|---|---|---|---|---|---|---|---|
| RF 05 | 0.5 | 0.507 | 0.819 | 0.857 | 0.240 | 0.280 | 0.500 | 2 | 12 | 0.046 |
| RF 05 | F2_entrainement | 0.507 | 0.819 | 0.336 | 0.860 | 0.655 | 0.145 | 85 | 43 | 0.421 |
| RF 06 | 0.5 | 0.532 | 0.820 | 0.750 | 0.240 | 0.278 | 0.500 | 4 | 12 | 0.053 |
| RF 06 | F2_entrainement | 0.532 | 0.820 | 0.316 | 0.840 | 0.631 | 0.147 | 91 | 42 | 0.438 |
| Logistique 04 | 0.5 | 0.685 | 0.869 | 0.800 | 0.320 | 0.364 | 0.500 | 4 | 16 | 0.066 |
| Dummy | 0.5 | 0.164 | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 0 | 0 | 0.000 |

La référence bénéficie elle aussi du réglage du seuil : rappel 86.0%, précision 33.6%, F2 0.655, contre F2 0.631 pour le candidat. Le choix du seuil et le choix du modèle sont donc deux décisions distinctes. La logistique reste une référence concurrente à comparer sur les mêmes objectifs métier.

## Configuration choisie sur entraînement

```json
{
  "candidate": 2,
  "subset": "Sans remuneration",
  "score_encoding": "numeric",
  "n_estimators": 200,
  "max_depth": null,
  "min_samples_leaf": 2,
  "max_features": "sqrt",
  "class_weight": null,
  "max_samples": null,
  "selected_columns": [
    "Age",
    "BusinessTravel",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "NumCompaniesWorked",
    "OverTime",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
  ],
  "f2_threshold": 0.14662202380952377,
  "reference_f2_threshold": 0.14508333333333334,
  "selection": "mean inner-fold average precision; threshold maximizes OOF F2"
}
```

## Interprétation et limites

La sélection optimise l’average precision, pas le score test. Le seuil maximise F2 sur des prédictions hors pli d’entraînement : le rappel reçoit davantage de poids que la précision. Ce choix est exploratoire, à remplacer par un objectif métier de capacité/coût ; il ne modifie ni l’AUC ni l’AP du modèle. Le réglage du seuil de la référence est également évalué pour distinguer l’effet du seuil de celui du modèle.

Cinq plis externes identiques aux expériences précédentes ; trois plis internes par pli externe pour les 20 configurations. La configuration et le seuil sont choisis sans les labels du pli externe. La sélection finale réutilise les cinq plis sur tout l’entraînement. Les recherches incluent des retraits de groupes et un encodage one-hot des scores pour éviter de supposer leur ordre. Pas de changement des valeurs sources, pas de SMOTE hors pipeline.

Les hypothèses de retrait et la référence viennent de travaux ayant déjà consulté ces données, parfois le test. La validation imbriquée contrôle la nouvelle recherche, sans effacer ce contexte exploratoire. Les IC bootstrap sur les prédictions externes sont conditionnels aux modèles et ne représentent pas toute la variabilité du réentraînement ; les plis ont des ensembles d’apprentissage qui se recouvrent. Aucun test de significativité fondé sur cinq plis supposés indépendants. Une différence faible ou un intervalle contenant zéro ne démontre pas une amélioration robuste.

20 configurations forment un budget limité, pas une recherche exhaustive. Une hausse sur le seul test ne suffit pas à choisir le modèle. Aucun essai additionnel n’est décidé après son évaluation. Le test existant sert de comparaison historique, pas de validation externe indépendante. Les scores hors dictionnaire et incohérences d’ancienneté restent présents. Les associations prédictives ne mesurent pas l’effet causal d’une politique RH.

## Suite proposée

Confronter le gain aux résultats externes par pli et aux intervalles. Valider un seuil avec la capacité d’accompagnement RH puis évaluer prospectivement sur une nouvelle cohorte temporelle. Conserver la logistique comme référence concurrente et ne pas remplacer le RF 05 sur la seule base du test.
