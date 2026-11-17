# Ozone — Importance des groupes de variables RH

## Constats principaux

**Logistic Regression** : Working Conditions arrive en tête (baisse moyenne d’AUC 0.127), puis Career Development (0.076). Première variable individuelle : OverTime (0.072).

**Random Forest** : Working Conditions arrive en tête (baisse moyenne d’AUC 0.110), puis Career Development (0.046). Première variable individuelle : OverTime (0.076).

Ce classement décrit l’information utilisée par les modèles sur cet échantillon. Il motive des diagnostics RH et des hypothèses d’action, sans mesurer l’effet d’une intervention. Le rappel au seuil 0,5 reste limité malgré une performance de classement supérieure à la baseline.

## Protocole et performance observée

Source : `data/variantes/01_baseline.xlsx`. Cible issue du chargeur existant : Attrition Yes=1 (départ), No=0. 1 518 salariés, 30 variables, 248 départs ; 1 214 salariés/198 départs en entraînement et 304/50 en test. Split et cinq plis ordonnés vérifiés à l’identique des expériences 04/05. Pipelines sauvegardés réutilisés pour le test ; prétraitement réappris dans chaque pli pour les comparaisons. Paramètres retenus précédemment sur AP, dont class_weight=None pour les deux modèles ; la recherche existante avait aussi testé balanced. Seuil de classification fixé à 0,5. PR-AUC est ici l’average precision (AP), pas l’intégrale trapézoïdale.

| model | roc_auc | pr_auc_ap | recall | precision | roc_auc_ci_low | roc_auc_ci_high | pr_auc_ap_ci_low | pr_auc_ap_ci_high |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.869 | 0.685 | 0.320 | 0.800 | 0.797 | 0.925 | 0.566 | 0.798 |
| Random Forest | 0.819 | 0.507 | 0.240 | 0.857 | 0.753 | 0.887 | 0.399 | 0.671 |
| Dummy | 0.500 | 0.164 | 0.000 | 0.000 | 0.500 | 0.500 | 0.164 | 0.164 |

Validation croisée des modèles globaux sur l’entraînement (moyenne et écart-type des cinq plis ; pas des IC) :

| model | roc_auc_mean | roc_auc_std | pr_auc_ap_mean | pr_auc_ap_std | recall_mean | recall_std | precision_mean | precision_std |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.840 | 0.065 | 0.650 | 0.080 | 0.318 | 0.032 | 0.850 | 0.123 |
| Random Forest | 0.800 | 0.069 | 0.559 | 0.112 | 0.167 | 0.045 | 0.782 | 0.228 |

Les IC de performance proviennent de 500 bootstraps stratifiés du test, conditionnels au modèle entraîné. Le test a déjà été consulté dans les notebooks : cette analyse est exploratoire, sans nouvelle validation externe. Les scores CV ne sont pas imbriqués et peuvent être optimistes après la sélection historique des paramètres.

## Classement des groupes — constat prédictif

### Logistic Regression

| group | importance_mean | permutation_sd | bootstrap_ci_low | bootstrap_ci_high |
|---|---|---|---|---|
| Working Conditions | 0.127 | 0.017 | 0.091 | 0.164 |
| Career Development | 0.076 | 0.018 | 0.041 | 0.107 |
| Employee Profile | 0.023 | 0.011 | 0.003 | 0.045 |
| Compensation | 0.018 | 0.013 | -0.006 | 0.039 |
| Engagement & Satisfaction | 0.013 | 0.007 | -0.005 | 0.031 |
| Gender Equality | 0.007 | 0.003 | 0.001 | 0.012 |

### Random Forest

| group | importance_mean | permutation_sd | bootstrap_ci_low | bootstrap_ci_high |
|---|---|---|---|---|
| Working Conditions | 0.110 | 0.020 | 0.069 | 0.149 |
| Career Development | 0.046 | 0.021 | 0.005 | 0.084 |
| Employee Profile | 0.032 | 0.014 | 0.005 | 0.064 |
| Compensation | 0.025 | 0.017 | -0.003 | 0.055 |
| Engagement & Satisfaction | 0.013 | 0.007 | -0.004 | 0.032 |
| Gender Equality | -0.001 | 0.001 | -0.004 | 0.003 |

Permutation conjointe des colonnes originales avec le même ordre de lignes (30 répétitions), puis passage dans le pipeline figé. Importance = AUC initiale − AUC permutée. Les IC bootstrap rééchantillonnent les mêmes salariés pour les prédictions initiales et permutées et moyennent les 30 différences d’AUC. Ils ne couvrent ni le réentraînement ni toute la variabilité de sélection. Les quantiles/écarts-types des permutations décrivent une variabilité de permutation, pas un IC populationnel. Une valeur négative est conservée ; les importances ne sont ni additives ni des pourcentages d’explication.

## Retrait de groupes et information autonome

Le retrait réentraîne le modèle sans le groupe : Δ positif = perte de performance par rapport au global sur le même pli. Le modèle par groupe seul mesure une information autonome, pas une contribution unique. Les hyperparamètres restent fixes ; ces résultats ne comparent pas des modèles réduits réoptimisés. Les CSV contiennent les cinq résultats appariés, sans test t sur des plis dépendants.

| model | group | delta_roc_auc_mean | delta_roc_auc_std | delta_pr_auc_ap_mean | delta_pr_auc_ap_std |
|---|---|---|---|---|---|
| Logistic Regression | Career Development | 0.036 | 0.032 | 0.086 | 0.044 |
| Logistic Regression | Compensation | 0.002 | 0.010 | 0.005 | 0.015 |
| Logistic Regression | Employee Profile | 0.003 | 0.009 | 0.014 | 0.038 |
| Logistic Regression | Engagement & Satisfaction | 0.025 | 0.024 | 0.056 | 0.068 |
| Logistic Regression | Gender Equality | 0.001 | 0.003 | -0.002 | 0.008 |
| Logistic Regression | Working Conditions | 0.059 | 0.023 | 0.186 | 0.047 |
| Random Forest | Career Development | 0.009 | 0.013 | 0.028 | 0.047 |
| Random Forest | Compensation | -0.017 | 0.016 | -0.015 | 0.007 |
| Random Forest | Employee Profile | -0.009 | 0.013 | -0.015 | 0.055 |
| Random Forest | Engagement & Satisfaction | 0.008 | 0.024 | 0.018 | 0.022 |
| Random Forest | Gender Equality | -0.001 | 0.008 | -0.003 | 0.015 |
| Random Forest | Working Conditions | 0.031 | 0.018 | 0.080 | 0.042 |

| model | group | roc_auc | pr_auc_ap |
|---|---|---|---|
| Logistic Regression | Career Development | 0.732 | 0.384 |
| Logistic Regression | Compensation | 0.690 | 0.341 |
| Logistic Regression | Employee Profile | 0.668 | 0.385 |
| Logistic Regression | Engagement & Satisfaction | 0.630 | 0.273 |
| Logistic Regression | Gender Equality | 0.526 | 0.173 |
| Logistic Regression | Working Conditions | 0.706 | 0.367 |
| Random Forest | Career Development | 0.674 | 0.328 |
| Random Forest | Compensation | 0.684 | 0.373 |
| Random Forest | Employee Profile | 0.629 | 0.332 |
| Random Forest | Engagement & Satisfaction | 0.600 | 0.233 |
| Random Forest | Gender Equality | 0.526 | 0.173 |
| Random Forest | Working Conditions | 0.642 | 0.301 |

## Variables individuelles et directions des associations

### Logistic Regression

| group | importance_mean | permutation_sd |
|---|---|---|
| OverTime | 0.072 | 0.011 |
| NumCompaniesWorked | 0.023 | 0.006 |
| YearsSinceLastPromotion | 0.022 | 0.008 |
| BusinessTravel | 0.019 | 0.005 |
| EnvironmentSatisfaction | 0.019 | 0.006 |
| JobLevel | 0.017 | 0.008 |
| DistanceFromHome | 0.015 | 0.005 |
| JobSatisfaction | 0.012 | 0.005 |
| StockOptionLevel | 0.010 | 0.008 |
| JobRole | 0.008 | 0.005 |
| Gender | 0.007 | 0.003 |
| TrainingTimesLastYear | 0.007 | 0.004 |
| EducationField | 0.006 | 0.003 |
| Department | 0.006 | 0.006 |
| YearsInCurrentRole | 0.005 | 0.005 |

### Random Forest

| group | importance_mean | permutation_sd |
|---|---|---|
| OverTime | 0.076 | 0.013 |
| JobSatisfaction | 0.014 | 0.005 |
| BusinessTravel | 0.011 | 0.005 |
| DistanceFromHome | 0.009 | 0.004 |
| NumCompaniesWorked | 0.009 | 0.005 |
| JobLevel | 0.008 | 0.007 |
| MaritalStatus | 0.008 | 0.006 |
| Age | 0.008 | 0.009 |
| MonthlyIncome | 0.008 | 0.012 |
| EnvironmentSatisfaction | 0.007 | 0.007 |
| MonthlyRate | 0.004 | 0.004 |
| Department | 0.004 | 0.007 |
| StockOptionLevel | 0.004 | 0.009 |
| YearsSinceLastPromotion | 0.003 | 0.004 |
| JobRole | 0.003 | 0.006 |

| feature | category | reference | log_odds_difference | odds_ratio |
|---|---|---|---|---|
| OverTime | 1 versus 0 | 0 | 1.207 | 3.344 |
| BusinessTravel | Travel_Frequently | Non-Travel | 0.945 | 2.572 |
| JobInvolvement | 1 | 0 | 0.617 | 1.853 |
| MaritalStatus | Single | Divorced | 0.598 | 1.819 |
| JobRole | Laboratory Technician | Healthcare Representative | 0.490 | 1.633 |
| YearsSinceLastPromotion | +1 training SD | current value | 0.445 | 1.560 |
| RelationshipSatisfaction | 1 | 0 | 0.441 | 1.555 |
| StockOptionLevel | 1 | 0 | -0.772 | 0.462 |
| StockOptionLevel | 2 | 0 | -0.696 | 0.499 |
| JobLevel | 2 | 0 | -0.616 | 0.540 |
| JobSatisfaction | 4 | 0 | -0.525 | 0.592 |
| StockOptionLevel | 4 | 0 | -0.464 | 0.629 |
| EducationField | Medical | Human Resources | -0.439 | 0.645 |
| WorkLifeBalance | 3 | 0 | -0.346 | 0.708 |

Les différences de log-cotes positives indiquent une association conditionnelle positive avec le départ ; les négatives une association négative. Les OR catégoriels sont exp(β_modalité − β_référence), car le codage existant conserve toutes les modalités. Pour les quantitatives, OR par écart-type appris sur l’entraînement ; Gender compare Female à Male et OverTime Yes à No. Ces paramètres sont pénalisés L2 et sensibles aux variables corrélées. Aucun IC classique ni p-value ne leur est attribué : le codage complet est redondant et une inférence MLE classique après sélection serait invalide. Les OR ne sont pas des effets causaux.

## Conditions de travail versus égalité femmes-hommes

Le groupe « Gender Equality » contient uniquement Gender. Son importance n’évalue ni les inégalités ni le rendement d’une politique d’égalité. Les revenus et carrières sont affectés à leurs propres groupes, sans duplication. Comparer cinq variables de conditions de travail à une seule variable démographique ne constitue pas un test symétrique de politiques RH.

| model | group | importance_mean | bootstrap_ci_low | bootstrap_ci_high |
|---|---|---|---|---|
| Logistic Regression | Working Conditions | 0.127 | 0.091 | 0.164 |
| Logistic Regression | Gender Equality | 0.007 | 0.001 | 0.012 |
| Random Forest | Working Conditions | 0.110 | 0.069 | 0.149 |
| Random Forest | Gender Equality | -0.001 | -0.004 | 0.003 |

Audit descriptif sur les seuls salariés d’entraînement : standardisation avec les mêmes poids de métier/niveau sur les strates communes. Seuils de 1, 5 et 10 personnes de chaque genre par strate pour exposer la sensibilité à la couverture. Les écarts ci-dessous sont observés, sans test de significativité et sans ajustement pour l’expérience, le temps de travail ou le pays.

| minimum_per_gender_per_stratum | outcome | covered_n | total_n | n_strata | female | male | female_minus_male | female_vs_male_percent |
|---|---|---|---|---|---|---|---|---|
| 1.000 | attrition | 1205.000 | 1214.000 | 28.000 | 0.154 | 0.168 | -0.014 | -8.341 |
| 1.000 | income | 1205.000 | 1214.000 | 28.000 | 6472.462 | 6536.462 | -64.000 | -0.979 |
| 1.000 | salary_hike | 1205.000 | 1214.000 | 28.000 | 15.103 | 15.267 | -0.164 | -1.074 |
| 1.000 | promotion_delay | 1205.000 | 1214.000 | 28.000 | 2.324 | 2.192 | 0.131 | 5.993 |
| 5.000 | attrition | 1150.000 | 1214.000 | 18.000 | 0.156 | 0.168 | -0.012 | -6.989 |
| 5.000 | income | 1150.000 | 1214.000 | 18.000 | 6358.489 | 6424.070 | -65.581 | -1.021 |
| 5.000 | salary_hike | 1150.000 | 1214.000 | 18.000 | 15.083 | 15.237 | -0.154 | -1.010 |
| 5.000 | promotion_delay | 1150.000 | 1214.000 | 18.000 | 2.371 | 2.098 | 0.273 | 13.019 |
| 10.000 | attrition | 1043.000 | 1214.000 | 13.000 | 0.157 | 0.170 | -0.013 | -7.710 |
| 10.000 | income | 1043.000 | 1214.000 | 13.000 | 5764.200 | 5822.457 | -58.257 | -1.001 |
| 10.000 | salary_hike | 1043.000 | 1214.000 | 13.000 | 15.139 | 15.285 | -0.146 | -0.954 |
| 10.000 | promotion_delay | 1043.000 | 1214.000 | 13.000 | 2.233 | 2.000 | 0.233 | 11.651 |

Les CSV fournissent aussi les effectifs, moyennes et médianes brutes ainsi que les strates. Le délai depuis la dernière promotion n’est pas un taux de promotion et ne permet pas de conclure à un accès égal ou inégal aux promotions.

## Robustesse et limites

| variable_1 | variable_2 | spearman | pearson | group_1 | group_2 |
|---|---|---|---|---|---|
| JobLevel | MonthlyIncome | 0.903 | 0.932 | Career Development | Compensation |
| MonthlyIncome | TotalWorkingYears | 0.701 | 0.764 | Compensation | Career Development |
| PercentSalaryHike | PerformanceRating | 0.587 | 0.681 | Compensation | Career Development |
| Age | TotalWorkingYears | 0.648 | 0.667 | Employee Profile | Career Development |
| MonthlyIncome | YearsAtCompany | 0.470 | 0.509 | Compensation | Career Development |
| Age | JobLevel | 0.468 | 0.489 | Employee Profile | Career Development |
| Age | MonthlyIncome | 0.450 | 0.480 | Employee Profile | Compensation |
| MonthlyIncome | YearsInCurrentRole | 0.398 | 0.361 | Compensation | Career Development |

Corrélations numériques calculées sur l’entraînement uniquement ; associations nominales (V de Cramér non corrigé) exportées séparément. Des groupes corrélés peuvent se substituer : permutation et retrait ne mesurent pas la même chose. Permuter peut créer des combinaisons intergroupes irréalistes. La taille des groupes influe sur leur importance. Une analyse alternative transfère DistanceFromHome du profil aux conditions de travail (`assignment_sensitivity.csv`). Les petits écarts et IC recouvrants ne justifient pas un classement ferme. Les IC marginaux ne sont pas des tests multiples ni des IC de différences entre groupes.

Un seul split et 50 départs en test limitent la précision. Si les IC des performances approchent la baseline ou si les contributions sont proches de zéro, le classement reste fragile. Même au-dessus de la baseline, une validation temporelle indépendante est nécessaire. Scores hors dictionnaire et 23 incohérences d’ancienneté sur 20 lignes conservés selon le protocole existant. La variante 01 n’est pas une correction des valeurs. Chronologie, devise, unités et représentativité géographique inconnues ; une fuite temporelle ne peut pas être exclue par le seul split. Aucune causalité ni discrimination n’est établie.

SHAP : Non calculé : bibliothèque SHAP absente ; importances individuelles et groupées par permutation disponibles pour les deux modèles.

## Trois recommandations à tester

1. **Conditions de travail** : prioriser un diagnostic de charge, heures supplémentaires et déplacements ; tester un pilote de planning/récupération avec groupes comparables et suivi prospectif. Les associations prédictives motivent le diagnostic ; la baisse des départs reste une hypothèse à évaluer.

2. **Parcours et rétention** : confronter les signaux de carrière, rémunération et engagement aux entretiens de départ ; tester un accompagnement des mobilités et des managers. Les contributions mesurées ne démontrent pas l’effet d’une promotion ou d’une hausse de salaire.

3. **Égalité** : poursuivre l’audit des rémunérations à responsabilités et temps de travail comparables, collecter candidatures et décisions de promotion datées. Un faible signal prédictif du genre ne justifie pas de déprioriser l’égalité ; les écarts descriptifs doivent être vérifiés avec ces données supplémentaires.

## Reproduction

`python -m src.group_feature_analysis` depuis la racine avec les dépendances existantes. Résultats dans `outputs/group_feature_analysis/`, figures PNG à 300 DPI. Sources et modèles 04/05 inchangés ; versions, paramètres et empreintes dans `analysis_metadata.json`.
