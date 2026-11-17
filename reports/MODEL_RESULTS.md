# Résultats statistiques et contrôle de provenance

**Traçabilité** : les identifiants S01–S14 renvoient au [registre exact des sources](tables/source_register.csv), qui indique script/fonction, notebook, sortie, méthode et population. Les chiffres des figures proviennent exclusivement de `reports/tables/`.

## Données, encodage et absence de fuite de partition

Source : `data/employee.csv.xlsx`, séparée par `src/data.py::load_data` ; modèle sur `data/variantes/01_baseline.xlsx`. Égalité cellule par cellule vérifiée sur les colonnes communes. Cible Attrition Yes=1 confirmée. `src/modeling.py::make_split` reproduit exactement les indices sauvegardés 04/05 ; train et test disjoints, plis sur train seulement. Le scaler logistique, le one-hot et les modèles sont ajustés sur train. Le notebook 04 compare 6 configurations, le 05 en compare 8 avec class_weight=None/balanced ; sélection sur AP. Paramètres retenus non pondérés. La logistique standardise les quantitatives et encode scores/nominales ; la forêt traite les scores comme ordonnés. Le pipeline exclut Attrition/depart, identifiant et constantes. Les données des variantes 02/03 ne remplacent pas la baseline. Sources S01, S10.

Les pipelines sauvegardés ont été reproduits par clonage et ajustement sur les seuls indices train ; écart maximal de prédictions avec les fichiers existants inférieur à 1e-12. Cette vérification et le code étayent la provenance de l’ajustement, sans prouver une antériorité temporelle des variables. Voir `tables/model_replay_checks.csv`, méthode `src/reporting.py::performance_checks`.

## Performance sur test

| model | n_test | departures_test | roc_auc | average_precision | precision_at_05 | recall_at_05 | brier_score | roc_auc_low | roc_auc_high | average_precision_low | average_precision_high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistique L2 | 304 | 50 | 0.869 | 0.685 | 0.800 | 0.320 | 0.088 | 0.797 | 0.925 | 0.566 | 0.798 |
| Random Forest 05 | 304 | 50 | 0.819 | 0.507 | 0.857 | 0.240 | 0.109 | 0.753 | 0.887 | 0.399 | 0.671 |
| Dummy prior | 304 | 50 | 0.500 | 0.164 | 0.000 | 0.000 | 0.137 | 0.500 | 0.500 | 0.164 | 0.164 |

La Brier score est la moyenne des erreurs quadratiques des probabilités (plus bas = mieux), pas une certification de calibration. Aucune courbe de calibration externe suffisamment documentée n’est disponible. Précision/rappel au seuil fixé de 0,5 ; AP sans interpolation trapézoïdale. Sources S02.

### Incertitude et comparaison appariée

| comparison | auc_difference | paired_bootstrap_low | paired_bootstrap_high | bootstrap_n |
|---|---|---|---|---|
| Logistique - RF 05 | 0.050 | -0.000 | 0.096 | 500 |

Différence AUC logistique − forêt : 0.0498, intervalle apparié [-0.0002 ; 0.0957]. Le calcul réutilise les mêmes salariés bootstrap pour les deux modèles. Ce résultat conditionnel sur un test déjà consulté n’établit pas une supériorité générale. Aucun test n’est déduit du chevauchement des IC marginaux. Source S11.

### Validation croisée historique

| modele | average_precision | average_precision_ecart_type | roc_auc | roc_auc_ecart_type |
|---|---|---|---|---|
| Logistique | 0.650 | 0.071 | 0.840 | 0.058 |
| RandomForest | 0.559 | 0.100 | 0.800 | 0.061 |
| Baseline | 0.163 | 0.002 | 0.500 | 0.000 |

Écarts-types entre cinq plis, pas des IC. Scores de sélection non imbriqués, potentiellement optimistes ; ne pas les agréger avec les scores test comme observations indépendantes. Source S02.

## Associations ajustées : contrastes et stabilité

| contrast | odds_ratio | stability_low | stability_high | positive_fraction | bootstrap_n |
|---|---|---|---|---|---|
| Heures supplémentaires : Oui / Non | 3.344 | 2.573 | 4.292 | 1.000 | 300 |
| Déplacements : fréquents / rares | 1.900 | 1.342 | 2.617 | 1.000 | 300 |
| Satisfaction environnement : code 1 / 4 | 1.961 | 1.354 | 2.690 | 1.000 | 300 |
| Satisfaction travail : code 1 / 4 | 2.226 | 1.641 | 3.080 | 1.000 | 300 |
| Niveau hiérarchique : code 1 / 2 | 2.464 | 1.756 | 3.431 | 1.000 | 300 |
| Genre : femmes / hommes | 0.818 | 0.637 | 1.049 | 0.057 | 300 |

Source S03 : `src/reporting.py::selected_log_odds` et `bootstrap_associations`. Même logistique L2 et C=0,1 ; 300 bootstraps stratifiés de l’entraînement, encodeur et scaler réajustés à chaque tirage. Pour chaque contraste catégoriel, OR=exp(β_modalité−β_référence) malgré le one-hot complet. OverTime et Gender sont binaires (Yes/Female=1). Pas de comparaison brute des amplitudes entre unités différentes. Les références du graphique sont des modalités courantes explicites, pas automatiquement le code 0. Les intervalles percentiles décrivent la stabilité de l’estimateur pénalisé conditionnel à sa sélection : ils ne garantissent pas une couverture nominale de paramètres non pénalisés, n’intègrent pas tout le biais de régularisation et ne constituent pas des tests post-sélection. Aucun p-value ajouté. Les OR ne sont ni des risques relatifs ni des effets causaux.

### Écart moyen de scores standardisés pour OverTime

| variable | mean_model_score_difference | stability_low | stability_high | population_n | bootstrap_n | interpretation |
|---|---|---|---|---|---|---|
| OverTime Yes - No | 0.142 | 0.109 | 0.167 | 1214.000 | 300 | Ecart moyen de scores standardises sur X_train fixe; non causal, probabilites non calibrees |

Sur la population train fixe, différence moyenne de scores prédits Yes−No : 14.24 points, intervalle de stabilité [10.86 ; 16.67] points. Les deux prédictions conservent les autres variables observées. Il s’agit d’un contraste de modèle avec probabilités non présumées calibrées ; certaines combinaisons peuvent être peu représentées. Ne pas l’annoncer comme une réduction attendue de turnover si les heures sont supprimées. Source S09.

## Contribution prédictive des groupes

| model | group | importance_mean | bootstrap_ci_low | bootstrap_ci_high |
|---|---|---|---|---|
| Logistique | Working Conditions | 0.127 | 0.091 | 0.164 |
| Logistique | Career Development | 0.076 | 0.041 | 0.107 |
| Logistique | Employee Profile | 0.023 | 0.003 | 0.045 |
| Logistique | Compensation | 0.018 | -0.006 | 0.039 |
| Logistique | Engagement & Satisfaction | 0.013 | -0.005 | 0.031 |
| Logistique | Gender Equality | 0.007 | 0.001 | 0.012 |
| Random Forest | Working Conditions | 0.110 | 0.069 | 0.149 |
| Random Forest | Career Development | 0.046 | 0.005 | 0.084 |
| Random Forest | Employee Profile | 0.032 | 0.005 | 0.064 |
| Random Forest | Compensation | 0.025 | -0.003 | 0.055 |
| Random Forest | Engagement & Satisfaction | 0.013 | -0.004 | 0.032 |
| Random Forest | Gender Equality | -0.001 | -0.004 | 0.003 |

Permutation : mêmes lignes permutées pour toutes les colonnes originales du groupe, relations internes conservées, pipeline figé, 30 répétitions. IC bootstrap conditionnels à 95 % sur 500 tirages du test ; variabilité de permutation également disponible dans les sorties sources. Les importances ne s’additionnent pas et ne sont pas des pourcentages d’explication. La sensibilité au transfert de DistanceFromHome vers les conditions de travail est disponible dans `outputs/group_feature_analysis/assignment_sensitivity.csv`. Source S04.

### Ablation avec réentraînement sur les mêmes plis

| model | group | delta_auc_mean | delta_auc_sd | delta_ap_mean | positive_auc_folds |
|---|---|---|---|---|---|
| Logistic Regression | Career Development | 0.036 | 0.032 | 0.086 | 4 |
| Logistic Regression | Compensation | 0.002 | 0.010 | 0.005 | 3 |
| Logistic Regression | Employee Profile | 0.003 | 0.009 | 0.014 | 3 |
| Logistic Regression | Engagement & Satisfaction | 0.025 | 0.024 | 0.056 | 4 |
| Logistic Regression | Gender Equality | 0.001 | 0.003 | -0.002 | 3 |
| Logistic Regression | Working Conditions | 0.059 | 0.023 | 0.186 | 5 |
| Random Forest | Career Development | 0.009 | 0.013 | 0.028 | 4 |
| Random Forest | Compensation | -0.017 | 0.016 | -0.015 | 1 |
| Random Forest | Employee Profile | -0.009 | 0.013 | -0.015 | 2 |
| Random Forest | Engagement & Satisfaction | 0.008 | 0.024 | 0.018 | 3 |
| Random Forest | Gender Equality | -0.001 | 0.008 | -0.003 | 2 |
| Random Forest | Working Conditions | 0.031 | 0.018 | 0.080 | 5 |

Delta positif = perte d’AUC/AP après retrait, modèles réajustés à paramètres fixes. Un groupe seul mesure une information autonome, pas sa contribution unique ; résultats conservés dans `outputs/group_feature_analysis/group_only_model_results.csv`. Les conditions de travail restent les plus contributives en ablation moyenne ; les autres petits écarts ne sont pas des priorités fermes. Source S05.

### Corrélations entre groupes

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

Corrélations sur entraînement uniquement. Métier/département et niveau/revenu se recouvrent ; un groupe corrélé peut se substituer à un autre. Permutation marginale et retrait réentraîné ne mesurent pas la même chose. Les tailles des groupes diffèrent et leur affectation est une convention métier. Source S12.

## Descriptifs et sous-groupes

| variable | category | n | departures | rate | wilson_low | wilson_high |
|---|---|---|---|---|---|---|
| OverTime | No | 1086 | 118 | 0.109 | 0.092 | 0.129 |
| OverTime | Yes | 432 | 130 | 0.301 | 0.260 | 0.346 |
| BusinessTravel | Non-Travel | 157 | 12 | 0.076 | 0.044 | 0.129 |
| BusinessTravel | Travel_Frequently | 287 | 75 | 0.261 | 0.214 | 0.315 |
| BusinessTravel | Travel_Rarely | 1074 | 161 | 0.150 | 0.130 | 0.173 |
| Gender | Female | 605 | 89 | 0.147 | 0.121 | 0.178 |
| Gender | Male | 913 | 159 | 0.174 | 0.151 | 0.200 |

| part | level_1 | overtime | label | n | departures | rate | wilson_low | wilson_high | training_support_ok | test_support_ok |
|---|---|---|---|---|---|---|---|---|---|---|
| ensemble | True | True | Niveau codé 1 · HS oui | 158 | 83 | 0.525 | 0.448 | 0.602 | True | True |
| ensemble | True | False | Niveau codé 1 · HS non | 392 | 62 | 0.158 | 0.125 | 0.198 | True | True |
| ensemble | False | True | Autres niveaux · HS oui | 274 | 47 | 0.172 | 0.132 | 0.221 | True | True |
| ensemble | False | False | Autres niveaux · HS non | 694 | 56 | 0.081 | 0.063 | 0.103 | True | True |
| entrainement | True | True | Niveau codé 1 · HS oui | 119 | 62 | 0.521 | 0.432 | 0.609 | True | True |
| entrainement | True | False | Niveau codé 1 · HS non | 313 | 50 | 0.160 | 0.123 | 0.204 | True | True |
| entrainement | False | True | Autres niveaux · HS oui | 232 | 40 | 0.172 | 0.129 | 0.226 | True | True |
| entrainement | False | False | Autres niveaux · HS non | 550 | 46 | 0.084 | 0.063 | 0.110 | True | True |
| test | True | True | Niveau codé 1 · HS oui | 39 | 21 | 0.538 | 0.386 | 0.684 | True | True |
| test | True | False | Niveau codé 1 · HS non | 79 | 12 | 0.152 | 0.089 | 0.247 | True | True |
| test | False | True | Autres niveaux · HS oui | 42 | 7 | 0.167 | 0.083 | 0.306 | True | True |
| test | False | False | Autres niveaux · HS non | 144 | 10 | 0.069 | 0.038 | 0.123 | True | True |

Proportion = départs/effectif ; IC de Wilson à 95 % sans correction de comparaisons multiples. Les sous-groupes OverTime × niveau codé 1 sont une hypothèse guidée par les analyses existantes, pas une découverte indépendante. Le regroupement « autres niveaux » comprend aussi le code 0 : le dictionnaire doit être validé. Règle de support fixée sur train : n≥80, départs≥20, non-départs≥20. Le test est détaillé avec un indicateur de support n≥20, départs≥5, non-départs≥10, sans sélection des taux les plus élevés. Un effectif suffisant ne garantit ni la représentativité ni l’absence de confusion. Les petites strates historiques sont conservées en annexe dans `tables/historical_subgroup_review.csv`. Aucune interaction du Random Forest n’est prouvée par ses seules importances ; les écarts entre strates ne sont pas des effets de traitement. Sources S06, S08.

## Égalité femmes-hommes : périmètre et couverture

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

Périmètre principal : entraînement, standardisation sur les mêmes poids JobRole × JobLevel, strates contenant au moins 5 salariés de chaque genre : 1150/1 214 personnes. Le résultat brut et la version sur toute la base sont dans le notebook 02 et `outputs/tables/02_egalite_femmes_hommes/02_standardisation.csv`. Les écarts de population, de poids et de filtre expliquent les différences numériques entre ces sorties ; les seuils 1/5/10 montrent la sensibilité à la couverture. Aucun ajustement complet sur temps de travail, expérience, responsabilités ou pays. Aucun IC d’écart salarial ni preuve de discrimination. YearsSinceLastPromotion est un délai, pas une probabilité de promotion. Source S07.

## Pourquoi ne pas multiplier les forêts

Le notebook 06 archivé a exploré 20 configurations, encodages, retraits et seuils. AP externe agrégée : 0,541 pour la référence contre 0,523 pour la procédure optimisée ; différence conditionnelle [-0,053 ; 0,014]. AP test du candidat : 0,532 contre 0,507, sans gain robuste confirmé. L’abaissement du seuil de la référence (appris sur train) donne 43/50 départs détectés mais 85 faux positifs ; ce compromis dépend d’un coût métier non validé. Le modèle 06 ne remplace pas le 05. Source S13.

SHAP absent de l’environnement : aucune figure ni conclusion SHAP. Pas d’autre algorithme testé identifié dans les sources du dépôt. Les résultats non linéaires de la forêt servent seulement de triangulation avec la logistique. Source S14.

## Limites et reproductibilité

Données observationnelles, sélection historique des modèles et hypothèses déjà exposées aux données, test réutilisé et seulement 50 départs en test. Chronologie des variables inconnue : fuite temporelle possible malgré un split et des pipelines corrects. Ni validation externe ni représentativité géographique établies. Scores non documentés, 23 incohérences d’ancienneté sur 20 salariés et corrélations entre groupes. Les importances et scores ne sont pas des effets causaux ; une AUC élevée ne garantit pas une probabilité individuelle fiable. Aucune mesure d’absence ; pas de résultat sur l’absentéisme et pas de taux annuel de turnover. Source : S01, S10, S12 ; voir `AUDIT.md` et le registre de sources.

`python -m src.reporting` régénère tableaux, rapports et quatre figures PNG/SVG. Les modèles historiques restent en lecture seule. `reports/consolidation_metadata.json` contient leurs empreintes et celles des données ; 300 réajustements L2 sont justifiés uniquement par les intervalles de stabilité manquants. Les recherches/permutations déjà réalisées ne sont pas rejouées.
