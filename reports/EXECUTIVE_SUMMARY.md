# Ozone — Synthèse pour décision

## 1. Executive summary

1. **Prioriser un diagnostic des conditions de travail**, tout en maintenant un chantier d’égalité professionnelle : les deux objectifs ne se mesurent pas avec les mêmes indicateurs. Sources S02, S04, S07.

2. **Les heures supplémentaires constituent le signal le plus convergent** : 30.1% de départs (130/432) contre 10.9% (118/1086), OR ajusté pénalisé 3.34. Ce n’est pas un effet causal. Sources S03, S06.

3. **Cibler le diagnostic du cumul heures supplémentaires–niveau codé 1** : 83/158 départs, soit 52.5%. Le code de niveau doit être validé avant de le traduire en population métier. Source S08.

4. **Croiser les méthodes plutôt que chercher un modèle plus complexe** : logistique et forêt classent conditions de travail puis parcours professionnel en tête par permutation. Source S04.

5. **Poursuivre l’audit d’égalité** : sur 1150 salariés d’entraînement comparables, l’écart standardisé de revenu femmes/hommes est de -1.02 %. Ni ce petit écart ni la faible importance du genre ne suffisent à conclure sur l’égalité. Source S07.

6. **Ne pas déployer un score individuel en l’état** : les AUC sont informatives mais la calibration et la validation temporelle manquent. La logistique détecte 32 % des départs au seuil 0,5, la forêt 24 %. Source S02.

7. **Évaluer les actions par des pilotes**, avec indicateurs et groupes comparables définis avant le lancement ; le rendement d’une intervention RH n’est pas estimé ici. Recommandation, pas résultat mesuré.

## 2. Dataset et méthodologie

1 518 salariés, 248 départs observés (16,34 %), cible Attrition Yes=1. 35 colonnes source ; 30 prédicteurs après exclusion de la cible, de l’identifiant et de trois constantes. Source originale intacte, baseline 01 sans changement de valeurs. Split stratifié 80/20 : 1 214/198 en entraînement et 304/50 en test. Prétraitement dans les pipelines, cinq plis identiques. La consolidation ne sélectionne aucun nouveau modèle. Sources S01–S02.

## 3. Performance des modèles

| model | roc_auc | roc_auc_low | roc_auc_high | average_precision | recall_at_05 | precision_at_05 |
|---|---|---|---|---|---|---|
| Logistique L2 | 0.869 | 0.797 | 0.925 | 0.685 | 0.320 | 0.800 |
| Random Forest 05 | 0.819 | 0.753 | 0.887 | 0.507 | 0.240 | 0.857 |
| Dummy prior | 0.500 | 0.500 | 0.500 | 0.164 | 0.000 | 0.000 |

IC de ROC-AUC : 500 bootstraps stratifiés du test, conditionnels aux modèles fixés. AP = average precision. Les intervalles ne couvrent pas toute la sélection ni le réentraînement. La différence des modèles est étudiée par bootstrap **apparié**, pas par chevauchement des IC. Source S11.

## 4. Facteurs associés au turnover

Heures supplémentaires : OR ajusté L2 3.34, intervalle de stabilité [2.57 ; 4.29]. Déplacements, satisfaction et niveau sont comparés selon des modalités explicites dans le [rapport statistique](MODEL_RESULTS.md). Les codes de satisfaction ne sont pas traduits en « faible/fort » sans dictionnaire. Sources S03 et S06.

## 5. Contribution des leviers RH

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

Baisse de ROC-AUC par permutation conjointe ; barres non additives. Le groupe « Gender Equality » contient seulement Gender. Il ne mesure pas les inégalités, tandis que les groupes de conditions et de carrière regroupent plusieurs variables. Les ablations avec réentraînement confirment la contribution des conditions de travail, sans mesurer un effet de politique RH. Sources S04–S05.

## 6. Sous-groupes prioritaires

| label | n | departures | rate | wilson_low | wilson_high | training_support_ok | test_support_ok |
|---|---|---|---|---|---|---|---|
| Niveau codé 1 · HS oui | 158 | 83 | 0.525 | 0.448 | 0.602 | True | True |
| Niveau codé 1 · HS non | 392 | 62 | 0.158 | 0.125 | 0.198 | True | True |
| Autres niveaux · HS oui | 274 | 47 | 0.172 | 0.132 | 0.221 | True | True |
| Autres niveaux · HS non | 694 | 56 | 0.081 | 0.063 | 0.103 | True | True |

Sous-groupes guidés par les analyses existantes. Support d’entraînement requis : au moins 80 salariés, 20 départs et 20 non-départs. Les tableaux détaillent séparément entraînement et test ; leur cohérence descriptive n’est pas une réplication indépendante. Pas de segmentation fine par sexe × niveau × heures sur les slides ; les petites cellules sont signalées dans l’audit. Source S08.

## 7. Recommandations Ozone

Donner la priorité au pilote horaires et récupération, sous réserve de valider les renforts, les économies et les engagements de service. Le [HR Action Business Case](HR_ACTION_BUSINESS_CASE.md) compare le budget, le seuil de rentabilité et les conditions de déploiement des actions horaires, déplacements et carrières. L’audit de rémunération et de promotion suit un chantier distinct. [Recommandations RH](HR_RECOMMENDATIONS.md)

## 8. Limites

Données observationnelles, sélection historique des modèles et hypothèses déjà exposées aux données, test réutilisé et seulement 50 départs en test. Chronologie des variables inconnue : fuite temporelle possible malgré un split et des pipelines corrects. Ni validation externe ni représentativité géographique établies. Scores non documentés, 23 incohérences d’ancienneté sur 20 salariés et corrélations entre groupes. Les importances et scores ne sont pas des effets causaux ; une AUC élevée ne garantit pas une probabilité individuelle fiable. Aucune mesure d’absence ; pas de résultat sur l’absentéisme et pas de taux annuel de turnover. Source : S01, S10, S12 ; voir `AUDIT.md` et le registre de sources.

**Traçabilité** : les identifiants S01–S14 renvoient au [registre exact des sources](tables/source_register.csv), qui indique script/fonction, notebook, sortie, méthode et population. Les chiffres des figures proviennent exclusivement de `reports/tables/`.
