# Audit et décisions — Ozone RH

Audit du 17 novembre 2026. Périmètre : données, huit notebooks, modules `src/`, script de préparation, tests, CSV/JSON, modèles, graphiques et guide PDF. Aucun fichier source ou résultat n'est supprimé. Les fichiers de travail personnels `.DS_Store`, `.git/` et `.venv/` sont hors périmètre analytique.

## Décision avant réorganisation

La priorité est un diagnostic RH, pas la recherche d'un score individuel maximal. Les analyses 00–05 et les analyses par groupes constituent le socle. Le notebook 01b devient une annexe pédagogique ; le 06 et ses sorties deviennent une archive d'exploration reproductible. Le PDF historique est archivé : sa page 5 indique un split 75/25, une imputation et une AUC de 0,833, incompatibles avec le protocole courant 80/20 sans imputation. Les données et modèles 04/05 restent à leurs chemins contractuels afin de conserver la provenance et les contrôles d'empreintes.

## Inventaire des analyses et arbitrages

| Analyse / source | Objectif et méthode | Résultat disponible | Fiabilité, limites et intérêt RH | Décision |
|---|---|---|---|---|
| `src/data.py`, `src/quality.py`, notebook 00 | Lecture CSV encapsulé dans Excel ; audit brut des 35 colonnes | 1 518 lignes, 248 départs ; pas de vides ; 23 incohérences d'ancienneté sur 20 lignes | Pas de dictionnaire validé ; IQR ne signifie pas erreur. Indispensable au cadrage | Conserver ; détail en annexe technique |
| `scripts/create_excel_variants.py`, `data/variantes/` | Baseline et deux simulations tracées | 01 : retrait identifiant/constantes ; 02 : 20 lignes retirées ; 03 : 26 cellules plafonnées | 02/03 ne sont pas des corrections validées ; aucune étude de sensibilité modèle complète sur ces variantes | Conserver comme préparation et simulations, ne pas régénérer pendant l'audit |
| Notebook 01, `src/features.py` | Corrélations, redondances, candidats descriptifs sur entraînement | Métier/département redondants ; indices d'ancienneté et référence salariale | Candidats non validés comme facteurs ; pas d'amélioration modèle démontrée | Conserver pour diagnostic ; aucune figure descriptive générique dans les slides |
| Notebook 01b | Démonstration encodage, standardisation, apprentissage par pli | Contrôles pédagogiques, aucun modèle supplémentaire | Redondant dans le parcours décisionnel ; utile à la formation | Archiver dans `notebooks/archives/` |
| Notebook 02 et tables égalité | Départs, rémunération, représentation ; standardisation métier/niveau | Femmes 89/605 ; hommes 159/913 ; revenus standardisés sur toute la base | Certaines strates petites ; genre seul ne mesure pas l'égalité ; pas de taux de promotion | Conserver ; consolider avec l'analyse d'entraînement et sa couverture |
| Notebook 03 et tables conditions | Proportions de départ, effectifs ; strates genre/niveau/OverTime | OverTime : 130/432 contre 118/1 086 | Association brute ; temporalité et volume horaire inconnus ; petites strates à écarter | Conserver ; ajouter intervalles et filtre explicite de sous-groupes |
| Notebook 04, `src/modeling.py`, modèles `04_*` | Logistique L2, sélection AP sur cinq plis ; test réservé | AUC 0,868661 ; AP 0,685237 ; rappel 0,32 au seuil 0,5 | One-hot complet : OR entre modalités via différences de coefficients ; pas de p-values classiques. CV après sélection optimiste | Conserver comme modèle principal d'association ; compléter stabilité des contrastes |
| Notebook 05, modèles `05_*` | Forêt 200 arbres, même split et plis | AUC 0,818819 ; AP 0,506820 ; rappel 0,24 | Complément non linéaire ; pas de preuve d'une interaction précise ; scores ordonnés par hypothèse | Conserver comme triangulation, pas comme preuve causale |
| `src/group_feature_analysis.py`, `outputs/group_feature_analysis/` | 30 permutations conjointes par groupe, ablations réentraînées, modèles par groupe seul | Conditions de travail et carrière en tête ; IC AUC test et importances disponibles | IC conditionnels ; permutation intergroupes peut sortir du support ; groupes inégaux et corrélés ; mapping métier discutable et documenté | Conserver et consolider, sans recalculer les recherches/permutations |
| Coefficients et `logistic_odds_ratios.csv` | Contrastes corrects malgré le one-hot complet | OR ponctuels L2 disponibles, aucun IC de coefficient | Références automatiques parfois code 0 inhabituel ; ne pas comparer amplitudes entre unités | Améliorer avec contrastes RH explicites et bootstrap de stabilité, sans p-values |
| SHAP | Tentative optionnelle dans le script groupes | Bibliothèque absente, aucune sortie SHAP exploitable | Ne pas présenter comme analyse réalisée | Écarter des slides ; documenter l'absence |
| Notebook 06, `src/forest_improvement.py` | 20 configurations, validation imbriquée 5 × 3, seuil F2 appris sur OOF | AP test 0,532 ; AP externe agrégée 0,523 contre 0,541 ; différence bootstrap [-0,053 ; 0,014] | Pas de gain robuste. Hypothèses héritées d'analyses déjà exposées aux données ; test déjà consulté | Archiver notebook et sorties ; garder code et tests pour reproduction |
| Seuil F2 du 06 | Comparaison décisionnelle, seuil appris sur entraînement | RF 05 : rappel 86 %, précision 33,6 %, 85 faux positifs en test | Pas un gain d'AUC ; coût/capacité RH non validés ; pas de déploiement proposé | Annexe uniquement |
| Guide PDF historique, 6 pages | Explication d'un ancien parcours | Split 75/25, AUC 0,833 et anciens chemins, page 5 | Obsolète pour la modélisation actuelle ; ne pas réutiliser ses chiffres | Archiver dans `docs/archives/`, conserver intégralement |
| Figures historiques `outputs/figures/` et figures groupes | Exploration et diagnostics de modèles | 16 figures historiques et 6 figures groupes (comptage vérifié par inventaire généré) | Formats variés ; certaines descriptives ou doublons de présentation | Conserver comme preuves techniques ; utiliser seulement `reports/figures/` pour les slides |

## Doublons et écarts documentaires

- Les exports `*_evaluation`, `*_validation_croisee`, `*_comparaison_performances` et `05_comparaison_modeles` se recouvrent volontairement. Ne pas les supprimer : les notebooks les consomment. `reports/tables/model_performance.csv` devient la vue consolidée.
- La baseline et les indices 04/05 sont redondants pour comparaison, mais nécessaires à la vérification exacte des expériences. Conserver leurs fichiers.
- La standardisation du notebook 02 porte sur toute la base ; celle des groupes porte sur l'entraînement et filtre les tailles de strates. Chiffres différents ne signifient pas contradiction. Toujours annoncer le périmètre et la couverture.
- `data/README.md` mentionne à tort les versions « 03/04 » ; il s'agit de 02/03. Le guide des tables indique d'anciens chemins de traçabilité et quatre Excel au lieu de trois variantes : à corriger.
- Les coefficients complets 04 et leur réexport groupes ne sont pas deux estimations différentes : le second explicite les contrastes. Une seule table ciblée est utilisée en présentation.
- Les graphiques finaux ne doivent pas être mélangés avec ceux de l'exploration 06 ou du PDF historique.

## Points méthodologiques à ne pas masquer

1. Split et apprentissage : vérifier source, indices ordonnés, colonnes exclues et prétraitement dans chaque fit. La vérification ne prouve pas la disponibilité temporelle des variables.
2. Le test est historique et déjà consulté ; toute sélection à partir de ce test réduit son indépendance. Aucune nouvelle sélection de modèle ou de variable n'est conduite ici.
3. Les IC de performance sont des percentiles sur 500 bootstraps stratifiés du test, conditionnels aux modèles ajustés et à la composition en classes.
4. Le chevauchement des IC de modèles ne teste pas leur différence. Calculer une différence appariée sur les mêmes rééchantillonnages, sans revendiquer une supériorité universelle.
5. Les nouveaux intervalles d'OR décrivent la stabilité de l'estimateur L2 à pénalité fixée. Ni IC MLE classiques, ni significativité après sélection, ni effets causaux.
6. Les sous-groupes sont exploratoires, préspecifiés avant leur nouvelle comparaison : OverTime × niveau codé 1. Une répétition train/test vérifie seulement une cohérence descriptive, pas une réplication indépendante ni une interaction causale.
7. Aucune donnée d'absence, date de départ, durée d'exposition, pays ni établissement. Ni absentéisme, ni taux annuel, ni extrapolation géographique.
8. Les scores peuvent être hors dictionnaire supposé ; les contrastes de codes ne sont pas traduits en niveaux « faible/fort » sans validation RH.

## Réorganisation appliquée et traçabilité

Les déplacements exacts et empreintes avant/après sont dans `reports/tables/archive_manifest.csv`. L'inventaire exhaustif des fichiers analytiques avant réorganisation est dans `reports/tables/repository_inventory.csv` ; les notebooks ont leur propre inventaire de cellules/exécution. Aucune suppression définitive. Les réajustements sont limités à la vérification de la référence et au bootstrap de stabilité de la logistique ; aucune recherche de paramètres additionnelle.

Livrables de référence : `reports/EXECUTIVE_SUMMARY.md`, `MODEL_RESULTS.md`, `HR_RECOMMENDATIONS.md`, `SLIDE_STORYLINE.md` et `reports/figures/`. Méthodes et provenance par résultat : `reports/tables/source_register.csv`.

## Vérification finale

- 162 fichiers initiaux retrouvés, soit à leur chemin d'origine, soit via le manifeste ; 21 fichiers déplacés et aucun fichier perdu.
- 13 tests automatisés réussis, dont les contrastes one-hot et les bornes de Wilson ; notebooks valides et aucune sortie d'erreur enregistrée.
- Scores et IC de performance existants reproduits à l'identique ; différence de prédictions des modèles réajustés pour contrôle inférieure à 1e-12.
- Sources Excel, pipelines et métadonnées 04/05 inchangés ; leurs empreintes sont vérifiées dans `reports/consolidation_metadata.json`.
- 17 CSV de consolidation/provenance ; quatre figures 16:9 en PNG 4 800 × 2 700 pixels à 300 DPI, plus leurs SVG. Revue visuelle des 25 figures historiques et des quatre figures finales.
- Liens des rapports, chemins des sources et destinations d'archives vérifiés. Les notebooks archivés ont seulement reçu une adaptation de découverte de racine et une mention de leur statut ; leurs résultats exécutés sont conservés.
