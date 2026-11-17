# Ozone RH — Diagnostic des facteurs associés aux départs

**Question de décision : sur quels leviers RH agir en priorité, et pourquoi ?** Le projet décrit des associations et des contributions prédictives ; il ne mesure pas les effets causaux d'une politique RH. L'absentéisme n'est pas mesurable avec ce fichier.

## Commencer ici

1. [Synthèse pour décision](reports/EXECUTIVE_SUMMARY.md) : conclusions, populations et limites.
2. [Recommandations RH](reports/HR_RECOMMENDATIONS.md) : actions proposées, faisabilité et KPI.
3. [Storyline de 7 slides](reports/SLIDE_STORYLINE.md) et [figures finales](reports/figures/README.md).
4. [Résultats statistiques](reports/MODEL_RESULTS.md), [registre des sources](reports/tables/source_register.csv) et [audit complet](AUDIT.md).

La logistique et la forêt 05 sont les références. L'exploration 06 n'a pas démontré de gain robuste et reste en annexe. Les rapports générés constituent la vue consolidée ; les exports historiques restent les preuves de calcul.

## Structure et statut des fichiers

```text
AUDIT.md                         inventaire, décisions et problèmes méthodologiques
reports/                         restitution principale et métadonnées de consolidation
  tables/                        chiffres consolidés, provenance, manifestes
  figures/                       uniquement les 4 figures pour slides, PNG 300 DPI + SVG
  EXECUTIVE_SUMMARY.md
  MODEL_RESULTS.md
  HR_RECOMMENDATIONS.md
  SLIDE_STORYLINE.md
data/
  employee.csv.xlsx              source brute immuable, CSV encapsulé dans Excel
  variantes/                    données préparées : baseline 01 et simulations 02/03
    tracabilite/                 journal des simulations et correspondances de lignes
scripts/create_excel_variants.py préparation explicite, pas un prérequis à relancer
src/
  data.py, quality.py            lecture et contrôles
  features.py                   diagnostics et variables candidates exploratoires
  modeling.py                   prétraitements, split, CV et modèles 04/05
  group_feature_analysis.py      interprétabilité par variables et groupes
  reporting.py, report_text.py   consolidation et restitution reproductibles
  forest_improvement.py         reproduction de l'exploration archivée 06
notebooks/
  00_audit_variables.ipynb       audit détaillé des 35 variables
  01_exploration.ipynb           redondances et diagnostics, annexe analytique
  02_gender_equality.ipynb       écarts descriptifs et standardisation métier/niveau
  03_working_conditions.ipynb    proportions de départ et situations de travail
  04_attrition_model.ipynb       logistique L2 et dummy
  05_random_forest.ipynb         forêt et comparaison aux références
  archives/                     01b pédagogique et exploration 06 exécutée
outputs/
  models/                       pipelines 04/05, indices et métadonnées
  tables/, figures/             preuves historiques par notebook, hors sélection slides
  group_feature_analysis/       permutations, ablations et analyses par groupe
  archives/                     sorties et modèle candidat 06
docs/archives/                  guide PDF historique obsolète, conservé intégralement
tests/                          contrôles des partitions, encodages et méthodes
```

Les chemins des données et modèles de référence sont conservés pour protéger les contrats de reproduction et les empreintes. `data/employee.csv.xlsx` est la source brute, `data/variantes/` distingue explicitement les préparations ; aucune copie supplémentaire « raw/processed » n'est créée. Les déplacements sont détaillés dans `reports/tables/archive_manifest.csv` avec empreintes avant/après et adaptations de chemins.

## Installation

L'environnement de consolidation est Python 3.9.6, pandas 2.3.3, numpy 2.0.2 et scikit-learn 1.6.1. Pour reproduire les modèles joblib fournis, utiliser de préférence les versions vérifiées :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproducible.txt
```

`requirements.txt` donne les plages de dépendances pour un nouvel environnement ; un changement de version peut empêcher la lecture fidèle des modèles sauvegardés. Consulter leurs métadonnées. Charger uniquement les fichiers joblib de confiance du projet.

Pour Jupyter :

```bash
python -m ipykernel install --user --name ozone --display-name "Python (O-zone)"
python -m jupyterlab
```

Dans VS Code ou Jupyter, choisir `.venv/bin/python` ou le kernel **Python (O-zone)**.

## Reproduire la restitution sans refaire les recherches de modèles

Depuis la racine, avec les modèles 04/05 et sorties groupes déjà fournis :

```bash
python -m src.reporting
python -m unittest discover -s tests -v
```

La consolidation vérifie la source, les indices train/test, reproduit les prédictions des modèles et les IC de performance existants, consolide les groupes, puis effectue **300 bootstraps d'entraînement de la logistique L2** pour les intervalles de stabilité des OR. Ce sont les seuls réajustements interprétatifs ajoutés ; pas de nouvelle recherche de paramètres. Les données, modèles et sorties historiques sont en lecture seule. Les fichiers de `reports/` sont régénérés, sauf les inventaires et manifestes de migration qui documentent l'audit initial.

Les métadonnées de consolidation incluent les empreintes protégées et des scripts. Les intervalles de stabilité L2 ne sont pas des IC classiques de paramètres non pénalisés ; aucune p-value de significativité n'est produite.

## Reproduire toutes les analyses

1. Exécuter les notebooks 00 à 05 dans l'ordre, depuis la racine ou `notebooks/`.
2. Exécuter `python -m src.group_feature_analysis` pour les 30 permutations, les ablations et les modèles par groupe seul.
3. Exécuter `python -m src.reporting` pour la restitution.

Les notebooks 04/05 partagent le split stratifié 80/20 et cinq plis d'entraînement (`random_state=42`). Le 05 vérifie l'égalité exacte avec le 04. Les notebooks régénèrent uniquement leurs exports ; les recherches complètes et l'interprétabilité peuvent prendre plusieurs minutes. Leurs PNG historiques ne sont pas le dossier de présentation.

**Ne pas régénérer les variantes pour simplement reproduire les modèles.** Les fichiers fournis portent une empreinte vérifiée. Les variantes 02 (suppression de lignes) et 03 (plafonnement) sont des simulations, pas des corrections validées ; leur rôle et les journaux sont dans `data/variantes/README.md`.

Les notebooks archivés fonctionnent depuis la racine ou un sous-dossier du projet. Le 06 utilise `src/forest_improvement.py` et exporte désormais dans `outputs/archives/06_random_forest_improvement/`. Ses sorties exécutées historiques peuvent imprimer l'ancien emplacement, conservé comme trace ; le manifeste fait la correspondance. Ne pas le relancer pour sélectionner itérativement un modèle sur le test.

## Limites d'interprétation

1 518 salariés, 248 départs ; test historique de 304 salariés dont 50 départs. La période, le pays, les unités monétaires/de distance et le dictionnaire des scores ne sont pas suffisamment documentés. Pas de taux annuel ni d'extrapolation à l'Afrique. Une bonne ROC-AUC ne valide pas les probabilités individuelles ; les effets des actions RH restent à tester prospectivement. Les résultats actuels sont ceux des rapports sourcés, pas ceux du guide PDF archivé (ancien split 75/25).
