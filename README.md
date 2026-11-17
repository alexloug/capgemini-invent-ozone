# capgemini-invent-ozone
Analyse des facteurs de départ des salariés d’O-zone Telecom et recommandations RH.


## Installation et kernel

Python 3.9 ou supérieur (3.11 conseillé). Depuis la racine :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ozone --display-name "Python (O-zone)"
python -m jupyterlab
```

Dans Jupyter, choisir **Kernel → Change Kernel → Python (O-zone)**. Dans VS Code, sélectionner `.venv/bin/python` comme interpréteur du notebook. Les chemins fonctionnent depuis la racine ou `notebooks/`.

## Exécution

Exécuter toutes les cellules dans l’ordre, dans :

0. `notebooks/00_audit_variables.ipynb` : contrôle détaillé des 35 variables, valeurs atypiques et incohérences ; à lancer avant les analyses.
1. `notebooks/01_exploration.ipynb` : structure, qualité et départs.
2. `notebooks/02_gender_equality.ipynb` : écarts bruts et comparaisons à métier/niveau similaires.
3. `notebooks/03_working_conditions.ipynb` : conditions de travail et départs.
4. `notebooks/04_attrition_model.ipynb` : comparaison logistique L2 / Random Forest / baseline, split commun 80/20 et validation croisée stratifiée à cinq plis.

Chaque notebook charge indépendamment les données. Les tableaux CSV (UTF-8 avec BOM) et figures PNG sont régénérés dans `outputs/tables/` et `outputs/figures/`. Les fichiers d’export portant le même nom sont remplacés lors d’une réexécution. Aucune prédiction individuelle n’est exportée par les notebooks. Le notebook 04 sauvegarde des repères de ligne, les indices du split et les plis pour la reproductibilité.

## Données et précautions

`data/employee.csv.xlsx` est un véritable classeur Excel dont la première colonne contient des lignes CSV : `src/data.py` détecte ce format et le sépare, sans modifier la source. `depart` est dérivé de `Attrition` (Yes=1, No=0). Les 35 colonnes source sont conservées.

La période, la devise, l’unité de distance et le dictionnaire des scores ne sont pas documentés. Les modalités atypiques sont conservées et exposées ; aucune correction arbitraire ni suppression de lignes n’est appliquée. Le notebook 01 exporte des contrôles d’ancienneté. L’absentéisme n’est pas mesurable. Les proportions de départ ne sont pas annualisées ; le délai depuis la dernière promotion n’est pas un taux de promotion. Les comparaisons et coefficients décrivent des associations, sans établir de causalité ni de discrimination. Aucun pays ne permet de valider la représentativité pour l’Afrique.

Le modèle utilise un pipeline ajusté uniquement sur l’entraînement, une régression pénalisée et une répartition exhaustive des variables sans identifiants ni colonnes constantes retirées de la variante 02. Le test n’est pas utilisé pour régler les paramètres. Les coefficients ne sont pas accompagnés de tests de significativité ; statsmodels n’est donc pas nécessaire.

## Vérification locale

Les quatre notebooks ont été exécutés dans l’ordre avec Python 3.9, pandas 2.3.3, numpy 2.0.2, openpyxl 3.1.5 et scikit-learn 1.6.1. Les exports sont fournis avec les sorties des cellules. Le lancement du kernel Jupyter a nécessité l’autorisation d’ouvrir des ports locaux dans l’environnement d’exécution.

## Audit détaillé des variables

Le notebook `00_audit_variables.ipynb` lit les chaînes brutes via `src/quality.py` pour détecter aussi les espaces et conversions impossibles. Il inclut une fiche par colonne, les lignes Excel signalées, les contrôles d’ancienneté et un repérage IQR. Les bornes numériques affichées sont les minimums et maximums observés, et non des plages de validité. Aucune alerte ne repose sur une plage numérique supposée ; un dictionnaire reste nécessaire pour valider les codes. Aucun recodage ni retrait d’observation n’est appliqué. Les trois exports `00_audit_*.csv` sont agrégés ; les lignes individuelles sont consultables dans les sorties du notebook.

## Comparaison prédictive (notebook 04)

Le fichier commun est `data/variantes/02_sans_colonnes_valeurs_inchangees.xlsx`. Il doit être présent avant l’exécution ; ne pas régénérer les variantes pour exécuter les modèles. Toutes les lignes, tous les scores (y compris hors dictionnaire) et toutes les anciennetés sont conservés. Aucune imputation, suppression ou correction n’est réalisée. Une valeur manquante ou un code binaire inattendu provoque un diagnostic explicite. `Attrition`, `depart` et les quatre colonnes exclues ne sont jamais dans X ; le repère Excel est conservé séparément.

- Logistique : binaires 0/1, nominales et scores one-hot (`drop=None`, `handle_unknown="ignore"`), quantitatives standardisées uniquement, régularisation L2.
- Random Forest : binaires 0/1, nominales one-hot, scores et quantitatives numériques inchangés, aucune standardisation. L’ordre des scores hors dictionnaire est donc utilisé par hypothèse.
- Un unique split stratifié 80/20 et les mêmes cinq plis stratifiés mélangés (`random_state=42`) pour les deux modèles et `DummyClassifier(strategy="prior")`. Sélection sur l’average precision CV, seuil 0,5 fixé ; aucun réglage sur le test. Les probabilités pondérées ne sont pas présumées calibrées.

Après l’installation des dépendances et la sélection du kernel, ouvrir le notebook dans Jupyter et exécuter toutes ses cellules. Pour l’exécuter depuis la racine en ligne de commande :

```bash
source .venv/bin/activate
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=ozone notebooks/04_attrition_model.ipynb
```

La commande suppose le kernel `ozone` installé selon les instructions plus haut. Les fonctions réutilisables sont dans `src/modeling.py` ; le notebook orchestre leur utilisation, sans seconde implémentation.

Les sorties incluent les recherches et métriques CV, les résultats test et leur comparaison (`outputs/tables/04_*.csv`), les matrices de confusion et courbes précision-rappel, la liste des variables transformées. Les pipelines entraînés, indices ordonnés du split et plis, et métadonnées avec empreinte source sont dans `outputs/models/04_*`. Charger uniquement des fichiers joblib de confiance, avec les versions indiquées dans les métadonnées et `src/` disponible. Les exportations de cette expérience sont remplacées lors de la réexécution.

Le guide PDF existant décrit l’ancien modèle 75/25. Les anciens exports devenus inapplicables sont conservés dans `outputs/archive/04_ancien_modele/` et ne doivent pas être mélangés aux nouveaux résultats. Les autres notebooks et le fichier Excel original restent inchangés.

Contrôles automatisés du protocole :

```bash
python -m unittest discover -s tests -v
```

Le notebook 04 mis à jour a été exécuté de bout en bout : 1 214 lignes d’entraînement (198 départs) et 304 lignes de test (50 départs), aucune ligne retirée. Les trois tests vérifient les partitions, les encodages et les diagnostics. Les contrôles de fin de notebook vérifient les empreintes de tous les Excel et rechargent les modèles sauvegardés sur un échantillon d’entraînement.
