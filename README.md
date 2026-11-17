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
1. `notebooks/01_exploration.ipynb` : exploration de la baseline, redondances et huit variables candidates ; diagnostics sur l’entraînement uniquement.
   - `notebooks/01b_encoding.ipynb` : démonstration avant/après de l’encodage binaire, du one-hot, de la standardisation et des pipelines ; à lire avant les modèles.
2. `notebooks/02_gender_equality.ipynb` : écarts bruts et comparaisons à métier/niveau similaires.
3. `notebooks/03_working_conditions.ipynb` : conditions de travail et départs.
4. `notebooks/04_attrition_model.ipynb` : régression logistique L2 et baseline ; explications du modèle, de l’encodage, des coefficients et des métriques.
5. `notebooks/05_random_forest.ipynb` : Random Forest et baseline, puis comparaison des deux méthodes ; exécuter le 04 avant le 05.

Chaque notebook charge indépendamment les données. Les tableaux CSV (UTF-8 avec BOM) et figures PNG sont régénérés dans `outputs/tables/` et `outputs/figures/`. Les fichiers d’export portant le même nom sont remplacés lors d’une réexécution. Aucune prédiction individuelle n’est exportée par les notebooks. Les notebooks 04 et 05 sauvegardent des repères de ligne, les indices du split et les plis pour la reproductibilité.

## Données et précautions

`data/employee.csv.xlsx` est un véritable classeur Excel dont la première colonne contient des lignes CSV : `src/data.py` détecte ce format et le sépare, sans modifier la source. `depart` est dérivé de `Attrition` (Yes=1, No=0). Les 35 colonnes source sont conservées.

La période, la devise, l’unité de distance et le dictionnaire des scores ne sont pas documentés. Les modalités atypiques sont conservées et exposées ; aucune correction arbitraire ni suppression de lignes n’est appliquée. Le notebook 01 exporte des contrôles d’ancienneté. L’absentéisme n’est pas mesurable. Les proportions de départ ne sont pas annualisées ; le délai depuis la dernière promotion n’est pas un taux de promotion. Les comparaisons et coefficients décrivent des associations, sans établir de causalité ni de discrimination. Aucun pays ne permet de valider la représentativité pour l’Afrique.

Le modèle utilise un pipeline ajusté uniquement sur l’entraînement, une régression pénalisée et une répartition exhaustive des variables sans identifiants ni colonnes constantes retirées de la variante 01. Le test n’est pas utilisé pour régler les paramètres. Les coefficients ne sont pas accompagnés de tests de significativité ; statsmodels n’est donc pas nécessaire.

## Vérification locale

Les quatre notebooks ont été exécutés dans l’ordre avec Python 3.9, pandas 2.3.3, numpy 2.0.2, openpyxl 3.1.5 et scikit-learn 1.6.1. Les exports sont fournis avec les sorties des cellules. Le lancement du kernel Jupyter a nécessité l’autorisation d’ouvrir des ports locaux dans l’environnement d’exécution.

## Audit détaillé des variables

Le notebook `00_audit_variables.ipynb` lit les chaînes brutes via `src/quality.py` pour détecter aussi les espaces et conversions impossibles. Il inclut une fiche par colonne, les lignes Excel signalées, les contrôles d’ancienneté et un repérage IQR. Les bornes numériques affichées sont les minimums et maximums observés, et non des plages de validité. Aucune alerte ne repose sur une plage numérique supposée ; un dictionnaire reste nécessaire pour valider les codes. Aucun recodage ni retrait d’observation n’est appliqué. Les trois exports `00_audit_*.csv` sont agrégés ; les lignes individuelles sont consultables dans les sorties du notebook.

## Modèles prédictifs (notebooks 04 et 05)

Le fichier commun est `data/variantes/01_baseline.xlsx`. Il doit être présent avant l’exécution ; ne pas régénérer les variantes pour exécuter les modèles. Toutes les lignes, tous les scores (y compris hors dictionnaire) et toutes les anciennetés sont conservés. Aucune imputation, suppression ou correction n’est réalisée. Une valeur manquante ou un code binaire inattendu provoque un diagnostic explicite. `Attrition`, `depart` et les quatre colonnes exclues ne sont jamais dans X ; le repère Excel est conservé séparément.

- Logistique : binaires 0/1, nominales et scores one-hot (`drop=None`, `handle_unknown="ignore"`), quantitatives standardisées uniquement, régularisation L2.
- Random Forest : binaires 0/1, nominales one-hot, scores et quantitatives numériques inchangés, aucune standardisation. L’ordre des scores hors dictionnaire est donc utilisé par hypothèse.
- Un unique split stratifié 80/20 et les mêmes cinq plis stratifiés mélangés (`random_state=42`) pour les deux modèles et `DummyClassifier(strategy="prior")`. Sélection sur l’average precision CV, seuil 0,5 fixé ; aucun réglage sur le test. Les probabilités pondérées ne sont pas présumées calibrées.

Après l’installation des dépendances et la sélection du kernel, ouvrir les notebooks dans Jupyter et exécuter toutes leurs cellules, d’abord 04 puis 05. Pour l’exécuter depuis la racine en ligne de commande :

```bash
source .venv/bin/activate
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=ozone notebooks/04_attrition_model.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=ozone notebooks/05_random_forest.ipynb
```

La commande suppose le kernel `ozone` installé selon les instructions plus haut. Les fonctions réutilisables sont dans `src/modeling.py` ; les deux notebooks orchestrent leur utilisation, sans seconde implémentation.

Les sorties incluent les recherches et métriques CV, les résultats test et leur comparaison (`outputs/tables/04_regression_logistique/` et `outputs/tables/05_random_forest/`), les matrices de confusion et courbes précision-rappel, la liste des variables transformées. Les pipelines entraînés, indices ordonnés du split et plis, et métadonnées avec empreinte source sont dans `outputs/models/04_*` et `05_*`. Charger uniquement des fichiers joblib de confiance, avec les versions indiquées dans les métadonnées et `src/` disponible. Les exportations de cette expérience sont remplacées lors de la réexécution.

Le guide PDF existant décrit l’ancien modèle 75/25. Les résultats actuels sont ceux des notebooks 04 et 05 ; leur comparaison est dans `outputs/tables/05_random_forest/05_comparaison_modeles.csv`. Les dossiers d’archives ont été supprimés.

Contrôles automatisés du protocole :

```bash
python -m unittest discover -s tests -v
```

Les notebooks 04 et 05 utilisent : 1 214 lignes d’entraînement (198 départs) et 304 lignes de test (50 départs), aucune ligne retirée. Les quatre tests vérifient les partitions, les encodages, les diagnostics et le rejet d’une référence différente dans le notebook 05. Les contrôles de fin de notebook vérifient les empreintes de tous les Excel et rechargent les modèles sauvegardés sur un échantillon d’entraînement.


Le 04 entraîne uniquement la logistique et la baseline ; le 05 entraîne uniquement Random Forest et la baseline. Le split et les plis sont reconstruits de façon déterministe dans chaque notebook, puis le 05 vérifie l’égalité complète avec les indices sauvegardés du 04 et l’empreinte du fichier de données. Il s’interrompt si le 04 n’a pas été exécuté ou si la référence diffère. Chaque modèle est sélectionné sur l’AP de validation croisée et évalué sur le test après la sélection, au seuil fixe de 0,5.

La régression additionne des contributions sur le logarithme des cotes puis calcule une probabilité ; la forêt moyenne les probabilités de 200 arbres entraînés séparément sur des échantillons bootstrap. Elle apprend des seuils et interactions sans standardiser les variables. Random Forest n’est pas du boosting : le boosting construit ses arbres successivement. Ces distinctions et les limites de lecture des coefficients avec toutes les modalités one-hot sont détaillées dans les notebooks.


## Où trouver les CSV ?

Les résultats sont rangés par notebook dans `outputs/tables/00_audit/`, `01_exploration/`, `02_egalite_femmes_hommes/`, `03_conditions_travail/`, `04_regression_logistique/` et `05_random_forest/`. Le guide `outputs/tables/README.md` indique les fichiers à lire en priorité. Dans les deux dossiers de modèles, `details/` contient les recherches de paramètres, listes de variables et répartitions des partitions.

Les indices nécessaires à la reproduction restent avec les modèles dans `outputs/models/`. Les journaux des variantes sont dans `data/variantes/tracabilite/`. Les anciens résultats archivés et le premier export Excel redondant ont été supprimés ; les résultats actuels sont conservés. `src.data.table_path()` et `export_table()` appliquent automatiquement cette organisation aux prochaines lectures et exécutions.


## Exploration de variables (notebook 01)

Le notebook 01 utilise `data/variantes/01_baseline.xlsx`. Il examine les corrélations Pearson/Spearman, le lien salaire/niveau/expérience et le recouvrement métier/département. Il explore huit colonnes : expérience hors entreprise, part de carrière dans l’entreprise, temps avant le rôle actuel, part d’ancienneté dans le rôle, temps avant le manager actuel, cumul heures supplémentaires/déplacements fréquents, interaction distance/heures supplémentaires et revenu relatif à la médiane métier/niveau.

Les fonctions sont dans `src/features.py`. Le notebook ne modifie aucun Excel ni modèle, et ne sélectionne pas les variables en fonction du test. Seules les colonnes dérivées non calculables reçoivent NaN. La référence salariale est calculée sur l’entraînement ; son intégration à un modèle nécessiterait un transformer ajusté dans chaque pli. Les diagnostics et exports sont dans `outputs/tables/01_exploration/`. Le contrôle détaillé de chaque variable reste dans le 00. Les anciens exports du 01 sur la cible et les descriptifs généraux ont été retirés pour éviter leur confusion avec cette nouvelle analyse.


Le notebook pédagogique `01b_encoding.ipynb` réutilise les prétraitements de `src/modeling.py` sur `01_baseline.xlsx`, avec le même split que les modèles. Les exemples restent dans le notebook : aucun Excel/CSV encodé ni pipeline sauvegardé n’est produit ou modifié. Il montre aussi une catégorie inconnue fictive et l’apprentissage d’un prétraitement dans un seul pli de validation croisée.
