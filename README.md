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

1. `notebooks/01_exploration.ipynb` : structure, qualité et départs.
2. `notebooks/02_gender_equality.ipynb` : écarts bruts et comparaisons à métier/niveau similaires.
3. `notebooks/03_working_conditions.ipynb` : conditions de travail et départs.
4. `notebooks/04_attrition_model.ipynb` : modèle logistique et test stratifié réservé.
5. `notebooks/05_women_attrition.ipynb` : focus sur les départs des femmes (tests, régression logistique, interactions Genre × facteur, survie, segments, Oaxaca-Blinder, scénarios, robustesse, équité). Exports `05_*` dans `outputs/`.

Chaque notebook charge indépendamment les données. Les tableaux CSV (UTF-8 avec BOM) et figures PNG sont régénérés dans `outputs/tables/` et `outputs/figures/`. Les fichiers d’export portant le même nom sont remplacés lors d’une réexécution. Aucune donnée individuelle ni prédiction individuelle n’est exportée.

## Données et précautions

`data/employee.csv.xlsx` est un véritable classeur Excel dont la première colonne contient des lignes CSV : `src/data.py` détecte ce format et le sépare, sans modifier la source. `depart` est dérivé de `Attrition` (Yes=1, No=0). Les 35 colonnes source sont conservées.

La période, la devise, l’unité de distance et le dictionnaire des scores ne sont pas documentés. Les modalités atypiques sont conservées et exposées ; aucune correction arbitraire ni suppression de lignes n’est appliquée. Le notebook 01 exporte des contrôles d’ancienneté. L’absentéisme n’est pas mesurable. Les proportions de départ ne sont pas annualisées ; le délai depuis la dernière promotion n’est pas un taux de promotion. Les comparaisons et coefficients décrivent des associations, sans établir de causalité ni de discrimination. Aucun pays ne permet de valider la représentativité pour l’Afrique.

Le modèle utilise un pipeline ajusté uniquement sur l’entraînement, une régression pénalisée et une sélection explicite de variables sans identifiants ni constantes. Le test n’est pas utilisé pour régler les paramètres. Dans le notebook 04, les coefficients pénalisés ne sont pas accompagnés de tests de significativité. Le notebook 05 utilise statsmodels et scipy pour les odds ratios avec intervalles de confiance, les tests et la correction de Holm. Ses résultats sont des associations ; les scénarios « et si » supposent un effet causal et ne donnent que des ordres de grandeur.

## Vérification locale

Les quatre notebooks ont été exécutés dans l’ordre avec Python 3.9, pandas 2.3.3, numpy 2.0.2, openpyxl 3.1.5 et scikit-learn 1.6.1. Les exports sont fournis avec les sorties des cellules. Le lancement du kernel Jupyter a nécessité l’autorisation d’ouvrir des ports locaux dans l’environnement d’exécution.
