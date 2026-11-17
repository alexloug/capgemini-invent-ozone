# Variantes du fichier salariés

La source `data/employee.csv.xlsx` reste intacte. Chaque classeur contient une variable par colonne et aucune variable dérivée `depart`.

- `01_baseline.xlsx` : retrait de EmployeeCount, Over18, StandardHours, EmployeeNumber, aucune ligne ni valeur modifiée.
- `02_rows_removed.xlsx` : retrait des mêmes colonnes et des 20 lignes distinctes concernées par les 23 incohérences initiales.
- `03_tenure_capped.xlsx` : toutes les lignes, sans les quatre colonnes, avec plafonnement séquentiel autorisé : YearsAtCompany ≤ TotalWorkingYears, puis YearsInCurrentRole, YearsSinceLastPromotion et YearsWithCurrManager ≤ YearsAtCompany plafonné.

Le plafonnement modifie 26 cellules sur 20 lignes distinctes. Il peut modifier davantage de cellules que les 23 alertes initiales, car abaisser YearsAtCompany peut imposer de nouveaux plafonnements sur ses sous-durées. Il s'agit d'une simulation de cohérence, pas d'une correction dont on connaît la valeur réelle. L'expérience totale est conservée comme plafond par convention ; elle pourrait elle-même être erronée.

Les journaux CSV, dans `tracabilite/`, indiquent les numéros de ligne et identifiants de la source, les motifs de retrait et chaque ancienne/nouvelle valeur. `correspondance_lignes.csv` permet de retrouver les lignes après suppression. Les exports sont régénérés sous les mêmes noms à chaque exécution de `scripts/create_excel_variants.py`.

Pour l'étude, garder la version 01 comme référence et utiliser 02/03 pour une analyse de sensibilité. La suppression peut biaiser les résultats si les lignes retirées ont des profils particuliers ; le plafonnement introduit une hypothèse. Les notebooks 00, 02 et 03 chargent la source originale ; le notebook 01 et les modèles 04 et 05 chargent la variante 01. Les variantes 02 et 03 ne sont pas utilisées automatiquement.
