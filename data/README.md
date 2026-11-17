# Quels fichiers utiliser ?

| Fichier | Rôle | Conserver ? |
| --- | --- | --- |
| `employee.csv.xlsx` | Source originale, utilisée par les notebooks 00, 02 et 03. | Oui, ne jamais modifier. |
| `variantes/01_baseline.xlsx` | Base du notebook 01 et référence des modèles 04 et 05 : 1 518 salariés, 31 colonnes, valeurs inchangées. | Oui, fichier de travail principal. |
| `variantes/02_rows_removed.xlsx` | Variante avec 20 lignes retirées : 1 498 × 31. | Oui, pour une analyse de sensibilité seulement. |
| `variantes/03_tenure_capped.xlsx` | Variante de simulation : 26 cellules plafonnées, 1 518 × 31. | Oui, pour une analyse de sensibilité seulement. |

## Les quatre CSV de traçabilité

Ils sont regroupés dans `variantes/tracabilite/` et doivent rester avec les variantes :

- `resume_variantes.csv` : dimensions des trois variantes ; commencer par celui-ci.
- `journal_lignes_supprimees.csv` : quelles lignes ont été retirées de la variante 02, et pourquoi.
- `journal_valeurs_plafonnees.csv` : anciennes et nouvelles valeurs de la variante 03.
- `correspondance_lignes.csv` : correspondance des numéros de ligne entre source et variantes.

Ces CSV ne sont pas des données d’entrée du modèle. Ne pas confondre les versions 03/04 avec des corrections vérifiées : la première supprime des salariés, la seconde suppose des valeurs.

`scripts/create_excel_variants.py` régénère les trois variantes et leurs journaux aux emplacements ci-dessus. Aucune donnée source ni valeur Excel n’a été modifiée lors du rangement.

Le premier export redondant et le dossier d’archives ont été supprimés.
