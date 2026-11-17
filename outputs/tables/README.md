# Guide des CSV actuels

Commencer par les fichiers ci-dessous. Un CSV est un résultat, pas un nouveau dataset à utiliser à la place de l’Excel. Les exports de la même expérience sont remplacés à la réexécution.

| Dossier | À consulter en priorité | Pourquoi le conserver |
| --- | --- | --- |
| `00_audit/` | `00_audit_synthese.csv` | Résume les incohérences signalées. |
| `01_exploration/` | `01_correlations_quantitatives.csv`, `01_recouvrement_metier_departement.csv`, `01_recouvrement_candidates.csv` | Redondances et intérêt descriptif des variables candidates sur l’entraînement. |
| `02_egalite_femmes_hommes/` | `02_depart_genre.csv`, `02_ecarts_comparables.csv`, `02_standardisation.csv` | Départs par genre, revenus à métier/niveau similaires et comparaison standardisée. |
| `03_conditions_travail/` | `03_OverTime.csv`, `03_BusinessTravel.csv`, `03_distance.csv`, `03_WorkLifeBalance.csv` | Départs selon les conditions de travail avec effectifs. |
| `04_regression_logistique/` | `04_evaluation.csv`, `04_validation_croisee.csv` | Performance test et sélection sur validation croisée, à distinguer. |
| `05_random_forest/` | `05_comparaison_modeles.csv` | Comparaison finale des deux modèles et de la baseline ; lire la colonne `evaluation`. |

## Compléments utiles à garder

- Audit : inventaire des colonnes et repérage des valeurs atypiques IQR. Une valeur atypique n’est pas nécessairement une erreur.
- Exploration : qualité de la baseline, contrôles d’ancienneté, corrélations, références salariales et résumés des huit variables candidates. Le notebook 00 conserve les contrôles détaillés par variable.
- Égalité : effectifs et parts par niveau/métier, indicateurs de parcours et strates. Ils expliquent la construction des résultats synthétiques.
- Conditions de travail : satisfactions, résumé des distances et croisement heures supplémentaires/genre/niveau.
- Modèles : `*_comparaison_performances.csv` réunit CV et test pour le modèle du notebook ; `*_evaluation.csv` et `*_validation_croisee.csv` les présentent séparément. Le recouvrement est volontaire pour faciliter leur lecture.
- `04_coefficients.csv` : coefficients pénalisés ; avec toutes les modalités one-hot conservées, ne pas les interpréter comme des écarts à une catégorie de référence supprimée.

Dans les dossiers 04 et 05, les sous-dossiers `details/` regroupent les recherches de paramètres, les variables avant/après encodage et les répartitions du split et des plis. Ils sont utiles pour vérifier la méthode, mais ne sont pas nécessaires à une première lecture.

## Les CSV conservés ailleurs

- `outputs/models/*_indices_split_plis.csv` : indices ordonnés indispensables à la reproduction.
- `data/variantes/tracabilite/resume_variantes.csv` : dimensions des trois variantes Excel.
- `data/variantes/tracabilite/journal_lignes_supprimees.csv` et `journal_valeurs_plafonnees.csv` : traitements simulés.
- `data/variantes/tracabilite/correspondance_lignes.csv` : liens avec la source.
- `outputs/group_feature_analysis/` : importances groupées, ablations, IC de performance et standardisation sur entraînement.
- `outputs/archives/06_random_forest_improvement/` : exploration secondaire, pas de nouveau modèle de référence.

La restitution principale se trouve dans `reports/`. Les exports historiques sont conservés comme preuves et dépendances des notebooks ; ils ne sont pas dupliqués dans le dossier des slides. Voir `AUDIT.md` pour les décisions et les manifestes.
