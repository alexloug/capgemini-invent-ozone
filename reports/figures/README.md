# Figures finales pour la présentation

| Fichier PNG/SVG | Message | Table source |
|---|---|---|
| `01_performance` | Signal prédictif et incertitude | `../tables/model_performance.csv` |
| `02_odds_ratios` | Associations ajustées et stabilité L2 | `../tables/logistic_or_stability.csv` |
| `03_groupes_rh` | Contributions prédictives des groupes | `../tables/group_permutation_importance.csv` |
| `04_sous_groupes` | Situations de travail à diagnostiquer | `../tables/subgroup_support.csv` |

Format 16:9, PNG 300 DPI et SVG. Les barres des OR sont des intervalles de stabilité de l’estimateur L2 ; les barres des taux sont des IC Wilson ; les autres IC sont conditionnels aux modèles fixés. Aucune figure historique ou exploratoire secondaire n’est copiée ici. Les détails sont dans `../MODEL_RESULTS.md`.
