"""Textes de restitution générés uniquement à partir des tableaux vérifiés."""
import pandas as pd
from .data import ROOT
from .group_feature_analysis import markdown_table

REPORTS = ROOT/'reports'


def table(frame):
    data = frame.copy()
    for column in ['n', 'departures', 'n_test', 'departures_test', 'bootstrap_n', 'positive_auc_folds']:
        if column in data:
            data[column] = data[column].map(lambda x: str(int(x)))
    return markdown_table(data)


def write_reports(performance, paired, odds, adjusted, groups, ablation, subgroups, descriptive):
    def write(name, sections):
        (REPORTS/name).write_text('\n\n'.join(sections)+'\n', encoding='utf-8')
    overtime = descriptive[descriptive.variable.eq('OverTime')].set_index('category')
    yes, no = overtime.loc['Yes'], overtime.loc['No']
    top = subgroups[subgroups.part.eq('ensemble') & subgroups.level_1 & subgroups.overtime].iloc[0]
    or_ot = odds.iloc[0]
    gender = pd.read_csv(REPORTS/'tables/gender_standardization.csv')
    income = gender[gender.minimum_per_gender_per_stratum.eq(5) & gender.outcome.eq('income')].iloc[0]
    perf = performance.copy()
    perf['model'] = ['Logistique L2', 'Random Forest 05', 'Dummy prior']
    pcols = ['model', 'roc_auc', 'roc_auc_low', 'roc_auc_high', 'average_precision', 'recall_at_05', 'precision_at_05']
    group_summary = groups[['model','group','importance_mean','bootstrap_ci_low','bootstrap_ci_high']].copy()
    full_sub = subgroups[subgroups.part.eq('ensemble')]
    limitations = ('Données observationnelles, sélection historique des modèles et hypothèses déjà exposées aux données, '
        'test réutilisé et seulement 50 départs en test. Chronologie des variables inconnue : fuite temporelle possible malgré '
        'un split et des pipelines corrects. Ni validation externe ni représentativité géographique établies. '
        'Scores non documentés, 23 incohérences d’ancienneté sur 20 salariés et corrélations entre groupes. '
        'Les importances et scores ne sont pas des effets causaux ; une AUC élevée ne garantit pas une probabilité individuelle fiable. '
        'Aucune mesure d’absence ; pas de résultat sur l’absentéisme et pas de taux annuel de turnover. '
        'Source : S01, S10, S12 ; voir `AUDIT.md` et le registre de sources.')
    source_note = ('**Traçabilité** : les identifiants S01–S14 renvoient au [registre exact des sources](tables/source_register.csv), '
                   'qui indique script/fonction, notebook, sortie, méthode et population. '
                   'Les chiffres des figures proviennent exclusivement de `reports/tables/`.')
    executive = ['# Ozone — Synthèse pour décision',
        '## 1. Executive summary',
        '1. **Prioriser un diagnostic des conditions de travail**, tout en maintenant un chantier d’égalité professionnelle : les deux objectifs ne se mesurent pas avec les mêmes indicateurs. Sources S02, S04, S07.',
        f'2. **Les heures supplémentaires constituent le signal le plus convergent** : {yes.rate:.1%} de départs ({int(yes.departures)}/{int(yes.n)}) '
        f'contre {no.rate:.1%} ({int(no.departures)}/{int(no.n)}), OR ajusté pénalisé {or_ot.odds_ratio:.2f}. Ce n’est pas un effet causal. Sources S03, S06.',
        f'3. **Cibler le diagnostic du cumul heures supplémentaires–niveau codé 1** : {int(top.departures)}/{int(top.n)} départs, '
        f'soit {top.rate:.1%}. Le code de niveau doit être validé avant de le traduire en population métier. Source S08.',
        '4. **Croiser les méthodes plutôt que chercher un modèle plus complexe** : logistique et forêt classent conditions de travail puis parcours professionnel en tête par permutation. Source S04.',
        f'5. **Poursuivre l’audit d’égalité** : sur {int(income.covered_n)} salariés d’entraînement comparables, '
        f'l’écart standardisé de revenu femmes/hommes est de {income.female_vs_male_percent:.2f} %. '
        'Ni ce petit écart ni la faible importance du genre ne suffisent à conclure sur l’égalité. Source S07.',
        '6. **Ne pas déployer un score individuel en l’état** : les AUC sont informatives mais la calibration et la validation temporelle manquent. La logistique détecte 32 % des départs au seuil 0,5, la forêt 24 %. Source S02.',
        '7. **Évaluer les actions par des pilotes**, avec indicateurs et groupes comparables définis avant le lancement ; le rendement d’une intervention RH n’est pas estimé ici. Recommandation, pas résultat mesuré.',
        '## 2. Dataset et méthodologie',
        '1 518 salariés, 248 départs observés (16,34 %), cible Attrition Yes=1. 35 colonnes source ; 30 prédicteurs après exclusion '
        'de la cible, de l’identifiant et de trois constantes. Source originale intacte, baseline 01 sans changement de valeurs. '
        'Split stratifié 80/20 : 1 214/198 en entraînement et 304/50 en test. Prétraitement dans les pipelines, cinq plis identiques. '
        'La consolidation ne sélectionne aucun nouveau modèle. Sources S01–S02.',
        '## 3. Performance des modèles', table(perf[pcols]),
        'IC de ROC-AUC : 500 bootstraps stratifiés du test, conditionnels aux modèles fixés. AP = average precision. '
        'Les intervalles ne couvrent pas toute la sélection ni le réentraînement. '
        'La différence des modèles est étudiée par bootstrap **apparié**, pas par chevauchement des IC. Source S11.',
        '## 4. Facteurs associés au turnover',
        f'Heures supplémentaires : OR ajusté L2 {or_ot.odds_ratio:.2f}, intervalle de stabilité '
        f'[{or_ot.stability_low:.2f} ; {or_ot.stability_high:.2f}]. '
        'Déplacements, satisfaction et niveau sont comparés selon des modalités explicites dans le [rapport statistique](MODEL_RESULTS.md). '
        'Les codes de satisfaction ne sont pas traduits en « faible/fort » sans dictionnaire. Sources S03 et S06.',
        '## 5. Contribution des leviers RH', table(group_summary),
        'Baisse de ROC-AUC par permutation conjointe ; barres non additives. Le groupe « Gender Equality » contient seulement Gender. '
        'Il ne mesure pas les inégalités, tandis que les groupes de conditions et de carrière regroupent plusieurs variables. '
        'Les ablations avec réentraînement confirment la contribution des conditions de travail, sans mesurer un effet de politique RH. Sources S04–S05.',
        '## 6. Sous-groupes prioritaires', table(full_sub[['label','n','departures','rate','wilson_low','wilson_high','training_support_ok','test_support_ok']]),
        'Sous-groupes guidés par les analyses existantes. Support d’entraînement requis : au moins 80 salariés, 20 départs et 20 non-départs. '
        'Les tableaux détaillent séparément entraînement et test ; leur cohérence descriptive n’est pas une réplication indépendante. '
        'Pas de segmentation fine par sexe × niveau × heures sur les slides ; les petites cellules sont signalées dans l’audit. Source S08.',
        '## 7. Recommandations Ozone',
        'Pilote charge/horaires et déplacements ; revue des parcours et entretiens de maintien en poste ; audit documenté des rémunérations '
        'et décisions de promotion. Populations, faisabilité, KPI et limites dans [HR_RECOMMENDATIONS.md](HR_RECOMMENDATIONS.md).',
        '## 8. Limites', limitations, source_note]
    write('EXECUTIVE_SUMMARY.md', executive)
    cv = pd.read_csv(ROOT/'outputs/tables/05_random_forest/05_comparaison_modeles.csv')
    cv = cv[cv.evaluation.eq('CV_entrainement')][['modele','average_precision','average_precision_ecart_type','roc_auc','roc_auc_ecart_type']]
    score = adjusted.iloc[0]
    pair = paired.iloc[0]
    numerical = pd.read_csv(ROOT/'outputs/group_feature_analysis/feature_correlations_training.csv')
    cross = numerical[numerical.group_1.ne(numerical.group_2)].head(8)
    model_sections = ['# Résultats statistiques et contrôle de provenance', source_note,
        '## Données, encodage et absence de fuite de partition',
        'Source : `data/employee.csv.xlsx`, séparée par `src/data.py::load_data` ; modèle sur `data/variantes/01_baseline.xlsx`. '
        'Égalité cellule par cellule vérifiée sur les colonnes communes. Cible Attrition Yes=1 confirmée. '
        '`src/modeling.py::make_split` reproduit exactement les indices sauvegardés 04/05 ; train et test disjoints, plis sur train seulement. '
        'Le scaler logistique, le one-hot et les modèles sont ajustés sur train. Le notebook 04 compare 6 configurations, le 05 en compare 8 '
        'avec class_weight=None/balanced ; sélection sur AP. Paramètres retenus non pondérés. '
        'La logistique standardise les quantitatives et encode scores/nominales ; la forêt traite les scores comme ordonnés. '
        'Le pipeline exclut Attrition/depart, identifiant et constantes. Les données des variantes 02/03 ne remplacent pas la baseline. Sources S01, S10.',
        'Les pipelines sauvegardés ont été reproduits par clonage et ajustement sur les seuls indices train ; '
        'écart maximal de prédictions avec les fichiers existants inférieur à 1e-12. '
        'Cette vérification et le code étayent la provenance de l’ajustement, sans prouver une antériorité temporelle des variables. '
        'Voir `tables/model_replay_checks.csv`, méthode `src/reporting.py::performance_checks`.',
        '## Performance sur test', table(perf),
        'La Brier score est la moyenne des erreurs quadratiques des probabilités (plus bas = mieux), pas une certification de calibration. '
        'Aucune courbe de calibration externe suffisamment documentée n’est disponible. '
        'Précision/rappel au seuil fixé de 0,5 ; AP sans interpolation trapézoïdale. Sources S02.',
        '### Incertitude et comparaison appariée', table(paired),
        f'Différence AUC logistique − forêt : {pair.auc_difference:.4f}, intervalle apparié '
        f'[{pair.paired_bootstrap_low:.4f} ; {pair.paired_bootstrap_high:.4f}]. '
        'Le calcul réutilise les mêmes salariés bootstrap pour les deux modèles. '
        'Ce résultat conditionnel sur un test déjà consulté n’établit pas une supériorité générale. '
        'Aucun test n’est déduit du chevauchement des IC marginaux. Source S11.',
        '### Validation croisée historique', table(cv),
        'Écarts-types entre cinq plis, pas des IC. Scores de sélection non imbriqués, potentiellement optimistes ; '
        'ne pas les agréger avec les scores test comme observations indépendantes. Source S02.',
        '## Associations ajustées : contrastes et stabilité', table(odds),
        'Source S03 : `src/reporting.py::selected_log_odds` et `bootstrap_associations`. '
        'Même logistique L2 et C=0,1 ; 300 bootstraps stratifiés de l’entraînement, encodeur et scaler réajustés à chaque tirage. '
        'Pour chaque contraste catégoriel, OR=exp(β_modalité−β_référence) malgré le one-hot complet. '
        'OverTime et Gender sont binaires (Yes/Female=1). Pas de comparaison brute des amplitudes entre unités différentes. '
        'Les références du graphique sont des modalités courantes explicites, pas automatiquement le code 0. '
        'Les intervalles percentiles décrivent la stabilité de l’estimateur pénalisé conditionnel à sa sélection : '
        'ils ne garantissent pas une couverture nominale de paramètres non pénalisés, n’intègrent pas tout le biais de régularisation '
        'et ne constituent pas des tests post-sélection. Aucun p-value ajouté. Les OR ne sont ni des risques relatifs ni des effets causaux.',
        '### Écart moyen de scores standardisés pour OverTime', table(adjusted),
        f'Sur la population train fixe, différence moyenne de scores prédits Yes−No : {100*score.mean_model_score_difference:.2f} points, '
        f'intervalle de stabilité [{100*score.stability_low:.2f} ; {100*score.stability_high:.2f}] points. '
        'Les deux prédictions conservent les autres variables observées. Il s’agit d’un contraste de modèle '
        'avec probabilités non présumées calibrées ; certaines combinaisons peuvent être peu représentées. '
        'Ne pas l’annoncer comme une réduction attendue de turnover si les heures sont supprimées. Source S09.',
        '## Contribution prédictive des groupes', table(group_summary),
        'Permutation : mêmes lignes permutées pour toutes les colonnes originales du groupe, relations internes conservées, '
        'pipeline figé, 30 répétitions. IC bootstrap conditionnels à 95 % sur 500 tirages du test ; variabilité de permutation '
        'également disponible dans les sorties sources. Les importances ne s’additionnent pas et ne sont pas des pourcentages d’explication. '
        'La sensibilité au transfert de DistanceFromHome vers les conditions de travail est disponible dans '
        '`outputs/group_feature_analysis/assignment_sensitivity.csv`. Source S04.',
        '### Ablation avec réentraînement sur les mêmes plis', table(ablation),
        'Delta positif = perte d’AUC/AP après retrait, modèles réajustés à paramètres fixes. '
        'Un groupe seul mesure une information autonome, pas sa contribution unique ; résultats conservés dans '
        '`outputs/group_feature_analysis/group_only_model_results.csv`. '
        'Les conditions de travail restent les plus contributives en ablation moyenne ; les autres petits écarts ne sont pas des priorités fermes. Source S05.',
        '### Corrélations entre groupes', table(cross[['variable_1','variable_2','spearman','pearson','group_1','group_2']]),
        'Corrélations sur entraînement uniquement. Métier/département et niveau/revenu se recouvrent ; '
        'un groupe corrélé peut se substituer à un autre. Permutation marginale et retrait réentraîné ne mesurent pas la même chose. '
        'Les tailles des groupes diffèrent et leur affectation est une convention métier. Source S12.',
        '## Descriptifs et sous-groupes', table(descriptive), table(subgroups),
        'Proportion = départs/effectif ; IC de Wilson à 95 % sans correction de comparaisons multiples. '
        'Les sous-groupes OverTime × niveau codé 1 sont une hypothèse guidée par les analyses existantes, pas une découverte indépendante. '
        'Le regroupement « autres niveaux » comprend aussi le code 0 : le dictionnaire doit être validé. '
        'Règle de support fixée sur train : n≥80, départs≥20, non-départs≥20. '
        'Le test est détaillé avec un indicateur de support n≥20, départs≥5, non-départs≥10, sans sélection des taux les plus élevés. '
        'Un effectif suffisant ne garantit ni la représentativité ni l’absence de confusion. '
        'Les petites strates historiques sont conservées en annexe dans `tables/historical_subgroup_review.csv`. '
        'Aucune interaction du Random Forest n’est prouvée par ses seules importances ; les écarts entre strates ne sont pas des effets de traitement. Sources S06, S08.',
        '## Égalité femmes-hommes : périmètre et couverture', table(gender),
        'Périmètre principal : entraînement, standardisation sur les mêmes poids JobRole × JobLevel, strates contenant au moins 5 salariés '
        f'de chaque genre : {int(income.covered_n)}/1 214 personnes. Le résultat brut et la version sur toute la base sont dans le notebook 02 '
        'et `outputs/tables/02_egalite_femmes_hommes/02_standardisation.csv`. '
        'Les écarts de population, de poids et de filtre expliquent les différences numériques entre ces sorties ; '
        'les seuils 1/5/10 montrent la sensibilité à la couverture. '
        'Aucun ajustement complet sur temps de travail, expérience, responsabilités ou pays. Aucun IC d’écart salarial ni preuve de discrimination. '
        'YearsSinceLastPromotion est un délai, pas une probabilité de promotion. Source S07.',
        '## Pourquoi ne pas multiplier les forêts',
        'Le notebook 06 archivé a exploré 20 configurations, encodages, retraits et seuils. '
        'AP externe agrégée : 0,541 pour la référence contre 0,523 pour la procédure optimisée ; différence conditionnelle '
        '[-0,053 ; 0,014]. AP test du candidat : 0,532 contre 0,507, sans gain robuste confirmé. '
        'L’abaissement du seuil de la référence (appris sur train) donne 43/50 départs détectés mais 85 faux positifs ; '
        'ce compromis dépend d’un coût métier non validé. Le modèle 06 ne remplace pas le 05. Source S13.',
        'SHAP absent de l’environnement : aucune figure ni conclusion SHAP. Pas d’autre algorithme testé identifié dans les sources du dépôt. '
        'Les résultats non linéaires de la forêt servent seulement de triangulation avec la logistique. Source S14.',
        '## Limites et reproductibilité', limitations,
        '`python -m src.reporting` régénère tableaux, rapports et quatre figures PNG/SVG. '
        'Les modèles historiques restent en lecture seule. `reports/consolidation_metadata.json` contient leurs empreintes '
        'et celles des données ; 300 réajustements L2 sont justifiés uniquement par les intervalles de stabilité manquants. '
        'Les recherches/permutations déjà réalisées ne sont pas rejouées.']
    write('MODEL_RESULTS.md', model_sections)
    recommendations = ['# Recommandations RH pour Ozone',
        'Les chiffres justifient un ordre de diagnostic, pas un classement de rentabilité des politiques. '
        'Les horizons ci-dessous sont des propositions de mise en œuvre, pas des résultats calculés.',
        '## 1. Lancer un pilote de charge, horaires et récupération',
        f'**Constat** : {int(yes.departures)}/{int(yes.n)} départs avec heures supplémentaires ({yes.rate:.1%}) contre '
        f'{int(no.departures)}/{int(no.n)} sans ({no.rate:.1%}), et OR ajusté pénalisé {or_ot.odds_ratio:.2f}. '
        f'Le cumul avec le niveau codé 1 concerne {int(top.n)} personnes et {int(top.departures)} départs. Sources S03, S06, S08.',
        '**Action proposée** : sous 0–3 mois, RH et responsables opérationnels valident les codes et les équipes concernées, '
        'mesurent les volumes d’heures et les contraintes de déplacement, puis testent des règles de planning/récupération. '
        '**Population** : équipes comparables exposées aux heures supplémentaires, priorité de diagnostic au niveau codé 1 sans l’assimiler automatiquement à des juniors. '
        '**Faisabilité** : diagnostic possible avec les managers ; volumes, dates et unités doivent être collectés. '
        '**KPI proposés** : heures réelles et jours de déplacement par personne, récupération effective, satisfaction répétée, '
        'départs volontaires avec dénominateur et période fixes, continuité de service et charge transférée aux collègues. '
        '**Évaluation** : si possible déploiement progressif randomisé par équipe, sinon groupes comparables et tendances antérieures ; '
        'fixer les indicateurs avant le pilote. Aucune baisse chiffrée des départs n’est promise.',
        '## 2. Clarifier les parcours et les moments de fragilité',
        '**Constat** : le parcours est le deuxième groupe par permutation dans les deux modèles, derrière les conditions de travail. '
        'Il regroupe des variables corrélées d’ancienneté, de niveau, métier et formation. Sources S04–S05.',
        '**Action proposée** : sous 0–3 mois, rapprocher entretiens de départ, mobilités, formation et retours managers ; '
        'formaliser des entretiens de maintien en poste et des critères de mobilité. '
        '**Population** : salariés des équipes pilotes et niveaux à valider avec RH ; pas de ciblage automatique par score individuel. '
        '**Faisabilité** : utiliser les processus existants, avec validation des incohérences d’ancienneté. '
        '**KPI proposés** : entretiens réalisés, candidatures et mobilités datées, accès à la formation, compréhension des critères, '
        'départs sur période définie. Ne pas assimiler YearsSinceLastPromotion à un taux de promotion. '
        '**Hypothèse** : une meilleure lisibilité des parcours peut améliorer la rétention ; son effet doit être testé.',
        '## 3. Maintenir un chantier d’égalité distinct du score de turnover',
        f'**Constat** : écart de revenu standardisé femmes/hommes de {income.female_vs_male_percent:.2f} % sur '
        f'{int(income.covered_n)} salariés train comparables à métier/niveau, sans ajustement complet ni test de discrimination. '
        'La faible contribution de Gender au classement des départs ne mesure pas l’égalité. Source S07.',
        '**Action proposée** : sous 0–3 mois, auditer grilles, temps de travail et responsabilités ; documenter candidatures, '
        'décisions de promotion et augmentations avec dates et populations éligibles. '
        '**Population** : tous les salariés comparables, avec attention à la représentation par métier/niveau ; '
        'ne pas tirer de conclusion d’une petite strate. '
        '**Faisabilité** : nécessite des données RH complémentaires et une revue avec les responsables de rémunération. '
        '**KPI proposés** : écarts comparables de rémunération avec effectifs/couverture, candidatures et promotions parmi les éligibles, '
        'accès aux formations et postes de responsabilité. '
        '**Hypothèse** : les processus peuvent nécessiter des ajustements ; les données actuelles ne démontrent ni égalité complète ni discrimination.',
        '## Socle commun avant déploiement',
        'Valider dictionnaire, chronologie, pays/établissements et caractère volontaire des départs ; '
        'collecter des épisodes d’absence si l’absentéisme doit être étudié. '
        'Une AUC de 0,87 ne constitue pas une validation d’usage individuel. '
        'La capacité d’accompagnement et les coûts des faux positifs doivent être définis avant tout seuil de décision. '
        'Pas de cible de ROI ni de réduction causale inventée.', source_note]
    write('HR_RECOMMENDATIONS.md', recommendations)
    slides = [
        ('1. Prioriser les conditions de travail, sans déprioriser l’égalité',
         'Réponse business : lancer un diagnostic charge/organisation, maintenir un audit d’égalité distinct.',
         'Permutation conditions : ΔAUC 0,127 logistique / 0,110 forêt ; parcours 0,076 / 0,046.',
         'figures/03_groupes_rh.png (ou SVG)', 'S04–S05, tables/group_permutation_importance.csv',
         'Résultats prédictifs + recommandation ; effets des politiques non estimés.', 'Quel est le périmètre du problème mesuré ?'),
        ('2. La base décrit 248 départs, mais ne mesure pas l’absentéisme',
         'Cadrer la population et les limites dès le début.',
         '1 518 salariés ; 248 départs (16,34 %) ; test 304/50 ; aucune période documentée.',
         'Tableau 3 lignes : effectif / départs / données manquantes au sens métier', 'S01, S10',
         'Constats de fichier, pas taux annuel ni extrapolation géographique.', 'Quelles associations convergent après ajustement ?'),
        ('3. Les heures supplémentaires constituent le signal RH le plus convergent',
         'Le descriptif et la logistique ajustée orientent le même diagnostic.',
         f'130/432 (30,09 %) contre 118/1 086 (10,87 %) ; OR L2 {or_ot.odds_ratio:.2f} '
         f'[{or_ot.stability_low:.2f} ; {or_ot.stability_high:.2f}].',
         'figures/02_odds_ratios.png ; proportions brutes en encadré', 'S03, S06',
         'Association observée/ajustée ; intervalle de stabilité, pas effet causal.', 'Où commencer les entretiens et le diagnostic de charge ?'),
        ('4. Le cumul heures supplémentaires–niveau codé 1 mérite une revue ciblée',
         'Prioriser le diagnostic par situation de travail, sans score individuel.',
         f'{int(top.departures)}/{int(top.n)} départs ({top.rate:.2%}) dans le cumul ; supports train/test dans le tableau annexe.',
         'figures/04_sous_groupes.png (Wilson et effectifs)', 'S08, tables/subgroup_support.csv',
         'Segmentation exploratoire ; code métier à valider ; pas interaction causale.', 'Cette priorité de rétention suffit-elle à juger l’égalité ?'),
        ('5. Un faible signal du genre ne permet pas de conclure à l’égalité',
         'L’égalité doit être évaluée sur la rémunération, la représentation et les processus.',
         f'Revenu standardisé : femmes {income.female:.2f}, hommes {income.male:.2f}, écart {income.female_vs_male_percent:.2f} % ; '
         f'couverture {int(income.covered_n)}/1 214 salariés train.',
         'Tableau comparatif et cartouche « temps de travail / responsabilités non ajustés »', 'S07, tables/gender_standardization.csv',
         'Écarts descriptifs ; aucune preuve de discrimination ni mesure du taux de promotion.', 'Quelle confiance accorder aux modèles qui soutiennent ce diagnostic ?'),
        ('6. Les modèles apportent un signal, sans justifier un déploiement individuel',
         'La robustesse du diagnostic prime sur la recherche d’un algorithme plus complexe.',
         'AUC logistique 0,869 [0,797 ; 0,925] ; forêt 0,819 [0,753 ; 0,887] ; rappel 32 % / 24 % au seuil 0,5.',
         'figures/01_performance.png ; AP et rappel en encadré', 'S02, S11, S13',
         'IC conditionnels ; test déjà consulté ; aucune supériorité déduite du chevauchement des IC.', 'Comment transformer ces signaux en actions évaluables ?'),
        ('7. Engager trois chantiers mesurables, puis valider leurs effets',
         'Pilote charge/horaires ; revue des parcours ; audit d’égalité et données datées.',
         'Les justifications sont celles des slides 1–5 ; aucun objectif de baisse ou ROI présenté comme estimé.',
         'Matrice action / population / responsable proposé / KPI / horizon 0–3 mois', 'HR_RECOMMENDATIONS.md',
         'Recommandations et hypothèses à tester, distinctes des constats.', 'Annexes : protocole, tableaux complets, audit et exploration 06.')]
    sections = ['# Storyline consulting — 7 slides principales',
                'Logique Minto : réponse business → ampleur → preuves → population → égalité → limites de confiance → action. '
                'Utiliser uniquement les quatre figures du dossier `reports/figures/`, fournies en PNG 300 DPI et SVG. '
                'Les nombres arrondis des slides renvoient aux valeurs non arrondies des tableaux.', source_note]
    for title, message, figures, chart, source, status, transition in slides:
        sections += ['## '+title, '**Message clé** : '+message, '**Chiffres** : '+figures,
                     '**Visuel recommandé** : '+chart, '**Sources exactes** : '+source,
                     '**Statut des affirmations** : '+status, '**Transition** : '+transition]
    sections += ['## Annexes techniques',
        '- Dictionnaire, 23 incohérences/20 lignes et simulations 02/03 ; audit 00.\n'
        '- Prétraitements, paramètres, indices et preuves de reproduction.\n'
        '- OR, intervalles de stabilité, contraste de scores non causal.\n'
        '- Ablations, modèles par groupe seul, corrélations et sensibilité d’affectation.\n'
        '- Sous-groupes train/test, strates exclues et couverture de l’égalité.\n'
        '- Exploration 06 et seuil F2 : pas de gain robuste ; coût des faux positifs.\n'
        '- Sources et limites temporelles/géographiques.']
    write('SLIDE_STORYLINE.md', sections)
    sources = [
        ('S01','Dataset et cible','src/modeling.py::load_model_data; src/data.py::load_data','notebooks/00_audit_variables.ipynb','data/variantes/01_baseline.xlsx','Comptage et égalité cellule par cellule avec la source','1518 salariés'),
        ('S02','Performance et IC','src/reporting.py::performance_checks; src/modeling.py::evaluate_test','notebooks/04_attrition_model.ipynb; notebooks/05_random_forest.ipynb','outputs/group_feature_analysis/model_test_performance.csv','Scores sauvegardés reproduits ; 500 bootstraps stratifiés du test','304 salariés / 50 départs'),
        ('S03','OR et stabilité','src/reporting.py::selected_log_odds; bootstrap_associations','notebooks/04_attrition_model.ipynb','reports/tables/logistic_or_stability.csv','Contrastes L2 ; 300 réajustements bootstrap entraînement','1214 salariés / 198 départs'),
        ('S04','Permutation groupée','src/group_feature_analysis.py::permutation_analysis','sans notebook : python -m src.group_feature_analysis','outputs/group_feature_analysis/group_importance_logistic.csv; outputs/group_feature_analysis/group_importance_random_forest.csv','30 permutations conjointes ; IC conditionnels de 500 bootstraps','304 salariés test'),
        ('S05','Ablation','src/group_feature_analysis.py::cv_comparisons','sans notebook : script groupes','outputs/group_feature_analysis/group_ablation_results.csv','Retrait et réentraînement à paramètres fixes ; mêmes 5 plis','1214 salariés entraînement'),
        ('S06','Associations brutes','src/data.py::attrition_table; src/reporting.py::main','notebooks/03_working_conditions.ipynb','outputs/tables/03_conditions_travail/03_OverTime.csv','Départs / effectif ; recomptage et IC Wilson','1518 salariés'),
        ('S07','Égalité et couverture','src/group_feature_analysis.py::gender_audit','notebooks/02_gender_equality.ipynb','outputs/group_feature_analysis/gender_standardized_outcomes.csv','Poids communs métier/niveau ; min 1/5/10 de chaque genre ; comparer périmètres','train 1214 ; communs min5 : 1150'),
        ('S08','Sous-groupes','src/reporting.py::subgroup_analysis','notebooks/03_working_conditions.ipynb (strates historiques)','reports/tables/subgroup_support.csv','OverTime × niveau codé1 ; Wilson ; support train puis divulgation test','ensemble, train et test séparés'),
        ('S09','Écart de scores ajustés','src/reporting.py::adjusted_score_contrast; bootstrap_associations','sans notebook : consolidation','reports/tables/adjusted_score_contrast.csv','Moyenne p(HS oui)-p(HS non) sur X_train fixe ; 300 bootstraps L2','1214 salariés ; score non calibré, non causal'),
        ('S10','Qualité des données','src/quality.py; scripts/create_excel_variants.py','notebooks/00_audit_variables.ipynb','outputs/tables/00_audit/00_audit_synthese.csv; data/variantes/tracabilite/journal_lignes_supprimees.csv','Contrôles d’ancienneté ; 23 règles en échec, 20 lignes distinctes','1518 salariés'),
        ('S11','Différence AUC appariée','src/reporting.py::performance_checks','sans notebook : consolidation','reports/tables/paired_model_difference.csv','500 tirages test communs aux deux modèles ; différence AUC à chaque tirage','304 salariés test'),
        ('S12','Corrélations','src/features.py::numeric_redundancies; src/group_feature_analysis.py::correlations','notebooks/01_exploration.ipynb','outputs/group_feature_analysis/feature_correlations_training.csv','Pearson/Spearman ; V de Cramér dans le fichier adjacent','1214 salariés entraînement'),
        ('S13','Exploration RF et seuil','src/forest_improvement.py','notebooks/archives/06_random_forest_improvement.ipynb','outputs/archives/06_random_forest_improvement/nested_pooled_scores.csv; outputs/archives/06_random_forest_improvement/test_comparison.csv','20 configurations ; 5 plis externes ×3 internes ; F2 sur OOF train','train 1214 puis test historique 304'),
        ('S14','SHAP indisponible','src/group_feature_analysis.py::main','sans notebook : script groupes','outputs/group_feature_analysis/analysis_metadata.json','Tentative optionnelle ; bibliothèque absente ; aucune valeur SHAP calculée','aucune conclusion SHAP')]
    pd.DataFrame(sources, columns=['id','result','script_function','notebook','output','method','population']).to_csv(REPORTS/'tables/source_register.csv',index=False,encoding='utf-8-sig')
    write('figures/README.md', ['# Figures finales pour la présentation',
        '| Fichier PNG/SVG | Message | Table source |\n|---|---|---|\n'
        '| `01_performance` | Signal prédictif et incertitude | `../tables/model_performance.csv` |\n'
        '| `02_odds_ratios` | Associations ajustées et stabilité L2 | `../tables/logistic_or_stability.csv` |\n'
        '| `03_groupes_rh` | Contributions prédictives des groupes | `../tables/group_permutation_importance.csv` |\n'
        '| `04_sous_groupes` | Situations de travail à diagnostiquer | `../tables/subgroup_support.csv` |',
        'Format 16:9, PNG 300 DPI et SVG. Les barres des OR sont des intervalles de stabilité de l’estimateur L2 ; '
        'les barres des taux sont des IC Wilson ; les autres IC sont conditionnels aux modèles fixés. '
        'Aucune figure historique ou exploratoire secondaire n’est copiée ici. Les détails sont dans `../MODEL_RESULTS.md`.'])
