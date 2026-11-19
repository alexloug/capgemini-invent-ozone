# Storyline consulting — 7 slides principales

Logique Minto : réponse business → ampleur → preuves → population → égalité → limites de confiance → action. Utiliser uniquement les quatre figures du dossier `reports/figures/`, fournies en PNG 300 DPI et SVG. Les nombres arrondis des slides renvoient aux valeurs non arrondies des tableaux.

**Traçabilité** : les identifiants S01–S14 renvoient au [registre exact des sources](tables/source_register.csv), qui indique script/fonction, notebook, sortie, méthode et population. Les chiffres des figures proviennent exclusivement de `reports/tables/`.

## 1. Prioriser les conditions de travail, sans déprioriser l’égalité

**Message clé** : Réponse business : lancer un diagnostic charge/organisation, maintenir un audit d’égalité distinct.

**Chiffres** : Permutation conditions : ΔAUC 0,127 logistique / 0,110 forêt ; parcours 0,076 / 0,046.

**Visuel recommandé** : figures/03_groupes_rh.png (ou SVG)

**Sources exactes** : S04–S05, tables/group_permutation_importance.csv

**Statut des affirmations** : Résultats prédictifs + recommandation ; effets des politiques non estimés.

**Transition** : Quel est le périmètre du problème mesuré ?

## 2. La base décrit 248 départs, mais ne mesure pas l’absentéisme

**Message clé** : Cadrer la population et les limites dès le début.

**Chiffres** : 1 518 salariés ; 248 départs (16,34 %) ; test 304/50 ; aucune période documentée.

**Visuel recommandé** : Tableau 3 lignes : effectif / départs / données manquantes au sens métier

**Sources exactes** : S01, S10

**Statut des affirmations** : Constats de fichier, pas taux annuel ni extrapolation géographique.

**Transition** : Quelles associations convergent après ajustement ?

## 3. Les heures supplémentaires constituent le signal RH le plus convergent

**Message clé** : Le descriptif et la logistique ajustée orientent le même diagnostic.

**Chiffres** : 130/432 (30,09 %) contre 118/1 086 (10,87 %) ; OR L2 3.34 [2.57 ; 4.29].

**Visuel recommandé** : figures/02_odds_ratios.png ; proportions brutes en encadré

**Sources exactes** : S03, S06

**Statut des affirmations** : Association observée/ajustée ; intervalle de stabilité, pas effet causal.

**Transition** : Où commencer les entretiens et le diagnostic de charge ?

## 4. Le cumul heures supplémentaires–niveau codé 1 mérite une revue ciblée

**Message clé** : Prioriser le diagnostic par situation de travail, sans score individuel.

**Chiffres** : 83/158 départs (52.53%) dans le cumul ; supports train/test dans le tableau annexe.

**Visuel recommandé** : figures/04_sous_groupes.png (Wilson et effectifs)

**Sources exactes** : S08, tables/subgroup_support.csv

**Statut des affirmations** : Segmentation exploratoire ; code métier à valider ; pas interaction causale.

**Transition** : Cette priorité de rétention suffit-elle à juger l’égalité ?

## 5. Un faible signal du genre ne permet pas de conclure à l’égalité

**Message clé** : L’égalité doit être évaluée sur la rémunération, la représentation et les processus.

**Chiffres** : Revenu standardisé : femmes 6358.49, hommes 6424.07, écart -1.02 % ; couverture 1150/1 214 salariés train.

**Visuel recommandé** : Tableau comparatif et cartouche « temps de travail / responsabilités non ajustés »

**Sources exactes** : S07, tables/gender_standardization.csv

**Statut des affirmations** : Écarts descriptifs ; aucune preuve de discrimination ni mesure du taux de promotion.

**Transition** : Quelle confiance accorder aux modèles qui soutiennent ce diagnostic ?

## 6. Les modèles apportent un signal, sans justifier un déploiement individuel

**Message clé** : La robustesse du diagnostic prime sur la recherche d’un algorithme plus complexe.

**Chiffres** : AUC logistique 0,869 [0,797 ; 0,925] ; forêt 0,819 [0,753 ; 0,887] ; rappel 32 % / 24 % au seuil 0,5.

**Visuel recommandé** : figures/01_performance.png ; AP et rappel en encadré

**Sources exactes** : S02, S11, S13

**Statut des affirmations** : IC conditionnels ; test déjà consulté ; aucune supériorité déduite du chevauchement des IC.

**Transition** : Comment transformer ces signaux en actions évaluables ?

## 7. Relier le diagnostic à la décision de pilote

Pour le rendu final, remplacer cette slide de transition par les [quatre slides du HR Action Business Case](HR_ACTION_BUSINESS_CASE_SLIDES.md). Leur trame en anglais couvre les actions, le budget et les seuils, le gain par scénario et la décision de déploiement.

**Message clé** : Pilote charge/horaires ; revue des parcours ; audit d’égalité et données datées.

**Chiffres** : Les justifications sont celles des slides 1–5 ; aucun objectif de baisse ou ROI présenté comme estimé.

**Visuel recommandé** : Matrice action / population / responsable proposé / KPI / horizon 0–3 mois

**Sources exactes** : HR_RECOMMENDATIONS.md

**Statut des affirmations** : Recommandations et hypothèses à tester, distinctes des constats.

**Transition** : Annexes : protocole, tableaux complets, audit et exploration 06.

## Annexes techniques

- Dictionnaire, 23 incohérences/20 lignes et simulations 02/03 ; audit 00.
- Prétraitements, paramètres, indices et preuves de reproduction.
- OR, intervalles de stabilité, contraste de scores non causal.
- Ablations, modèles par groupe seul, corrélations et sensibilité d’affectation.
- Sous-groupes train/test, strates exclues et couverture de l’égalité.
- Exploration 06 et seuil F2 : pas de gain robuste ; coût des faux positifs.
- Sources et limites temporelles/géographiques.
