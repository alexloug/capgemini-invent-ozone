"""Variables candidates d'exploration, sans modification des colonnes d'origine."""
import numpy as np
import pandas as pd


def numeric_redundancies(frame, columns):
    """Toutes les paires numériques, sans sélectionner sur la cible."""
    pearson = frame[columns].corr(method='pearson')
    spearman = frame[columns].corr(method='spearman')
    rows = []
    for i, left in enumerate(columns):
        for right in columns[i+1:]:
            rows.append({'variable_1': left, 'variable_2': right,
                         'pearson': pearson.loc[left, right],
                         'spearman': spearman.loc[left, right]})
    result = pd.DataFrame(rows)
    result['correlation_max_absolue'] = result[['pearson', 'spearman']].abs().max(axis=1)
    return result.sort_values('correlation_max_absolue', ascending=False), spearman


def build_candidates(frame):
    """Masque uniquement les nouvelles valeurs non calculables ou incohérentes.

    NaN signifie ici « variable dérivée non interprétable », sans corriger la source.
    Les ancêtres et sous-durées doivent être non négatifs et cohérents.
    """
    company = frame.YearsAtCompany
    total = frame.TotalWorkingYears
    company_ok = company.ge(0) & total.ge(company)
    role_ok = company_ok & frame.YearsInCurrentRole.between(0, company)
    manager_ok = company_ok & frame.YearsWithCurrManager.between(0, company)
    result = pd.DataFrame(index=frame.index)
    result['PriorExperienceYears'] = (total-company).where(company_ok)
    result['CompanyTenureShare'] = (company / total.where(total.gt(0))).where(company_ok)
    result['YearsBeforeCurrentRole'] = (company-frame.YearsInCurrentRole).where(role_ok)
    result['RoleTenureShare'] = (frame.YearsInCurrentRole / company.where(company.gt(0))).where(role_ok)
    result['YearsBeforeCurrentManager'] = (company-frame.YearsWithCurrManager).where(manager_ok)
    result['OverTimeFrequentTravel'] = (frame.OverTime.eq('Yes') & frame.BusinessTravel.eq('Travel_Frequently')).astype(int)
    result['DistanceIfOverTime'] = frame.DistanceFromHome * frame.OverTime.map({'No': 0, 'Yes': 1})
    return result


def fit_income_reference(train):
    """Référence descriptive apprise sur l'entraînement, jamais sur le test."""
    return (train.groupby(['JobRole', 'JobLevel'], observed=True).MonthlyIncome
            .agg(median_income='median', reference_count='size').reset_index())


def relative_income(frame, reference):
    """Pas de repli arbitraire : une strate inconnue ou petite donne NaN.

    Le seuil de cinq salariés est un filtre de prudence modifiable, pas une
    garantie de fiabilité. La référence n'utilise ni Attrition ni depart.
    """
    joined = frame[['JobRole', 'JobLevel', 'MonthlyIncome']].merge(reference, on=['JobRole', 'JobLevel'], how='left', sort=False)
    valid = joined.reference_count.ge(5) & joined.median_income.gt(0)
    return pd.Series((joined.MonthlyIncome / joined.median_income).where(valid).to_numpy(),
                     index=frame.index, name='IncomeToRoleLevelMedian')
