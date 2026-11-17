"""Audit sans correction : les règles métier proposées restent à valider."""
import csv
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import ROOT


def read_raw():
    """Conserve les espaces et les chaînes ; contrôle les largeurs des lignes CSV."""
    raw = pd.read_excel(ROOT / "data/employee.csv.xlsx", header=None,
                        dtype=str, keep_default_na=False, engine="openpyxl")
    if raw.shape[1] == 1 and "," in str(raw.iloc[0, 0]):
        rows = [next(csv.reader([value])) for value in raw.iloc[:, 0]]
    else:
        rows = raw.values.tolist()
    header, records = rows[0], rows[1:]
    if len(header) != len(set(header)):
        raise ValueError("Noms de colonnes dupliqués : examiner le classeur.")
    widths = Counter(map(len, records))
    if any(len(row) != len(header) for row in records):
        raise ValueError(f"Nombre de champs incorrect : {dict(widths)}")
    # L'index correspond aux lignes Excel, en comptant l'en-tête en ligne 1.
    df = pd.DataFrame(records, columns=header, index=range(2, len(records)+2))
    df.index.name = "ligne_excel"
    return df


def typed_copy(raw, numeric_columns):
    """Copie d'analyse : coercitions visibles dans le contrôle des types."""
    df = raw.copy()
    for col in numeric_columns:
        df[col] = pd.to_numeric(raw[col].str.strip().replace("", np.nan), errors="coerce")
    return df


def profile(raw, df):
    rows = []
    for col in raw:
        numeric = pd.api.types.is_numeric_dtype(df[col])
        rows.append({"variable": col, "type_analyse": str(df[col].dtype),
                     "effectif": len(df), "vides": raw[col].str.strip().eq("").sum(),
                     "espaces_exterieurs": raw[col].ne(raw[col].str.strip()).sum(),
                     "valeurs_distinctes_brutes": raw[col].nunique(),
                     "non_numeriques": (raw[col].str.strip().ne("") & df[col].isna()).sum() if numeric else 0,
                     "minimum": df[col].min() if numeric else np.nan,
                     "maximum": df[col].max() if numeric else np.nan})
    return pd.DataFrame(rows)


def review_variable(raw, df, col):
    """Résumés et figure pour une variable, sans interpréter les extrêmes comme erreurs."""
    from IPython.display import display
    s = df[col]
    print(f"{col} : {len(s)} lignes ; {raw[col].nunique()} valeurs distinctes brutes")
    if pd.api.types.is_numeric_dtype(s):
        display(s.describe(percentiles=[.01, .25, .5, .75, .99]).to_frame("valeur"))
    # Toutes les modalités sont montrées pour les variables peu cardinales.
    if raw[col].nunique() <= 40 or not pd.api.types.is_numeric_dtype(s):
        frequencies = raw[col].value_counts(dropna=False).rename_axis("valeur_brute").reset_index(name="effectif")
        frequencies["part"] = frequencies.effectif / len(raw)
        display(frequencies)
    else:
        display(pd.DataFrame({"minimums": s.nsmallest(5).to_numpy(),
                              "maximums": s.nlargest(5).to_numpy()}))
    if col == "EmployeeNumber":
        print("Identifiant : une distribution numérique ne mesure pas un phénomène RH.")
        return
    fig, ax = plt.subplots(figsize=(8, 3.6))
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 15:
        ax.hist(s.dropna(), bins=25, edgecolor="white")
        ax.set(xlabel=col, ylabel="Effectif", title=f"Distribution de {col}")
    else:
        counts = s.value_counts(dropna=False).sort_index()
        ax.barh(counts.index.astype(str), counts.values)
        ax.set(xlabel="Effectif", ylabel=col, title=f"Modalités observées de {col}")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def find_anomalies(raw, df, numeric_columns, expected):
    """Une ligne par règle et observation ; les alertes peuvent se chevaucher."""
    issues = []
    def add(mask, variable, rule, kind, values):
        for idx in df.index[mask.fillna(False)]:
            issues.append({"ligne_excel": idx, "EmployeeNumber": raw.loc[idx, "EmployeeNumber"],
                           "variable": variable, "categorie": kind, "regle": rule,
                           "valeur": str(values.loc[idx])})
    for col in raw:
        add(raw[col].str.strip().eq(""), col, "Valeur vide", "complétude", raw[col])
        add(raw[col].ne(raw[col].str.strip()), col, "Espaces en début ou fin", "format", raw[col])
    for col in numeric_columns:
        s = df[col]
        add(raw[col].str.strip().ne("") & s.isna(), col, "Valeur non numérique", "type", raw[col])
        add(s.notna() & ~np.isfinite(s), col, "Valeur numérique non finie", "type", raw[col])
        add(s < 0, col, "Valeur négative pour une quantité supposée non négative", "plausibilité", raw[col])
        add(s.notna() & (s % 1 != 0), col, "Valeur non entière : vérifier la précision attendue", "format à valider", raw[col])
    for col, values in expected.items():
        add(~raw[col].isin(values), col, f"Modalité hors liste proposée : {values}", "dictionnaire à valider", raw[col])
    add(df.EmployeeNumber.duplicated(keep=False), "EmployeeNumber", "Identifiant dupliqué", "unicité", raw.EmployeeNumber)
    add(df.EmployeeNumber <= 0, "EmployeeNumber", "Identifiant non positif : vérifier la convention", "format à valider", raw.EmployeeNumber)
    add(raw.duplicated(keep=False), "toutes", "Ligne intégralement dupliquée", "unicité", raw.EmployeeNumber)
    for left, right in [("YearsAtCompany", "TotalWorkingYears"),
                        ("YearsInCurrentRole", "YearsAtCompany"),
                        ("YearsSinceLastPromotion", "YearsAtCompany"),
                        ("YearsWithCurrManager", "YearsAtCompany"),
                        ("TotalWorkingYears", "Age")]:
        pair = df[left].astype(str) + " > " + df[right].astype(str)
        add(df[left] > df[right], left, f"{left} > {right}", "cohérence entre variables", pair)
    # L'âge minimal de début de carrière est une hypothèse, pas une règle légale.
    add(df.Age - df.TotalWorkingYears < 14, "TotalWorkingYears",
        "Âge implicite de début de carrière < 14 ans : cas à examiner",
        "plausibilité à valider", df.Age - df.TotalWorkingYears)
    return pd.DataFrame(issues, columns=["ligne_excel", "EmployeeNumber", "variable", "categorie", "regle", "valeur"])


def iqr_alerts(df, columns):
    """Repérage statistique, qui ne constitue pas un contrôle de validité."""
    rows = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile([.25, .75])
        lower, upper = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        rows.append({"variable": col, "borne_basse_iqr": lower, "borne_haute_iqr": upper,
                     "effectif_atypique": ((s < lower) | (s > upper)).sum()})
    return pd.DataFrame(rows)
