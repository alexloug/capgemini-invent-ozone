"""Chargement reproductible sans écriture dans la source Excel."""
from pathlib import Path
from io import StringIO
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"


def load_data(path=None):
    """Accepte un classeur tabulaire ou des lignes CSV dans sa première colonne."""
    path = Path(path) if path else ROOT / "data" / "employee.csv.xlsx"
    df = pd.read_excel(path, engine="openpyxl")
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        lines = [str(df.columns[0])] + df.iloc[:, 0].dropna().astype(str).tolist()
        df = pd.read_csv(StringIO("\n".join(lines)))
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip().replace("", pd.NA)
    if "Attrition" not in df or not df["Attrition"].isin(["Yes", "No"]).all():
        raise ValueError("Attrition doit contenir uniquement Yes et No, sans valeur manquante.")
    df["depart"] = df["Attrition"].map({"No": 0, "Yes": 1})
    return df


def export_table(table, name, index=False):
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / f"{name}.csv", index=index, encoding="utf-8-sig")
    return table


def attrition_table(df, groups):
    """Dénominateur : salariés observés du groupe ; aucune annualisation."""
    return (df.groupby(groups, observed=True, dropna=False)["depart"]
            .agg(effectif="size", departs="sum", proportion_depart="mean").reset_index())
