"""Crée trois variantes comparables, sans modifier la source.

Le plafonnement est une simulation autorisée, pas une correction vérifiée.
Exécution depuis la racine : .venv/bin/python scripts/create_excel_variants.py
"""
from pathlib import Path
import sys
from hashlib import sha256
import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_data

OUT = ROOT / 'data' / 'variantes'
RULES = [('YearsAtCompany', 'TotalWorkingYears'),
         ('YearsInCurrentRole', 'YearsAtCompany'),
         ('YearsSinceLastPromotion', 'YearsAtCompany'),
         ('YearsWithCurrManager', 'YearsAtCompany')]
DROP = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']


def write_excel(df, name):
    """Une variable par colonne, filtre et en-tête figé."""
    path = OUT / name
    df.to_excel(path, index=False, sheet_name='Salaries', engine='openpyxl')
    book = load_workbook(path)
    sheet = book.active
    sheet.freeze_panes = 'A2'
    table = Table(displayName='Salaries', ref=sheet.dimensions)
    table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
    sheet.add_table(table)
    for i, col in enumerate(df.columns, 1):
        sheet.column_dimensions[get_column_letter(i)].width = min(32, max(12, len(col)+2, df[col].astype(str).str.len().max()+2))
    book.save(path)
    pd.testing.assert_frame_equal(pd.read_excel(path), df.reset_index(drop=True))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data' / 'employee.csv.xlsx'
    initial_hash = sha256(source.read_bytes()).hexdigest()
    # Seules les 35 colonnes source : ne pas exporter la cible dérivée « depart ».
    original = load_data().drop(columns='depart')
    reduced = original.drop(columns=DROP)
    flags = pd.DataFrame({f'{a} > {b}': original[a] > original[b] for a, b in RULES})
    excluded = flags.any(axis=1)
    retained = reduced.loc[~excluded].copy()
    capped = reduced.copy()
    changes = []
    # L'ordre est important : plafonner l'ancienneté entreprise avant ses sous-durées.
    for child, parent in RULES:
        mask = capped[child] > capped[parent]
        for idx in capped.index[mask]:
            changes.append({'ligne_excel_source': idx+2,
                            'EmployeeNumber_source': original.loc[idx, 'EmployeeNumber'],
                            'variable': child, 'ancienne_valeur': capped.loc[idx, child],
                            'nouvelle_valeur': capped.loc[idx, parent],
                            'plafond_utilise': parent,
                            'statut': 'Simulation : valeur réelle inconnue'})
        capped[child] = capped[child].clip(upper=capped[parent])
    variants = [('01_baseline.xlsx', reduced),
                ('02_rows_removed.xlsx', retained),
                ('03_tenure_capped.xlsx', capped)]
    for name, frame in variants:
        write_excel(frame, name)
    trace_dir = OUT / 'tracabilite'
    trace_dir.mkdir(exist_ok=True)
    # Les journaux font le lien avec la source malgré le retrait de l'identifiant.
    trace = pd.DataFrame({'ligne_excel_source': original.index+2,
                          'EmployeeNumber_source': original.EmployeeNumber,
                          'ligne_excel_variantes_01_03': original.index+2,
                          'supprimee_variante_02': excluded})
    trace['ligne_excel_variante_02'] = pd.Series(pd.NA, index=trace.index, dtype='Int64')
    trace.loc[~excluded, 'ligne_excel_variante_02'] = range(2, len(retained)+2)
    deleted = trace.loc[excluded, ['ligne_excel_source', 'EmployeeNumber_source']].copy()
    deleted['motifs'] = flags.loc[excluded].apply(lambda row: ' ; '.join(row.index[row]), axis=1)
    deleted.to_csv(trace_dir/'journal_lignes_supprimees.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(changes).to_csv(trace_dir/'journal_valeurs_plafonnees.csv', index=False, encoding='utf-8-sig')
    trace.to_csv(trace_dir/'correspondance_lignes.csv', index=False, encoding='utf-8-sig')
    summary = pd.DataFrame([{'fichier': name, 'lignes': len(frame), 'colonnes': len(frame.columns)} for name, frame in variants])
    summary.to_csv(trace_dir/'resume_variantes.csv', index=False, encoding='utf-8-sig')
    for child, parent in RULES:
        assert not (retained[child] > retained[parent]).any()
        assert not (capped[child] > capped[parent]).any()
    assert sha256(source.read_bytes()).hexdigest() == initial_hash
    assert (original.loc[~excluded].drop(columns=DROP) == retained).all().all()
    (OUT/'README.md').write_text(f'''# Variantes du fichier salariés

La source `data/employee.csv.xlsx` reste intacte. Chaque classeur contient une variable par colonne et aucune variable dérivée `depart`.

- `01_baseline.xlsx` : retrait de {', '.join(DROP)}, aucune ligne ni valeur modifiée.
- `02_rows_removed.xlsx` : retrait des mêmes colonnes et des {int(excluded.sum())} lignes distinctes concernées par les {int(flags.sum().sum())} incohérences initiales.
- `03_tenure_capped.xlsx` : toutes les lignes, sans les quatre colonnes, avec plafonnement séquentiel autorisé : YearsAtCompany ≤ TotalWorkingYears, puis YearsInCurrentRole, YearsSinceLastPromotion et YearsWithCurrManager ≤ YearsAtCompany plafonné.

Le plafonnement modifie {len(changes)} cellules sur {len(set(c['ligne_excel_source'] for c in changes))} lignes distinctes. Il peut modifier davantage de cellules que les 23 alertes initiales, car abaisser YearsAtCompany peut imposer de nouveaux plafonnements sur ses sous-durées. Il s'agit d'une simulation de cohérence, pas d'une correction dont on connaît la valeur réelle. L'expérience totale est conservée comme plafond par convention ; elle pourrait elle-même être erronée.

Les journaux CSV, dans `tracabilite/`, indiquent les numéros de ligne et identifiants de la source, les motifs de retrait et chaque ancienne/nouvelle valeur. `correspondance_lignes.csv` permet de retrouver les lignes après suppression. Les exports sont régénérés sous les mêmes noms à chaque exécution de `scripts/create_excel_variants.py`.

Pour l'étude, garder la version 01 comme référence et utiliser 02/03 pour une analyse de sensibilité. La suppression peut biaiser les résultats si les lignes retirées ont des profils particuliers ; le plafonnement introduit une hypothèse. Les notebooks 00, 02 et 03 chargent la source originale ; le notebook 01 et les modèles 04 et 05 chargent la variante 01. Les variantes 02 et 03 ne sont pas utilisées automatiquement.
''', encoding='utf-8')
    print(summary.to_string(index=False))
    print(f'{len(changes)} cellules plafonnées ; {int(excluded.sum())} lignes supprimées dans la variante 02.')


if __name__ == '__main__':
    main()
