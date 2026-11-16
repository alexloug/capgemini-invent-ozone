"""Graphiques communs et export PNG."""
import matplotlib.pyplot as plt
import seaborn as sns
from .data import FIGURES, attrition_table

sns.set_theme(style="whitegrid", palette="colorblind")


def save_figure(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_attrition(df, group, name):
    table = attrition_table(df, group)
    fig, ax = plt.subplots(figsize=(max(7, len(table) * .8), 4.5))
    ax.bar(table[group].astype(str), table["proportion_depart"] * 100)
    for i, row in table.iterrows():
        ax.text(i, row.proportion_depart * 100 + 1,
                f"{row.proportion_depart:.1%}\nn={int(row.effectif)}, d={int(row.departs)}",
                ha="center", fontsize=9)
    ax.set(xlabel=group, ylabel="Proportion de départs observés (%)",
           title=f"Départs selon {group}", ylim=(0, max(10, table.proportion_depart.max()*100+15)))
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, name)
    return table
