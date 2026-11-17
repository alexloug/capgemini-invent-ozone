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


def plot_model_comparison(y_test, predictions):
    """Figures du test à partir des prédictions déjà calculées, sans réévaluation."""
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
    fig, axes = plt.subplots(1, len(predictions), figsize=(14, 4))
    for ax, (name, result) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay(result['matrice'], display_labels=['Non-départ', 'Départ']).plot(
            ax=ax, colorbar=False, cmap='Blues')
        ax.set(title=f'{name} - seuil 0,5', xlabel='Classe prédite', ylabel='Classe observée')
        ax.grid(False)
    save_figure(fig, '04_matrices_confusion')
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, result in predictions.items():
        PrecisionRecallDisplay.from_predictions(y_test, result['probabilites'], name=name, ax=ax)
    ax.axhline(y_test.mean(), linestyle='--', color='gray', label='Prévalence du test')
    ax.set(xlabel='Rappel', ylabel='Précision', title='Comparaison sur le test réservé')
    ax.legend()
    save_figure(fig, '04_precision_rappel')
