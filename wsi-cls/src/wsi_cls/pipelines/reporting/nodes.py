import math

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def corr_matrix_report(df: pd.DataFrame) -> Figure :
    corr_matrix = df.corr(numeric_only=True)

    fig = plt.figure(figsize=(20, 20))

    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Macierz Korelacji", fontsize=16)

    return fig


def plot_distributions(dataframe: pd.DataFrame) -> plt.Figure:
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    n_cols = len(numeric_cols)

    cols_grid = round(math.sqrt(n_cols))
    rows_grid = math.ceil(n_cols / cols_grid)

    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(20, 5 * rows_grid))

    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]

        sns.histplot(data=dataframe, x=col, ax=ax, kde=True, bins=30, color="skyblue")

        ax.set_title(f'Rozkład: {col}', fontweight='bold')
        ax.set_ylabel('Liczba wystąpień')
        ax.set_xlabel('')

    for i in range(n_cols, len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()

    return fig
