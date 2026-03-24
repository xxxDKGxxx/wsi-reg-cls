import math

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn import clone
from sklearn.metrics import get_scorer, classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


def corr_matrix_report(X: pd.DataFrame, y: pd.Series) -> Figure :
    df = pd.concat([X, y], axis=1)

    corr_matrix = df.corr(numeric_only=True)

    fig = plt.figure(figsize=(20, 20))

    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Matrix", fontsize=16)

    return fig


def plot_distributions(X: pd.DataFrame, y: pd.Series) -> plt.Figure:
    dataframe = pd.concat([X, y], axis=1)

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

def evaluate_model_cv(model: Pipeline, X_train, y_train, experiment_params: dict) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=experiment_params.get("random_state"))

    res = cross_validate(model, X_train, y_train, scoring=experiment_params['scoring'], cv=cv)

    res_df = pd.DataFrame(res)

    summary_df = res_df.agg(['mean', 'std'])

    final_df = pd.concat([res_df, summary_df]).T

    return final_df

def evaluate_model_test(
        model: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series, experiment_params: dict) -> pd.DataFrame:
    # scoring = experiment_params.get("scoring", ["accuracy"])
    # results = {}
    #
    # for metric_name in scoring:
    #     scorer = get_scorer(metric_name)
    #     results[metric_name] = scorer(model, X_test, y_test)
    #
    # return pd.DataFrame([results])

    scoring = experiment_params.get("scoring", ["accuracy"])

    random_states = experiment_params.get("test_random_states")
    all_results = []

    for rs in random_states:
        cloned_model = clone(model)

        model_step_name = cloned_model.steps[-1][0]
        cloned_model.set_params(**{f"{model_step_name}__random_state": rs})

        cloned_model.fit(X_train, y_train)

        run_results = {}
        for metric_name in scoring:
            scorer = get_scorer(metric_name)
            run_results[metric_name] = scorer(cloned_model, X_test, y_test)

        all_results.append(run_results)

    res_df = pd.DataFrame(all_results)

    summary_df = res_df.agg(['mean', 'std'])
    final_test_df = pd.concat([res_df, summary_df]).T

    return final_test_df

def diagnose_per_class_metrics(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv
    )

    print("--- RAPORT KLASYFIKACYJNY Z WALIDACJI KRZYŻOWEJ ---")
    print(classification_report(y_train, y_pred))

    report_dict = classification_report(y_train, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={'index': 'class_or_metric'})

    return report_df