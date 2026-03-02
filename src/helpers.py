import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from pycaret.regression import setup, compare_models, evaluate_model, predict_model

def missing_value_table(df, name: str, as_rate: bool = True) -> pd.Series:
    """Return missing-value counts or rates per column."""
    counts = df.isna().sum()
    if as_rate:
        print(f"\nMissing-value rate (%) for {name}:")
        return (counts / len(df) * 100).round(2)
    print(f"\nMissing-value counts for {name}:")
    return counts

def plot_spearman_heatmap(df, name: str) -> None:
    """Plot Spearman correlation heatmap (lower-right triangle, no diagonal)."""
    corr = df.select_dtypes("number").corr(method="spearman")

    # Mask upper triangle + diagonal
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask,
        annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title(f"Spearman correlation – {name}", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_histogram(df: pd.DataFrame, col: str, display_name: str, fzx: int = 10, fzy: int = 5, bins: int = 30, ) -> None:
    """
    Tracer un histogramme pour une colonne d'un dataframe.
    
    Paramètre:
    - df           : DataFrame 
    - col          : Nom de colonne
    - display_name : Nom sur l'histogramme
    - bins         : Nombre de barre d'histogramme
    """
    if col not in df.columns:
        print(f"No column {col}")
        return

    data = df[col].dropna()

    fig, ax = plt.subplots(figsize=(fzx, fzy))
    ax.hist(data, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)

    ax.set_title(f"Distribution de {display_name}", fontsize=14)
    ax.set_xlabel(display_name, fontsize=12)
    ax.set_ylabel("Effectif", fontsize=12)

    # Annotations statistiques
    ax.axvline(data.mean(),   color="tomato",  linestyle="--", linewidth=1.5, label=f"Moyenne : {data.mean():.2f}")
    ax.axvline(data.median(), color="orange",  linestyle="--", linewidth=1.5, label=f"Médiane : {data.median():.2f}")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def run_linear_regression(
    df: pd.DataFrame,
    name: str,
    y_col: str,
    x_cols: list[str],
    impute: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train OLS regression on complete rows; optionally impute missing y_col values.
    Returns the (possibly updated) DataFrame.
    """
    required = [y_col] + list(x_cols)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"\n[{name}] Skipping – missing columns: {missing_cols}")
        return df

    complete = df[required].notna().all(axis=1)
    data = df[complete]
    if data.empty:
        print(f"\n[{name}] No complete rows for '{y_col}' regression.")
        return df

    X, y = data[x_cols].values, data[y_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LinearRegression().fit(X_tr, y_tr)
    r2_tr, r2_te = model.score(X_tr, y_tr), model.score(X_te, y_te)

    print(f"\n[{name}] {y_col} ~ {', '.join(x_cols)}")
    print(f"  rows: {len(data)}  train: {len(X_tr)}  test: {len(X_te)}")
    print(f"  intercept: {model.intercept_:.4f}")
    for col, coef in zip(x_cols, model.coef_):
        print(f"  {col}: {coef:.4f}")
    print(f"  R² train: {r2_tr:.4f}  |  R² test: {r2_te:.4f}")

    if impute:
        mask = df[y_col].isna() & df[x_cols].notna().all(axis=1)
        n = mask.sum()
        if n:
            df.loc[mask, y_col] = model.predict(df.loc[mask, x_cols].values)
            print(f"  Imputed {n} missing '{y_col}' values.")
        still = df[y_col].isna().sum()
        if still:
            print(f"  {still} rows still missing '{y_col}' (incomplete predictors).")

    return df


def run_mice_imputation(
    df: pd.DataFrame,
    cols: list[str],
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    MICE imputation in 3 rigorous steps:
      1. Fit IterativeImputer on complete rows only → evaluate R² on train/test split
      2. Print R² diagnostics (unbiased — no imputed values involved)
      3. Impute missing rows using the fitted imputer
    """
    targets = [c for c in cols if c in df.columns and df[c].isna().any()]
    print(f"\n[{name}] MICE imputation on: {targets}")

    numeric      = df.select_dtypes("number")
    complete_mask = numeric.notna().all(axis=1)
    complete_rows = numeric[complete_mask]
    missing_rows  = numeric[~complete_mask]

    print(f"  Complete rows : {len(complete_rows)} | Rows with missing values: {len(missing_rows)}")

    # ── Step 1 : Split complete rows into train / test ────────
    train_df, test_df = train_test_split(complete_rows, test_size=test_size, random_state=random_state)

    # ── Step 2 : Fit MICE on train, evaluate on test ──────────
    imputer = IterativeImputer(random_state=random_state)
    imputer.fit(train_df)

    train_imputed = pd.DataFrame(imputer.transform(train_df), columns=numeric.columns, index=train_df.index)
    test_imputed  = pd.DataFrame(imputer.transform(test_df),  columns=numeric.columns, index=test_df.index)

    print(f"\n  {'Column':<42} {'R² train':>10} {'R² test':>10} {'N train':>10} {'N test':>8}")
    print(f"  {'-'*80}")

    for col in targets:
        others = [c for c in targets if c != col]
        if not others:
            continue

        # Train
        X_tr, y_tr = train_imputed[others].values, train_imputed[col].values
        # Test — use ORIGINAL values (ground truth, no imputation involved)
        X_te, y_te = test_df[others].values,       test_df[col].values

        model    = LinearRegression().fit(X_tr, y_tr)
        r2_train = model.score(X_tr, y_tr)
        r2_test  = model.score(X_te, y_te)

        print(f"  {col:<42} {r2_train:>10.4f} {r2_test:>10.4f} {len(X_tr):>10} {len(X_te):>8}")

    # ── Step 3 : Impute the actual missing rows ───────────────
    print(f"\n  Imputing {len(missing_rows)} rows with missing values...")
    full_imputed = pd.DataFrame(
        imputer.transform(numeric),
        columns=numeric.columns, index=numeric.index,
    )
    df[targets] = full_imputed[targets]
    print(f"  Done. Missing values remaining: {df[targets].isna().sum().sum()}")


def run_mice_imputation_xgboost(
    df: pd.DataFrame,
    cols: list[str],
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> None:
    """
    MICE imputation using XGBoost as the base estimator:
      1. Fit IterativeImputer with XGBoostRegressor on complete rows → evaluate R² on train/test split
      2. Print R² diagnostics (unbiased — no imputed values involved)
      3. Impute missing rows using the fitted imputer
    """
    from xgboost import XGBRegressor

    targets = [c for c in cols if c in df.columns and df[c].isna().any()]
    print(f"\n[{name}] MICE XGBoost imputation on: {targets}")

    numeric = df.select_dtypes("number")
    complete_mask = numeric.notna().all(axis=1)
    complete_rows = numeric[complete_mask]
    missing_rows = numeric[~complete_mask]

    print(f"  Complete rows : {len(complete_rows)} | Rows with missing values: {len(missing_rows)}")

    # ── Step 1 : Split complete rows into train / test ────────
    train_df, test_df = train_test_split(complete_rows, test_size=test_size, random_state=random_state)

    # ── Step 2 : Fit MICE with XGBoost on train, evaluate on test ──────────
    xgb_estimator = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbosity=0,
    )
    imputer = IterativeImputer(estimator=xgb_estimator, random_state=random_state, max_iter=10)
    imputer.fit(train_df)

    train_imputed = pd.DataFrame(imputer.transform(train_df), columns=numeric.columns, index=train_df.index)
    test_imputed = pd.DataFrame(imputer.transform(test_df), columns=numeric.columns, index=test_df.index)

    print(f"\n  {'Column':<42} {'R² train':>10} {'R² test':>10} {'N train':>10} {'N test':>8}")
    print(f"  {'-'*80}")

    for col in targets:
        others = [c for c in targets if c != col]
        if not others:
            continue

        # Train XGBoost model
        X_tr, y_tr = train_imputed[others].values, train_imputed[col].values
        # Test — use ORIGINAL values (ground truth, no imputation involved)
        X_te, y_te = test_df[others].values, test_df[col].values

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=0,
        ).fit(X_tr, y_tr)

        r2_train = model.score(X_tr, y_tr)
        r2_test = model.score(X_te, y_te)

        print(f"  {col:<42} {r2_train:>10.4f} {r2_test:>10.4f} {len(X_tr):>10} {len(X_te):>8}")

    # ── Step 3 : Impute the actual missing rows ───────────────
    print(f"\n  Imputing {len(missing_rows)} rows with missing values...")
    full_imputed = pd.DataFrame(
        imputer.transform(numeric),
        columns=numeric.columns, index=numeric.index,
    )
    df[targets] = full_imputed[targets]
    print(f"  Done. Missing values remaining: {df[targets].isna().sum().sum()}")


# KNN for code_commune
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score

def impute_code_commune_knn(df: pd.DataFrame, name: str, n_neighbors: int = 5,
                             test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """
    KNN classification for code_commune (categorical French commune code).
    - Splits complete rows into train/test to evaluate accuracy & R²-equivalent
    - Imputes missing code_commune values using the model trained on complete rows
    Must be called AFTER all other features have been imputed.
    """
    feature_cols = [c for c in df.select_dtypes("number").columns if c != "code_commune"]

    missing_mask   = df["code_commune"].isna()
    complete_mask  = ~missing_mask & df[feature_cols].notna().all(axis=1)
    imputable_mask =  missing_mask  & df[feature_cols].notna().all(axis=1)

    n_missing     = missing_mask.sum()
    n_imputable   = imputable_mask.sum()
    n_unimputable = n_missing - n_imputable

    if n_missing == 0:
        print(f"[{name}] code_commune: no missing values – skipping.")
        return df

    print(f"[{name}] code_commune: {n_missing} missing | "
          f"{n_imputable} imputable | {n_unimputable} cannot impute (incomplete features)")

    # ── Train / test split on complete rows ──────────────────
    complete_df = df[complete_mask]
    X = complete_df[feature_cols].values
    y = complete_df["code_commune"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )

    # ── Fit KNeighborsClassifier (handles majority vote internally) ──
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="ball_tree")
    knn.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────
    acc_train = accuracy_score(y_train, knn.predict(X_train))
    acc_test  = accuracy_score(y_test,  knn.predict(X_test))

    # R²-equivalent for classification: 1 - (error_rate / baseline_error_rate)
    # Baseline = always predicting the most frequent class
    from collections import Counter
    most_common_freq = Counter(y_train).most_common(1)[0][1] / len(y_train)
    baseline_error   = 1 - most_common_freq
    model_error_test = 1 - acc_test
    r2_equiv = 1 - (model_error_test / baseline_error) if baseline_error > 0 else float("nan")

    print(f"\n[{name}] KNN classifier: code_commune ~ {len(feature_cols)} features")
    print(f"  n_samples (complete rows): {len(complete_df)}  |  train: {len(X_train)}  |  test: {len(X_test)}")
    print(f"  k = {n_neighbors}")
    print(f"  Accuracy  (train): {acc_train:.4f}  |  Accuracy  (test): {acc_test:.4f}")
    print(f"  R²-equiv  (train): {1 - (1 - acc_train) / baseline_error:.4f}  |  R²-equiv (test): {r2_equiv:.4f}")
    print(f"  Baseline accuracy (most frequent class): {most_common_freq:.4f}")

    # ── Impute missing rows ───────────────────────────────────
    if n_imputable > 0:
        query_X = df.loc[imputable_mask, feature_cols].values
        df.loc[imputable_mask, "code_commune"] = knn.predict(query_X)
        print(f"\n  Filled {n_imputable} missing code_commune values.")

    if n_unimputable > 0:
        print(f"  Warning: {n_unimputable} rows still missing code_commune "
              f"(incomplete feature vectors — drop or handle separately).")

    return df

def run_xgboost_regression(
    df: pd.DataFrame,
    name: str,
    y_col: str,
    x_cols: list[str],
    impute: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    **xgb_kwargs,
) -> pd.DataFrame:
    """
    Train XGBoost regression on complete rows; optionally impute missing y_col values.
    Returns the (possibly updated) DataFrame.
    """
    required = [y_col] + list(x_cols)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"\n[{name}] Skipping – missing columns: {missing_cols}")
        return df

    complete = df[required].notna().all(axis=1)
    data = df[complete]
    if data.empty:
        print(f"\n[{name}] No complete rows for '{y_col}' regression.")
        return df

    X, y = data[x_cols].values, data[y_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = xgb.XGBRegressor(random_state=random_state, **xgb_kwargs)
    model.fit(X_tr, y_tr)

    r2_tr = r2_score(y_tr, model.predict(X_tr).clip(0, 1))
    r2_te = r2_score(y_te, model.predict(X_te).clip(0, 1))

    print(f"\n[{name}] {y_col} ~ {', '.join(x_cols)}")
    print(f"  rows: {len(data)}  train: {len(X_tr)}  test: {len(X_te)}")
    print(f"  feature importances (gain):")
    for col, imp in zip(x_cols, model.feature_importances_):
        print(f"    {col}: {imp:.4f}")
    print(f"  R² train: {r2_tr:.4f}  |  R² test: {r2_te:.4f}")

    if impute:
        mask = df[y_col].isna() & df[x_cols].notna().all(axis=1)
        n = mask.sum()
        if n:
            df.loc[mask, y_col] = model.predict(df.loc[mask, x_cols].values).clip(0, 1)
            print(f"  Imputed {n} missing '{y_col}' values.")
        still = df[y_col].isna().sum()
        if still:
            print(f"  {still} rows still missing '{y_col}' (incomplete predictors).")

    return df

def compare_ML(df, col):
  df_complete = df.dropna(subset=[col])
  df_missing  = df[df[col].isna()]
  reg = setup(df_complete, target=col, session_id=42)
  best = compare_models()

from sklearn.ensemble import RandomForestRegressor

def run_rf_regression(
    df: pd.DataFrame,
    name: str,
    y_col: str,
    x_cols: list[str],
    impute: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    **rf_kwargs,
) -> pd.DataFrame:
    """
    Train Random Forest regression on complete rows; optionally impute missing y_col values.
    Returns the (possibly updated) DataFrame.
    """
    required = [y_col] + list(x_cols)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"\n[{name}] Skipping – missing columns: {missing_cols}")
        return df

    complete = df[required].notna().all(axis=1)
    data = df[complete]
    if data.empty:
        print(f"\n[{name}] No complete rows for '{y_col}' regression.")
        return df

    X, y = data[x_cols].values, data[y_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(random_state=random_state, **rf_kwargs)
    model.fit(X_tr, y_tr)

    r2_tr = r2_score(y_tr, model.predict(X_tr))
    r2_te = r2_score(y_te, model.predict(X_te))

    print(f"\n[{name}] {y_col} ~ {', '.join(x_cols)}")
    print(f"  rows: {len(data)}  train: {len(X_tr)}  test: {len(X_te)}")
    print(f"  feature importances (mean decrease impurity):")
    for col, imp in zip(x_cols, model.feature_importances_):
        print(f"    {col}: {imp:.4f}")
    print(f"  R² train: {r2_tr:.4f}  |  R² test: {r2_te:.4f}")

    if impute:
        mask = df[y_col].isna() & df[x_cols].notna().all(axis=1)
        n = mask.sum()
        if n:
            df.loc[mask, y_col] = model.predict(df.loc[mask, x_cols].values)
            print(f"  Imputed {n} missing '{y_col}' values.")
        still = df[y_col].isna().sum()
        if still:
            print(f"  {still} rows still missing '{y_col}' (incomplete predictors).")

    return df