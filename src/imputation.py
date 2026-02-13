"""In case of using Pyzo instead of Cursor
import os
os.chdir("E:\master\s2\project\src")
"""
# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# Import data
df_dpe = pd.read_csv("./data/fait_dpe_traite.csv", sep = ";")
df_com = pd.read_csv("./data/communes_moyennes_traite.csv", sep = ";", decimal = ",")

# Clean up data by drop unuseful columns
selected_columns = ['numero_dpe', 
# 'etiquette_dpe', 
# 'etiquette_ges',
# 'type_installation_chauffage', 
# 'nombre_appartement',
# 'nombre_niveau_immeuble', 
# 'nombre_niveau_logement',
# 'surface_habitable_immeuble', 
# 'surface_habitable_logement',
# 'indicateur_confort_ete', 
# 'protection_solaire_exterieure',
# 'inertie_lourde', 
'isolation_toiture', 
# 'deperditions_enveloppe',
# 'deperditions_murs', 
# 'qualite_isolation_enveloppe',
# 'qualite_isolation_murs', 
# 'qualite_isolation_plancher_h',
# 'apport_solaire_saison_chauffe', 
# 'apport_solaire_saison_froide',
# 'type_energie', 
# 'consommation_kwh', 
# 'consommation_euro',
# 'type_ins_solaire_ecs', 
# 'presence_production_pv', 
# 'code_commune',
# 'annee_construction', 
# 'annee_etablissement_dpe',
# 'decennie_construction', 
# 'id_type_batiment',
# 'qualite_isolation_plancher_haut_comble_amenage',
# 'qualite_isolation_plancher_haut_comble_perdu',
# 'qualite_isolation_plancher_haut_toit_terrasse',
# 'qualite_isolation_menuiseries', 
# 'deperditions_ponts_thermiques',
# 'deperditions_planchers_hauts', 
# 'deperditions_planchers_bas',
# 'deperditions_portes', 
# 'deperditions_baies_vitrees',
# 'deperditions_renouvellement_air', 
# 'qualite_isolation_plancher_bas',
'etiquette_dpe_norm', 
'etiquette_ges_norm', 
'annee_construction_norm',
# 'apport_solaire_saison_froide_norm', 
'consommation_kwh_norm',
'deperditions_baies_vitrees_norm', 
'deperditions_enveloppe_norm',
'deperditions_murs_norm', 
'deperditions_planchers_bas_norm',
'deperditions_planchers_hauts_norm', 
'deperditions_portes_norm',
'surface_habitable_logement_norm', 
'type_appartement', 
'type_immeuble',
'type_maison', 
'qualite_isolation_enveloppe_norm',
'qualite_isolation_murs_norm']

df_dpe = df_dpe[selected_columns]
print(df_dpe.shape) #Verify column number 1048575

# Drop all rows wher building type 2 or type_immeuble == 1
df_dpe = df_dpe[df_dpe['type_immeuble'] == 0] 
print(df_dpe.shape) #Verify column number 1025455

# Create seperate DataFrame by building type
df_app = df_dpe[df_dpe['type_appartement'] == 1].copy()
df_maison = df_dpe[df_dpe['type_maison'] == 1].copy()

print(df_app.shape) #Verify column number 572 512 x 27
print(df_maison.shape) #Verify column number 452 943 x 27

# Drop 'isolation_toiture' in df_app because appartements have no toir
df_app = df_app.drop(columns = ["isolation_toiture", "type_appartement", "type_maison" , "type_immeuble", "numero_dpe"]) # After this line, df_app has 23 columns
# Drop building type in df_maison
df_maison = df_maison.drop(columns = ["type_appartement", "type_maison", "type_immeuble", "numero_dpe" ]) # After this line, df_maison has 24 columns
# Transform data type from float64 to number
df_app = df_app.apply(pd.to_numeric, errors = "coerce")
df_maison = df_maison.apply(pd.to_numeric, errors = "coerce")
print(df_app.dtypes)
print(df_maison.dtypes)

# Count the columns missing value rate for each dataframe
def missing_value_table (df, name):
    miss_val_counts = df.isna().sum()
    miss_val_rate = round((miss_val_counts/len(df))*100,2)
    print("Percentage of missing values per column of ", name)
    return(miss_val_rate)

print(missing_value_table(df_app, 'des appartements'))
print(missing_value_table(df_maison, 'des maisons'))

"""
# Spearman correlation heatmaps
numeric_app = df_app.select_dtypes(include="number")
numeric_maison = df_maison.select_dtypes(include="number")

print("\nComputing Spearman correlation for df_app (numeric columns only)...")
corr_app = numeric_app.corr(method="spearman")

plt.figure(figsize=(14, 12))  # Larger figure
sns.heatmap(corr_app, 
            annot=True,      # Show correlation values
            fmt='.2f',       # 2 decimal places
            cmap="coolwarm", 
            center=0,
            square=True,     # Square cells
            linewidths=0.5,  # Grid lines
            cbar_kws={'shrink': 0.8})
plt.title("Spearman correlation heatmap - df_app", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

print("\nComputing Spearman correlation for df_maison (numeric columns only)...")
corr_maison = numeric_maison.corr(method="spearman")
plt.figure(figsize=(10, 8))
sns.heatmap(corr_maison, annot=False, cmap="coolwarm", center=0)
plt.title("Spearman correlation heatmap - df_maison")
plt.tight_layout()
plt.show()
"""

# Imputation missing value=================================

# Imputation by re-encode
# Re-encode isolation_toiture in df_maison:
#  - original value 0 -> -1
#  - missing values  -> 0
#  - 1    -> kept as-is
iso = df_maison["isolation_toiture"]
is_na = iso.isna()
is_zero = iso == 0
df_maison.loc[is_zero, "isolation_toiture"] = -1
df_maison.loc[is_na, "isolation_toiture"] = 0
# check missing value rate of "isolation_toiture"
print(missing_value_table(df_maison))

# Imputation by linear regression
def run_linear_regression(df, name: str, x_cols) -> None:
    """Fit linear regression: Y=qualite_isolation_murs, X columns given in x_cols."""
    required_cols = ["qualite_isolation_murs_norm"] + list(x_cols)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"\n[{name}] Cannot run regression, missing columns: {missing}")
        return

    data = df[required_cols].dropna()
    if len(data) == 0:
        print(f"\n[{name}] No rows left after dropping NaNs for regression.")
        return

    X = data[x_cols].values
    y = data["qualite_isolation_murs_norm"].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    print(f"\n[{name}] Linear regression: qualite_isolation_murs ~ {', '.join(x_cols)}")
    print(f"  n_samples: {len(data)}")
    print(f"  intercept: {model.intercept_:.4f}")
    for col_name, coef in zip(x_cols, model.coef_):
        print(f"  coef {col_name}: {coef:.4f}")
    print(f"  R^2 (on training data): {r2:.4f}")


# Run regressions:
# - df_app: X = [qualite_isolation_enveloppe, consommation_kwh]
# - df_maison: X = [qualite_isolation_enveloppe, annee_construction, consommation_kwh]
run_linear_regression(
    df_app,
    "df_app",
    ["qualite_isolation_enveloppe_norm", "consommation_kwh_norm"],
)
run_linear_regression(
    df_maison,
    "df_maison",
    ["qualite_isolation_enveloppe_norm", "annee_construction_norm", "consommation_kwh_norm"],
)

"""
# ---- MICE imputation for deperditions* variables ----
deperditions_cols = [
    "deperditions_baies_vitrees",
    "deperditions_murs",
    "deperditions_planchers_bas",
    "deperditions_planchers_hauts",
    "deperditions_ponts_thermiques",
    "deperditions_portes",
    "deperditions_enveloppe"
]


def run_mice_imputation(df, name: str):
    print(f"\n[{name}] MICE imputation for deperditions columns")
    missing_any = [c for c in deperditions_cols if c in df.columns and df[c].isna().any()]
    if not missing_any:
        print(f"[{name}] No missing values in target deperditions columns; skipping MICE.")
        return

    # Work only with numeric columns for IterativeImputer
    numeric_df = df.select_dtypes(include="number").copy()

    # Keep only columns that actually exist
    target_cols = [c for c in deperditions_cols if c in numeric_df.columns]

    print(f"[{name}] Columns used in MICE (numeric): {list(numeric_df.columns)}")
    print(f"[{name}] Target columns: {target_cols}")

    imputer = IterativeImputer(random_state=0)
    imputed_values = imputer.fit_transform(numeric_df)
    numeric_imputed = pd.DataFrame(imputed_values, columns=numeric_df.columns, index=numeric_df.index)


    # Replace in original df
    df[target_cols] = numeric_imputed[target_cols]

    # Simple R^2 diagnostics: for each target column, predict it from the other
    # deperditions columns (on the fully imputed numeric data).
    for col in target_cols:
        others = [c for c in target_cols if c != col]
        if not others:
            continue
        X = numeric_imputed[others].values
        y = numeric_imputed[col].values
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        print(f"[{name}] R^2 for {col}: {r2:.4f}")


run_mice_imputation(df_app, "df_app")
run_mice_imputation(df_maison, "df_maison")

print("\nAppartment: --------------------------------------\n")
missing_counts = df_app.isna().sum()
print(missing_counts)

print("\nPercentage of missing values per column:")
missing_percent = (missing_counts / len(df_app)) * 100
print(missing_percent)

print("\nMaison: --------------------------------------\n")
missing_counts = df_maison.isna().sum()
print(missing_counts)

print("\nPercentage of missing values per column:")
missing_percent = (missing_counts / len(df_maison)) * 100
print(missing_percent)
"""