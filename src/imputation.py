#─────────────────────── Import libraries ───────────────────────
import pandas as pd
import src.helpers as fdpe
# ─────────────────────── Constants ───────────────────────
data_dir = "./data/"
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
'code_commune',
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

# 'qualite_isolation_plancher_bas',
'etiquette_dpe_norm', 
'etiquette_ges_norm', 
# 'annee_construction_norm',
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

deperditions_cols = [
    "deperditions_baies_vitrees_norm",
    "deperditions_murs_norm",
    "deperditions_planchers_bas_norm",
    "deperditions_planchers_hauts_norm",
    #"deperditions_portes_norm",
    "deperditions_enveloppe_norm"
]
base_predictors = [
    'etiquette_dpe_norm', 'etiquette_ges_norm', 'consommation_kwh_norm',
    'deperditions_baies_vitrees_norm', 'deperditions_enveloppe_norm',
    'deperditions_murs_norm', 'deperditions_planchers_bas_norm',
    'deperditions_planchers_hauts_norm', 'deperditions_portes_norm',
    'qualite_isolation_enveloppe_norm', 'qualite_isolation_murs_norm',
]

# ─────────────────────── Load & clean ───────────────────────

df_dpe  = pd.read_csv(f"{data_dir}fait_dpe_traite.csv", sep=";")

df_dpe = df_dpe[selected_columns]
# Initial shape
print(df_dpe.shape) #(1048575, 18)
# After dropping type_immeuble=1
df_dpe = df_dpe[df_dpe["type_immeuble"] == 0]
print(df_dpe.shape) #(1025455, 18)

drop_shared = ["type_appartement", "type_maison", "type_immeuble", "numero_dpe"]

df_app    = (df_dpe[df_dpe["type_appartement"] == 1]
             .copy()
             .drop(columns=["isolation_toiture"] + drop_shared))
df_maison = (df_dpe[df_dpe["type_maison"] == 1]
             .copy()
             .drop(columns=drop_shared))

print("df_app shape:", df_app.shape) #(572512, 13)
print("df_maison shape:", df_maison.shape) #(452943, 14)

df_app    = df_app.apply(pd.to_numeric, errors="coerce")
df_maison = df_maison.apply(pd.to_numeric, errors="coerce")

print(fdpe.missing_value_table(df_app, "appartements", as_rate=False))
print(fdpe.missing_value_table(df_maison, "maisons", as_rate=False))

# ─────────────────────── Spearman heatmaps ───────────────────────

# fdpe.plot_spearman_heatmap(df_app, "Appartements")
# fdpe.plot_spearman_heatmap(df_maison, "Maisons")

# ─────────────────────── Imputation ───────────────────────

# Re-encode isolation_toiture: 0 → -1, NaN → 0
iso = df_maison["isolation_toiture"]
df_maison.loc[iso == 0, "isolation_toiture"] = -1
df_maison.loc[iso.isna(), "isolation_toiture"] = 0

# Wall insulation quality via linear regression

print(fdpe.missing_value_table(df_app, "appartements", as_rate=False))
print(fdpe.missing_value_table(df_maison, "maisons", as_rate=False))
# ____MICE for deperditions columns____
print("--Run MICE for deperditions_baies_vitrees_norm")
fdpe.run_mice_imputation_xgboost(df_app, deperditions_cols, "df_app")
"""   R²(deperditions_baies_vitrees_norm): 0.9027
  R²(deperditions_murs_norm): 0.9322
  R²(deperditions_planchers_bas_norm): 0.8712
  R²(deperditions_planchers_hauts_norm): 0.8156
  R²(deperditions_portes_norm): 0.6804
  R²(deperditions_enveloppe_norm): 0.9816 """
fdpe.run_mice_imputation_xgboost(df_maison,deperditions_cols, "df_maison")
"""   R²(deperditions_baies_vitrees_norm): 0.6144
  R²(deperditions_murs_norm): 0.8518
  R²(deperditions_planchers_bas_norm): 0.5129
  R²(deperditions_planchers_hauts_norm): 0.5461
  R²(deperditions_portes_norm): 0.1759
  R²(deperditions_enveloppe_norm): 0.9394 """
print("Imputing consommation_kwh_norm")
#___Energy consumption imputation____
df_app = fdpe.run_linear_regression(df_app, "df_app", "consommation_kwh_norm",[p for p in base_predictors if p != "consommation_kwh_norm"] + ["surface_habitable_logement_norm"],impute=True)
# R² train: 0.8532  |  R² test: 0.8537
df_maison = fdpe.run_linear_regression(df_maison, "df_maison", "consommation_kwh_norm",[p for p in base_predictors if p != "consommation_kwh_norm"] + ["surface_habitable_logement_norm", "isolation_toiture"], impute=True)
# R² train: 0.8628  |  R² test: 0.8591

qual_mur_predictors = [
    "qualite_isolation_enveloppe_norm", "etiquette_dpe_norm",
    "etiquette_ges_norm", "deperditions_murs_norm", "consommation_kwh_norm",
]
print("Imputing surface_habitable_immeuble....")
#df_app = fdpe.run_linear_regression(df_app, "df_app", "qualite_isolation_murs_norm", qual_mur_predictors, impute=True)
# R² train: 0.3695  |  R² test: 0.3657 (243/572512)
#df_maison = fdpe.run_linear_regression(df_maison, "df_maison", "qualite_isolation_murs_norm", qual_mur_predictors, impute=True)
# R² train: 0.7061  |  R² test: 0.7057
# Surface habitable imputation
df_app = fdpe.run_xgboost_regression(df_app, "df_app", "surface_habitable_logement_norm", base_predictors, impute=True)
# R² train: 0.7503  |  R² test: 0.7495
# R² train: 0.9155  |  R² test: 0.9115
df_maison = fdpe.run_xgboost_regression(df_maison, "df_maison", "surface_habitable_logement_norm", base_predictors + ["isolation_toiture"], impute=True)
# R² train: 0.8635  |  R² test: 0.8620
# R² train: 0.9069  |  R² test: 0.9009
print("imputing code_commune.....")
fdpe.run_rf_regression(df_app, "df_app", "code_commune", base_predictors)


""" Construction year – low R², no imputation
all_predictors_app    = base_predictors + ["surface_habitable_logement_norm"]
all_predictors_maison = base_predictors + ["surface_habitable_logement_norm", "isolation_toiture"]
fdpe.run_linear_regression(df_app,    "df_app",    "annee_construction_norm", all_predictors_app,    impute=False)
fdpe.run_linear_regression(df_maison, "df_maison", "annee_construction_norm", all_predictors_maison, impute=False)
 """
# ── Final missing-value check & export ───────────────────────
print(fdpe.missing_value_table(df_app, "appartement", as_rate=False))
print(fdpe.missing_value_table(df_maison, "maison", as_rate=False))

# df_app.to_csv(f"{data_dir}df_app.csv",    index=False)
# df_maison.to_csv(f"{data_dir}df_maison.csv", index=False)
# print("Saved df_app.csv and df_maison.csv.")


# ── Run LAST, once all other features are fully imputed ──────
# df_app    = impute_code_commune_knn(df_app,    "df_app",    n_neighbors=5)
# df_maison = impute_code_commune_knn(df_maison, "df_maison", n_neighbors=5)

# ─────────────────────── Spearman heatmaps after ───────────────────────
"""
fdpe.plot_spearman_heatmap(df_app, "Appartements après l'imputation")
fdpe.plot_spearman_heatmap(df_maison, "Maisons après l'imputation")
fdpe.plot_histogram(df_maison, 'isolation_toiture', 'Isolation toiture',7,5)
print(fdpe.missing_value_table(df_maison, "maisons", as_rate=True))
print(fdpe.missing_value_table(df_app, "appartements", as_rate=True))
fdpe.plot_histogram(df_app, 'surface_habitable_logement_norm', 'Surface habitable (appartement) - XGBoost',7,5,15)
fdpe.plot_histogram(df_app, 'deperditions_murs_norm', 'deperditions_murs_norm (appartement)',7,5,15)
fdpe.plot_histogram(df_app, 'deperditions_portes_norm', 'deperditions_portes_norm (appartement)',7,5,15)
"""
fdpe.compare_ML(df_app, "qualite_isolation_murs_norm")
