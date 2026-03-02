import pandas as pd
from src.imputation import heat_map_corr

# -- Constants--

# Load datasets
df_app = pd.read_csv("./data/df_app.csv")
df_maison = pd.read_csv("./data/df_maison.csv")

# DPE/GES label mapping
dpe_map = {round(i/6, 8): l for i, l in enumerate('ABCDEFG')}

# Decode normalized numeric values into categorical labels for distribution analysis
for df in [df_maison, df_app]:
    df['dpe_label'] = df['etiquette_dpe_norm'].apply(lambda x: dpe_map.get(round(x, 8)))
    df['ges_label'] = df['etiquette_ges_norm'].apply(lambda x: dpe_map.get(round(x, 8)))

labels = list('ABCDEFG')

g = df_app.groupby('code_commune')
num_cols = df_app.select_dtypes('number').columns.difference(['etiquette_dpe_norm', 'etiquette_ges_norm'])

def agg_df(df, suffix):
    """
    Aggregates DPE metrics by commune.
    Calculates:
    -Total count of buildings by building type
    -Percentage distribution of DPE and GES labels (A through G)
    -Mean of all others numeric features
    """
    g = df.groupby('code_commune')
    num_cols = df.select_dtypes('number').columns.difference(['etiquette_dpe_norm', 'etiquette_ges_norm','code_commune'])
    # Count buildings per commune
    agg = g.size().rename(f'nb_{suffix}').reset_index()
    # Calculate means for numeric attributes
    agg = agg.merge(g[num_cols].mean().add_suffix(f'_mean_{suffix}').reset_index(), on='code_commune')
    # Calculate percentage distribution for DPE and GES labels
    for label_col, prefix in [('dpe_label', 'pct_dpe'), ('ges_label', 'pct_ges')]:
        counts = (g[label_col].value_counts(normalize=True)
                  .unstack(fill_value=0)
                  .reindex(columns=labels, fill_value=0) # Ensure all labels A-G are present
                  .add_prefix(f'{prefix}_').add_suffix(f'_{suffix}')
                  .reset_index())
        # Combine with the building number of each community
        agg = agg.merge(counts, on='code_commune')
    
    return agg
# Compute and merge House and Apartment aggregations using an outer join
df_agg = agg_df(df_maison, 'maison').merge(agg_df(df_app, 'app'), on='code_commune', how='outer')

# Calculate the weight of each building type within the commune
df_agg['pct_maison'] = df_agg['nb_maison'] / (df_agg['nb_maison'].fillna(0) + df_agg['nb_app'].fillna(0))
df_agg['pct_app']    = 1 - df_agg['pct_maison']
# Export the community-level dataset for analysing the imputation impact on distribution
df_agg.to_csv("./data/df_communes_before_imputation.csv", index = False)
# IMPUTATION:
# If a commune has 0 apartments, all apartment-related features (means and percentages) are NaN. 
# We fill with 0 because the "intensity" of apartment features in these areas is null.
df_agg = df_agg.fillna(0)
# Export the final community-level dataset for clustering
df_agg.to_csv("./data/df_communes.csv", index = False)

#Verify if 715 communities
import requests

def get_population_commune(code_insee):
    """
    Interroge l'API Géo de gouv.fr pour obtenir la population d'une commune.
    """
    # URL de l'API Géo (point d'entrée communes)
    url = f"https://geo.api.gouv.fr/communes/{int(code_insee)}"
    
    # Paramètres pour demander spécifiquement le champ population
    params = {
        "fields": "population,nom",
        "format": "json"
    }

    try:
        response = requests.get(url, params=params)
        
        # Vérifie si la requête a réussi (code 200)
        if response.status_code == 200:
            data = response.json()
            if data and 'population' in data:
                return {
                    "nom": data['nom'],
                    "population": data['population']
                }
            else:
                return "Donnée de population non disponible pour ce code."
        elif response.status_code == 404:
            return "Code commune introuvable."
        else:
            return f"Erreur API : {response.status_code}"
            
    except Exception as e:
        return f"Une erreur est survenue : {e}"

# EXECUTION
code_com_test = df_agg[df_agg['nb_app']==0]["code_commune"]
code = []
nom = []
population = []
row_not_found = []
for code_test in code_com_test:
    resultat = get_population_commune(code_test)
    if isinstance(resultat, dict):
        code.append(code_test)
        nom.append(resultat['nom'])
        population.append(resultat['population'])
    else:
        row_not_found.append(code_test)

df_missing = pd.DataFrame({'Code_commune': code, 'Nom': nom, 'Population': population})
(df_missing['Population'] <= 500).sum() #563 /699 80,6 % communities with less than 500 peoples
(df_missing['Population'] <= 600).sum()
(df_missing['Population'] <= 700).sum()
(df_missing['Population'] <= 1000).sum() #686/699 98,1%