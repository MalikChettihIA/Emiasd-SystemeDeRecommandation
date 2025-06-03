import pandas as pd


def read_df_metadata(df):
    colonnes_types = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            type_col = 'numeric'
        else:
            type_col = 'char'
        colonnes_types.append({'colonne': col, 'type': type_col})

    # On convertit la liste en DataFrame pour un affichage sous forme de tableau
    types_df = pd.DataFrame(colonnes_types)
    return types_df