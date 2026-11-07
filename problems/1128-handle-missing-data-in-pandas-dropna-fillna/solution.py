import pandas as pd

def solution(df):
    df_copy = df.copy()
    keep_cols = df_copy.columns[df_copy.isna().mean(axis=0) <= 0.5]
    df_copy = df_copy[keep_cols]
    keep_rows = df_copy.isna().mean(axis=1) <= 0.5
    df_copy = df_copy[keep_rows]

    for c in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[c]):
            df_copy[c] = df_copy[c].fillna(df_copy[c].mean())
        else:
            df_copy[c] = df_copy[c].fillna(df_copy[c].mode().iloc[0])

    return df_copy