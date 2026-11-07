import pandas as pd

def solution(df1, df2, df3):
    df12 = pd.merge(df1, df2, on='emp_id', how='inner')
    df123 = pd.merge(df12, df3, on='emp_id', how='left')
    return df123