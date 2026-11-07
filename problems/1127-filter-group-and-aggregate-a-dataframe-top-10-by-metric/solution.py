import pandas as pd

def solution(df):
    df = df[df['status'] == 'completed']
    df = df.groupby('region', as_index=False)['amount'].sum()
    df = df.sort_values('amount', ascending=False).head(10)
    df.reset_index(drop=True)
    return df