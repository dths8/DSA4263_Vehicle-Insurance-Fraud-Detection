import numpy as np
import pandas as pd

def fillna_missing(data):
    """
    Custom helper function to handle specific missing value imputation
    for categorical columns, including filling 'authorities_contacted'
    with 'None', and other NaNs with the string 'Missing'.
    """
    if not isinstance(data, pd.DataFrame):
        if isinstance(data, pd.Series):
             data = data.to_frame()
        else:
             data = pd.DataFrame(data)
             
    df = data.copy()

    if 'authorities_contacted' in df.columns:
        df['authorities_contacted'] = df['authorities_contacted'].fillna('None')
        
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].isnull().any():
            df[col] = df[col].fillna('Missing')
            
    return df
