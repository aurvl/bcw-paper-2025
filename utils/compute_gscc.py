# Create a function reading the coutry-level Social Cost of Carbon (SCC) data and deriving the global SCC
import pandas as pd

def gscc_computer(path: str) -> float:
    """
    This function computes the global Social Cost of Carbon (GSCC) by reading country-level SCC data
    from a CSV file located at the given path.
    """
    df = pd.read_csv(path)
    df_filtered = df[(df['dmgfuncpar'] == 'bootstrap') & 
                     (df['climate'] == 'uncertain') & 
                     (df['prtp'].isna()) & 
                     (df['eta'].isna()) & 
                     (df['dr'] == 3.) & 
                     (df['run'].isin(['bhm_richpoor_lr']))]
    
    group = df_filtered.groupby('ISO3')[['16.7%', '50%', '83.3%']].median()
    group['country'] = group.index
    group = group[['country', '16.7%', '50%', '83.3%']]
    group = group.reset_index(drop=True)
    group.rename(columns={'50%':'median'}, inplace=True)
    
    cscc = group[group['country'] != 'WLD']
    gscc = cscc['median'].sum()
    group.to_csv('data_source/gscc/country_level_gscc.csv', index=False)
    return gscc