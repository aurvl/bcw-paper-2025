# This script add economic social and environmental factors to the rest of data
import pandas as pd

def _add_groups(group_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding informations on continents and development level categories to the dataset
    """
    group = pd.read_csv(group_path).drop(columns='World')
    group = group.dropna(subset='ISO')
    merged = df.merge(
        group, how='left', left_on='ISO_TER1', right_on='ISO'
    ).drop(columns=['ISO'])
    
    return merged

def _add_population(pop_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding informations on populaitons to the dataset
    """
    # import and processing the population data
    pop = pd.read_excel(pop_path)
    pop = pop.rename(columns={
        'ISO3 Alpha-code': 'ISO_TER1',
        'Total Population, as of 1 July (thousands)': 'pop_thousands'
    })[['ISO_TER1', 'pop_thousands']]
    pop = pop.dropna(subset='ISO_TER1')

    merged = df.merge(pop, on='ISO_TER1', how='left')
    same_territory = merged['UNION'] == merged['TERRITORY1'] # ensuring the pop correspond to the main territory
    merged['Population'] = merged['pop_thousands'] * 1000
    merged.loc[~same_territory, 'Population'] = pd.NA
    merged = merged.drop(columns=['pop_thousands'])
    
    return merged

def _add_gdp(gdp_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding informations on GDP to the dataset
    """
    gdp = pd.read_excel(gdp_path, sheet_name='GDP')
    gdp = gdp[['Country Code', 'GDP (constant 2015 US$)']] # we just use GDP of the last year for each country
    gdp = gdp.rename(columns={
        'Country Code': 'ISO_TER1',
        'GDP (constant 2015 US$)': 'GDP'
    }).dropna(subset='ISO_TER1')
    
    merged = df.merge(gdp, on='ISO_TER1', how='left')
    same_territory = merged['UNION'] == merged['TERRITORY1']
    merged.loc[~same_territory, 'GDP'] = pd.NA

    return merged

def _add_carbon_emissions(cb_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding informations on CO2 emissions to the dataset
    """
    cb = pd.read_csv(cb_path)
    cb = cb[cb['Year'] == 2023][['Code', 'Annual CO₂ emissions']]
    cb = cb.rename(columns={
        'Code': 'ISO_TER1',
        'Annual CO₂ emissions': 'CO2_emissions_2023'
    }).dropna(subset='ISO_TER1')
    
    merged = df.merge(cb, on='ISO_TER1', how='left')
    same_territory = merged['UNION'] == merged['TERRITORY1']
    merged.loc[~same_territory, 'CO2_emissions_2023'] = pd.NA

    return merged

def _add_debt(debt_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding informations on Total external debt of 2023 (in 2015 US$) to the dataset
    """
    db = pd.read_csv(debt_path)
    db.columns = ['ISO_TER1', 'Debt (2015 US$)']
    db = db.dropna(subset='ISO_TER1')

    merged = df.merge(db, on='ISO_TER1', how='left')
    same_territory = merged['UNION'] == merged['TERRITORY1']
    merged.loc[~same_territory, 'Debt (2015 US$)'] = pd.NA
    
    return merged

def reorganize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['UNION', 'TERRITORY1', 'ISO_TER1', 'SOVEREIGN1', 'Continent', 'Groups', 'Population',
             'Area_EEZ_KM2', 'GDP', 'CO2_emissions_2023', 'Debt (2015 US$)', 'saltmarshes_area_km2', 
             'seagrasses_area_km2', 'mangroves_area_km2']].copy()
    return df

def add_eco_data(df: pd.DataFrame,
                  group_path: str,
                  pop_path: str,
                  gdp_path: str,
                  cb_path: str,
                  debt_path: str) -> pd.DataFrame:
    """
    This function adds economic, social and environmental data to the BCE areas dataframe.
    - df : DataFrame containing BCE areas by EEZs
    - group_path : Path to the continent and development level categories data file
    - pop_path : Path to the population data file
    - gdp_path : Path to the GDP data file
    - cb_path : Path to the CO2 emissions data file
    - debt_path : Path to the Total external debt data file
    """
    df = _add_groups(group_path, df)
    df = _add_population(pop_path, df)
    df = _add_gdp(gdp_path, df)
    df = _add_carbon_emissions(cb_path, df)
    df = _add_debt(debt_path, df)
    df = reorganize_df(df)

    return df