from pathlib import Path

# set project root
ROOT_DIR = Path(__file__).parent.parent
# print(f"Project root set to: {ROOT_DIR}")

# set data & output dir
DATA_DIR = ROOT_DIR / "data_source"
SUMMARY_DIR = ROOT_DIR / "data_source" / "summary"
OUTPUT_DIR = ROOT_DIR / "output"

# set paths to data files
WLD_SHP = DATA_DIR / "shp" / "map" / "ne_110m_admin_0_countries.shp"
USA_SPLIT_CSV = DATA_DIR / "shp" / "map" / "usa_split.csv"
EEZ_PATH = DATA_DIR / "shp" / "data_EEZ_areas_by_zone.xlsx"
EEZ_SHP = DATA_DIR / "shp" / "eez" / "EEZ_land_union_v4_202410.shp"
SALTMARSHES_PATH = DATA_DIR / "shp" / "data_saltmarshes_areas_by_country.xlsx"
SEAGRASSES_PATH = DATA_DIR / "shp" / "data_seagrasses_areas_by_country.xlsx"
MANGROVES_PATH = DATA_DIR / "shp" / "data_mangroves_areas_by_country.xlsx"

CSCC_CSV = DATA_DIR / "gscc" / "cscc_db_v2.csv"
B_LIGHT = 20_000
SEED_LIGHT = 42

CMOL = 44 / 12

SELECT_COLS = ["UNION", "TERRITORY1", "ISO_TER1", "SOVEREIGN1"]
SALTMARSHES_AREA_COL = "saltmarshes_area_km2"
SEAGRASSES_AREA_COL = "seagrasses_area_km2"
MANGROVES_AREA_COL = "mangroves_area_km2"

JSON_PATH = DATA_DIR / "shp" / "sequestration_rates.json"
GROUP_PATH = DATA_DIR / "economy" / "country_classification.csv"
POP_PATH = DATA_DIR / "economy" / "population.xlsx"
GDP_PATH = DATA_DIR / "economy" / "gdp.xlsx"
CB_PATH = DATA_DIR / "economy" / "annual-co2-emissions-per-country.csv"
DEBT_PATH = DATA_DIR / "economy" / "TotalExternalDebt.csv"
BCP_PATH = DATA_DIR / "bcp" / "BCP_dta.csv"

# Dict for ISO completion and continent assignment, based on manual review of 
# the data and external sources (e.g., Wikipedia, CIA World Factbook). 
# This is used to fill in missing ISO codes and assign continents for 
# territories that are not standard countries or have special statuses. 
# The keys are the territory names as they appear in the dataset, and 
# the values are the corresponding ISO codes or continent names.
ISO_MAP1 = {
    # Substitutions for territories with ISO code (not NaN)
    "Bonaire": "BES",
    "Sint-Eustatius": "BQ-SE",
    "Saba": "BQ-SA",
    "Sapodilla Cayes": "BZ-TOL",
    "Kiribati": "KIR",
    "Tristan Da Cunha": "SH-TA",
    "Ascension": "SH-AC",
    "Saint Helena": "SHN",
    "Svalbard": "NO-21",
    "Jan Mayen": "NO-22",
    "Jarvis Island": "UM-86",
    "Johnston Atoll": "UM-67",
    "Howland and Baker islands": "UM-81",
    "Palmyra Atoll": "UM-95",
    "Wake Island": "UM-79",
    "Navassa Island": "UM-76",
    "Islas San Félix and San Ambrosio": "CL-V",
    # Territories without ISO code (NaN)
    "Abu musa, Greater and Lesser Tunb": "AE",
    "Alaska": "US-AK",
    "Alhucemas Islands": "ES-ALH",
    "Andaman and Nicobar": "IN-AN",
    "Antarctica": "ATA",
    "Azores": "PT-20",
    "Bajo Nuevo Bank": "CO-BN",
    "Canary Islands": "ES-CN",
    "Ceuta": "ES-CE",
    "Chafarinas Islands": "ES-CHAF",
    "Chagos Archipelago": "IO",
    "Clipperton Island": "CP",
    "Colombian Exclusive Economic Zone (Quitasueño Bank)": "CO-QS",
    "Doumeira Islands": "DJ",
    "Easter Island": "CL-EI",
    "Galapagos": "EC-W",
    "Glorioso Islands": "FR-GLO",
    "Hala'ib Triangle": "SD",
    "Hawaii": "US-HI",
    "Kuril Islands": "RU-SA",
    "Macquarie Island": "AU-TAS",
    "Madeira": "PT-30",
    "Matthew and Hunter Islands": "VU",
    "Melilla": "ES-ML",
    "Oecusse": "TL-OE",
    "Peñón de Vélez de la Gomera": "ES-PVG",
    "Perejil Island": "ES-PER",
    "Prince Edward Islands": "ZA-PE",
    "Senkaku Islands": "JP",
    "Serrana Bank": "CO-SR",
    "Serranilla Bank": "CO-SL",
    "Trinidade": "BR-ES",
    "Tromelin Island": "FR-TRO",
}

CONTINENT_MAP1 = {
    "Bonaire": "Caribbean",
    "Sint-Eustatius": "Caribbean",
    "Kiribati": "Caribbean",
    "Sapodilla Cayes": "Americas",
    "Gilbert Islands": "Oceania",
    "Saba": "Caribbean",
    "Line Group": "Oceania",
    "Phoenix Group": "Oceania",
    "Tristan Da Cunha": "Africa",
    "Ascension": "Africa",
    "Saint Helena": "Africa",
    "Svalbard": "Europe",
    "Jan Mayen": "Europe",
    "Jarvis Island": "Oceania",
    "Johnston Atoll": "Oceania",
    "Howland and Baker islands": "Oceania",
    "Palmyra Atoll": "Oceania",
    "Wake Island": "Oceania",
    "Navassa Island": "Caribbean",
    "Islas San Félix and San Ambrosio": "Americas",
    # continents for NaN
    "Abu musa, Greater and Lesser Tunb": "Asia",
    "Alaska": "Americas",
    "Alhucemas Islands": "Africa",
    "Andaman and Nicobar": "Asia",
    "Azores": "Europe",
    "Bajo Nuevo Bank": "Caribbean",
    "Canary Islands": "Africa",
    "Ceuta": "Africa",
    "Chafarinas Islands": "Africa",
    "Chagos Archipelago": "Africa",
    "Clipperton Island": "Americas",
    "Colombian Exclusive Economic Zone (Quitasueño Bank)": "Caribbean",
    "Doumeira Islands": "Africa",
    "Easter Island": "Oceania",
    "Galapagos": "Americas",
    "Glorioso Islands": "Africa",
    "Hala'ib Triangle": "Africa",
    "Hawaii": "Oceania",
    "Kuril Islands": "Asia",
    "Macquarie Island": "Oceania",
    "Madeira": "Europe",
    "Matthew and Hunter Islands": "Oceania",
    "Melilla": "Africa",
    "Oecusse": "Asia",
    "Peñón de Vélez de la Gomera": "Africa",
    "Perejil Island": "Africa",
    "Prince Edward Islands": "Oceania",
    "Senkaku Islands": "Asia",
    "Serrana Bank": "Caribbean",
    "Serranilla Bank": "Caribbean",
    "Trinidade": "Americas",
    "Tromelin Island": "Africa",
}

ISO_OVERRIDES1 = {
    "Bonaire": "BES",
    "Belize": "BLZ",
    "Chile": "CHL",
    "Saint Helena": "SHN",
    "Jan Mayen": "SJM",
}

# Dictionnary to complete ISO codes and continents for BCW Analysis
ISO_MAP2 = {
    # Remplacements pour les ISO dupliqués (ex : BES, SHN, KIR...)
    'Bonaire': 'BES',          # Bonaire, Sint-Eustatius et Saba ont des codes BQ selon l’ISO 3166‑2:contentReference[oaicite:3]{index=3}
    'Sint-Eustatius': 'BQ-SE',
    'Saba': 'BQ-SA',
    'Sapodilla Cayes': 'BZ-TOL', # Sapodilla Cayes (Belize) → district de Toledo:contentReference[oaicite:4]{index=4}
    'Kiribati': 'KIR',   # Groupes d’îles de Kiribati
    'Tristan Da Cunha': 'SH-TA', # Îles de Saint‑Hélène : SH‑TA, SH‑AC, SH‑HL:contentReference[oaicite:6]{index=6}
    'Ascension': 'SH-AC',
    'Saint Helena': 'SHN',
    'Svalbard': 'NO-21',         # Sous‑codes de la Norvège pour Svalbard et Jan Mayen:contentReference[oaicite:7]{index=7}
    'Jan Mayen': 'NO-22',
    'Jarvis Island': 'UM-86',    # Codes des îles mineures des États‑Unis:contentReference[oaicite:8]{index=8}
    'Johnston Atoll': 'UM-67',
    'Howland and Baker islands': 'UM-81/UM-84',
    'Palmyra Atoll': 'UM-95',
    'Wake Island': 'UM-79',
    'Navassa Island': 'UM-76',
    'Islas San Félix and San Ambrosio': 'CL-V',
    # Territoires sans code ISO initial (NaN)
    'Abu musa, Greater and Lesser Tunb': 'AE',
    'Alaska': 'US-AK',
    'Alhucemas Islands': 'ES-ALH',
    'Andaman and Nicobar': 'IN-AN',
    'Antarctica': 'ATA',
    'Azores': 'PT-20',
    'Bajo Nuevo Bank': 'CO-BN',
    'Canary Islands': 'ES-CN',
    'Ceuta': 'ES-CE',
    'Chafarinas Islands': 'ES-CHAF',
    'Chagos Archipelago': 'IO',
    'Clipperton Island': 'CP',
    'Colombian Exclusive Economic Zone (Quitasueño Bank)': 'CO-QS',
    'Doumeira Islands': 'DJ',
    'Easter Island': 'CL-EI',
    'Galapagos': 'EC-W',
    'Glorioso Islands': 'FR-GLO',
    "Hala'ib Triangle": 'SD',
    'Hawaii': 'US-HI',
    'Kuril Islands': 'RU-SA',
    'Macquarie Island': 'AU-TAS',
    'Madeira': 'PT-30',
    'Matthew and Hunter Islands': 'VU',
    'Melilla': 'ES-ML',
    'Oecusse': 'TL-OE',
    'Peñón de Vélez de la Gomera': 'ES-PVG',
    'Perejil Island': 'ES-PER',
    'Prince Edward Islands': 'ZA-PE',
    'Senkaku Islands': 'JP',
    'Serrana Bank': 'CO-SR',
    'Serranilla Bank': 'CO-SL',
    'Trinidade': 'BR-ES',
    'Tromelin Island': 'FR-TRO',
}

CONTINENT_MAP2 = {
    # précisions des continents (à adapter selon vos besoins)
    'Bonaire': 'Caribbean', 'Sint-Eustatius': 'Caribbean', 'Kiribati': 'Caribbean',
    'Sapodilla Cayes': 'Americas', 'Gilbert Islands': 'Oceania',
    'Line Group': 'Oceania', 'Phoenix Group': 'Oceania',
    'Tristan Da Cunha': 'Africa', 'Ascension': 'Africa', 'Saint Helena': 'Africa',
    'Svalbard': 'Europe', 'Jan Mayen': 'Europe',
    'Jarvis Island': 'Oceania', 'Johnston Atoll': 'Oceania',
    'Howland and Baker islands': 'Oceania', 'Palmyra Atoll': 'Oceania',
    'Wake Island': 'Oceania', 'Navassa Island': 'Caribbean',
    'Islas San Félix and San Ambrosio': 'Americas',
    # continents pour les NaN
    'Abu musa, Greater and Lesser Tunb': 'Asia', 'Alaska': 'Americas',
    'Alhucemas Islands': 'Africa', 'Andaman and Nicobar': 'Asia', 'Azores': 'Europe',
    'Bajo Nuevo Bank': 'Caribbean', 'Canary Islands': 'Africa',
    'Ceuta': 'Africa', 'Chafarinas Islands': 'Africa', 'Chagos Archipelago': 'Africa',
    'Clipperton Island': 'Americas', 'Colombian Exclusive Economic Zone (Quitasueño Bank)': 'Caribbean',
    'Doumeira Islands': 'Africa', 'Easter Island': 'Oceania',
    'Galapagos': 'Americas', 'Glorioso Islands': 'Africa',
    "Hala'ib Triangle": 'Africa', 'Hawaii': 'Oceania', 'Kuril Islands': 'Asia',
    'Macquarie Island': 'Oceania', 'Madeira': 'Europe',
    'Matthew and Hunter Islands': 'Oceania', 'Melilla': 'Africa',
    'Oecusse': 'Asia', 'Peñón de Vélez de la Gomera': 'Africa',
    'Perejil Island': 'Africa', 'Prince Edward Islands': 'Oceania',
    'Senkaku Islands': 'Asia', 'Serrana Bank': 'Caribbean', 'Serranilla Bank': 'Caribbean',
    'Trinidade': 'Americas', 'Tromelin Island': 'Africa'
}

ISO_OVERRIDES2 = {}

