README: Description of columns in country_level_bcw.csv
======================================================

This file describes each column of output/country_level_bcw.csv.

1. country_name:
   - Name of the country or territory.
2. TERRITORY1:
   - Name of the main territory (often identical to country_name).
3. ISO:
   - ISO code of the country or territory (3-letter or custom code).
4. SOVEREIGN1:
   - Name of the sovereign country (if different from the territory).
5. Continent:
   - Continent.
6. Groups:
   - Economic group(s) (e.g., LDCs, SIDS, Developed economies, etc.).
7. Population:
   - Total population of the country/territory.
8. Area_EEZ_KM2:
   - Area of the Exclusive Economic Zone (EEZ) in km².
9. GDP:
   - Gross Domestic Product (GDP), total (in constant 2015 US dollars).
10. CO2_emissions_2023:
    - CO2 emissions in 2023 (in tonnes).
11. Debt (2015 US$):
    - Total debt (in 2015 US dollars).
12. saltmarshes_area_km2:
    - Area of saltmarshes (km²).
13. seagrasses_area_km2:
    - Area of seagrasses (km²).
14. mangroves_area_km2:
    - Area of mangroves (km²).
15. uptake_salt_mean:
    - Mean annual carbon uptake by saltmarshes (tC/year).
16. uptake_salt_se:
    - Standard error of saltmarshes uptake.
17. uptake_seag_mean:
    - Mean annual carbon uptake by seagrasses (tC/year).
18. uptake_seag_se:
    - Standard error of seagrasses uptake.
19. uptake_mang_mean:
    - Mean annual carbon uptake by mangroves (tC/year).
20. uptake_mang_se:
    - Standard error of mangroves uptake.
21. uptake_total_mean:
    - Total mean annual blue carbon uptake (tC/year) (sum of the 3 BCEs).
22. uptake_total_se:
    - Standard error of total uptake.
23. BCP Seq (tC):
    - Total amount of carbon sequestered by BCP (tC).
24. cBCW:
    - Monetary value of coastal blue carbon sequestration (2015 US$).
25. cBCW_se:
    - Standard error of cBCW.
26. oBCW:
    - Monetary value of oceanic blue carbon( BCP) sequestration (2015 US$).
27. oBCW_se:
    - Standard error of oBCW.
28. Total BCseq:
    - Total amount of blue carbon sequestered by BCEs and BCP (tC).
29. Total BCseq_se:
    - Standard error of total sequestration.
30. Total BCW:
    - Total (BCEs+BCP) monetary value of blue carbon, the BCW (2015 US$).
31. Total BCW_se:
    - Standard error of BCW.
32. Area_EEZ_KM2_per_capita:
    - EEZ area per capita (km²/person).
33. GDP_per_capita:
    - GDP per capita (2015 US$/person).
34. CO2_emissions_2023_per_capita:
    - CO2 emissions per capita (t/person).
35. Debt (2015 US$)_per_capita:
    - Debt per capita (2015 US$/person).
36. Total BCW_per_capita:
    - BCW per capita (2015 US$/person).

Notes:
- Columns *_mean and *_se correspond respectively to the mean and standard error of estimates from Monte Carlo simulations.
- Monetary values are calculated from carbon sequestration and the social cost of carbon (GSCC).
- BCE = Blue Carbon Ecosystems (saltmarshes, seagrasses, mangroves).
- BCW = Blue Carbon Wealth.
- Detailed calculations are in main.py and scripts in the src/ folder.
