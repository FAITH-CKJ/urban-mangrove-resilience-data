Source Data for Supplementary Fig. 1c and Table 12

buffer_annual.csv: 858 rows = 3 city domains x 11 regional/pooled groups x 26 years.
buffer_comparison_table.csv: endpoint values, reference support, annual-trajectory
correlation, direction comparison and observed counts for every domain and region.
buffer_component_endpoint_UMRI.csv: the five exact contributions to each
2000-2025 UMRI change, expressed in UMRI.
methods.json: fixed definitions and the single varying domain parameter.

buffer_km is the city-domain buffer, including the city polygon interior. The
ecological neighbourhood remains 500 m. region_id=0 pools all ten regions;
region names are included in each file. centres is the fixed reference population;
valid_centres is the observed valid population in the stated year. The known
reference is GMW v4.1.12 presence in at least one year during 2000-2004.
Each centre has equal 900-m2 area. reference_area_km2 refers to distinct centres,
not the sum of overlapping 500-m ecological windows.

S_sum and each component_sum aggregate valid centre-level weighted components.
S (also S_mean) is S_sum/valid_centres. UMRI=S/(lambda+abs(S)), with fixed
lambda=0.05934161890467785. P/N/A retain the gross positive,
adverse and total magnitudes of the existing reporting definition. Component
mean fields are weighted intermediate values in S units, before normalization.
The endpoint contribution uses the common secant of the fixed normalization,
so the five values sum exactly to the stated endpoint change in UMRI.

The 2-km record reproduces the primary historical series. The 1-km domain is its
spatial subset; 5 km includes 625,120 additional reference centres. Each estimate
uses its annual validity mask. Missing observations retain their meaning and
are excluded from the valid sum/count rather than assigned an ecological zero.

Overall UMRI declines under all three domains. Regional endpoint directions
agree in 19 of the 20 alternative-domain comparisons with 2 km. Eastern Asia
increases by 0.0105 UMRI at 5 km, compared with changes of -0.0793 and -0.0763 UMRI at 1 and 2 km. All values, including this regional difference, are retained.
From the repository root, run:
python SupplementarySoftware/code/render_supplementary_b.py --figures S1
This command verifies the values and regenerates Supplementary Fig. 1, including
the city-domain comparison in panel c.
