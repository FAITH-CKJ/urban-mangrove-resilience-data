# Urbanization erodes resilience in global mangroves

Complete numerical source data and reproduction software for the study.

The author-generated data are shared under **CC BY 4.0** to support verification,
reuse and further research. This public repository can be accessed and downloaded
without registration. The current release is **v1.2.4**.

## Download

- [Download the complete v1.2.4 data and software](https://github.com/FAITH-CKJ/urban-mangrove-resilience-data/archive/refs/tags/v1.2.4.zip)
- [Browse SourceData](SourceData/README.md): individual observations, full
  projection-member tables, annual values, component contributions, construction
  accounts, model evaluation, analytical fields and spatial inputs.
- [Browse SupplementarySoftware](SupplementarySoftware/README.txt): source code,
  fitted models, dependencies and a real-data forecasting demonstration.

`SourceData/FIGURE_TABLE_INDEX.csv` maps Figs. 2–6, Supplementary Figs. 1–6 and
Supplementary Tables 1–12 to their supporting files. Figure 1 is conceptual.
`FIELD_SCHEMA.csv` and `FIELD_DEFINITIONS.txt` explain variables, units,
observation populations and missing values. Each directory has a SHA-256 manifest.

The journal Source Data attachment contains the values needed to interpret the
displayed summaries, including individual distribution/scatter values and annual
time points. Its index links to this repository for the complete numerical
records, larger spatial inputs and analytical grids.

## Reproduction

Keep `SourceData/` and `SupplementarySoftware/` side by side. With Python 3.11,
follow the environment instructions in the software README, then run from
`SupplementarySoftware/`:

```text
python -m pip install -r requirements.txt -r requirements-model.txt
python code/verify_reporting.py
python code/verify_figure_story.py
python code/reproduce_forecast_demo.py
python code/reproduce_main_figures.py
python code/reproduce_supplementary.py
```

The figure/reporting commands use the complete `SourceData/` directory.
Outputs are written to `SupplementarySoftware/reproduced/`. The forecast example
contains 16 real reference centres, three urban-development branches and 75 annual
predictions per branch. The software README and model card specify its inputs,
expected outputs, numerical tolerance, training design and evaluation.

Full geospatial reconstruction uses the original satellite and forcing products
listed in `SourceData/THIRD_PARTY_SOURCES.txt` and the paper's Methods. Those
provider archives retain their own terms. Numerical observations and derived
values distributed here preserve original precision and distinguish true zeros
from missing or unknown states.

## Citation and terms

See `SourceData/CITATION.cff` for the dataset citation and
`SourceData/LICENSE.txt` for the author-generated data licence. The coastal
land-cover software carries its supplied MIT licence; third-party libraries
retain their own licences. Source code, model definitions and fitted parameters
are provided with reproduction instructions in `SupplementarySoftware/`.
