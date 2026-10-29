




from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.source_data.resolve()
output = args.output.resolve()
figdir, qadir = output / "figures", output / "qa"
figdir.mkdir(parents=True, exist_ok=True)
qadir.mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 171.0, 118.0
INK, MID, GRID = "#25393C", "#677977", "#E4EAE7"
PATHWAYS = ["ssp126", "ssp245", "ssp370", "ssp585"]
COLORS = dict(zip(PATHWAYS, ["#3C7C9B", "#B38B35", "#9274A5", "#B55755"]))
MARKERS = dict(zip(PATHWAYS, ["o", "D", "^", "s"]))
LABELS = dict(zip(PATHWAYS, ["SSP1–2.6", "SSP2–4.5", "SSP3–7.0", "SSP5–8.5"]))
STYLES = {"ssp126": "-", "ssp245": (0, (4, 1.4)), "ssp370": (0, (1, 1.2)), "ssp585": "-"}
REGIONS = [
    "Australia &\nNew Zealand", "Central\nAmerica", "Eastern\nAfrica",
    "Eastern\nAsia", "Northern\nAmerica", "South\nAmerica",
    "Southeastern\nAsia", "Southern\nAsia", "Western\nAfrica", "Western\nAsia"]
LIMITS = {1: (-.35, .006), 2: (-.002, .000035), 3: (-1, .018),
          4: (-.2, .003), 5: (-.3, .005), 6: (-.4, .006),
          7: (-.25, .004), 8: (-.004, .00006),
          9: (-.07, .0012), 10: (-.06, .001)}
TICKS = {1: [-.3, 0], 2: [-.002, 0], 3: [-1, -.5, 0],
         4: [-.2, 0], 5: [-.3, 0], 6: [-.4, 0],
         7: [-.2, 0], 8: [-.004, 0], 9: [-.06, 0], 10: [-.06, 0]}
mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7.4, "axes.labelsize": 7.4,
    "axes.titlesize": 7.6, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 7.2, "axes.linewidth": .55,
    "xtick.major.width": .5, "ytick.major.width": .5,
    "xtick.major.size": 2.1, "ytick.major.size": 2.1,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MID,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
    "path.simplify": False, "mathtext.fontset": "dejavusans"})
inputs = {}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(relative):
    path = source / relative
    inputs[relative] = sha(path)
    return pd.read_csv(path, float_precision="round_trip")
def ax(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT])
def text(fig, x, y, value, **kwargs):
    return fig.text(x / WIDTH, y / HEIGHT, value, **kwargs)
def panel(fig, x, y, letter, title):
    text(fig, x, y, letter, weight="bold", fontsize=8.4, va="baseline")
    text(fig, x + 4.5, y, title, fontsize=7.6, va="baseline")
def grid(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID, lw=.42)
def valid_series(frame):
    assert len(frame) == 75 and set(frame.year) == set(range(2026, 2101))
    assert frame.members.eq(15).all()
    values = frame[["delta_UMRI_q17", "delta_UMRI_q50", "delta_UMRI_q83"]].to_numpy()
    assert np.isfinite(values).all()
    assert np.all(values[:, 0] <= values[:, 1]) and np.all(values[:, 1] <= values[:, 2])

annual_all = read("main_figures/Fig4_urban_paired_annual.csv")
protection_all = read("main_figures/Fig5_protection_paired_annual.csv")
assert annual_all.scheme.eq("pooled_centres").all()
assert protection_all.scheme.eq("pooled_centres").all()
assert annual_all.support.eq("forecast").all() and protection_all.support.eq("forecast").all()
assert annual_all.contrast.eq("urban_growth_minus_fixed").all()
assert protection_all.contrast.eq("protection_minus_growth").all()
annual = annual_all[annual_all.region_id.gt(0)].copy()
protection = protection_all[protection_all.region_id.eq(0)].copy()
assert len(annual) == 3000 and len(protection) == 150
assert not annual[["scenario", "region_id", "year"]].duplicated().any()
assert not protection[["scenario", "year"]].duplicated().any()
assert set(annual.region_id) == set(range(1, 11))
assert set(annual.scenario) == set(PATHWAYS)
assert set(protection.scenario) == {"ssp245", "ssp585"}



member_rel = "reporting/primary_paired_member_annual.parquet"
inputs[member_rel] = sha(source / member_rel)
members_all = pd.read_parquet(source / member_rel)
members = members_all[
    members_all.region_id.gt(0) & members_all.scheme.eq("pooled_centres")
    & members_all.support.eq("forecast")
    & members_all.contrast.eq("urban_growth_minus_fixed")].copy()
keys = ["region_id", "scenario", "year"]
member_fields = ["delta_S", "delta_UMRI", "direct_urban_delta_S", "ecological_delta_S"]
assert len(members) == 45000
assert not members[keys + ["member_id"]].duplicated().any()
assert np.isfinite(members[member_fields].to_numpy()).all()
counts = members.groupby(keys).size()
assert len(counts) == 3000 and counts.eq(15).all()
summary = annual.set_index(keys).sort_index()
quantiles = members.groupby(keys).delta_UMRI.quantile([.17, .5, .83]).unstack().sort_index()
assert summary.index.equals(quantiles.index)
quantile_error = float(np.max(np.abs(
    quantiles.to_numpy() - summary[["delta_UMRI_q17", "delta_UMRI_q50", "delta_UMRI_q83"]].to_numpy())))
assert quantile_error < 2e-15
negative = members.assign(negative=members.delta_UMRI.lt(0)).groupby(keys).negative.sum().sort_index()
positive = members.assign(positive=members.delta_UMRI.gt(0)).groupby(keys).positive.sum().sort_index()
assert np.array_equal(negative.to_numpy(), summary.delta_UMRI_negative_members.to_numpy())
assert np.array_equal(positive.to_numpy(), summary.delta_UMRI_positive_members.to_numpy())
omitted = set()
zero_records = []
for (region, pathway), part in members.groupby(["region_id", "scenario"]):
    assert len(part) == 1125 and part.year.nunique() == 75 and part.member_id.nunique() == 15
    if part.delta_UMRI.eq(0).all():
        assert part[member_fields].eq(0).to_numpy().all()
        shown = annual[annual.region_id.eq(region) & annual.scenario.eq(pathway)]
        assert shown[["delta_UMRI_q17", "delta_UMRI_q50", "delta_UMRI_q83",
                      "delta_UMRI_mean", "delta_UMRI_negative_members",
                      "delta_UMRI_positive_members"]].eq(0).to_numpy().all()
        omitted.add((int(region), pathway))
        zero_records.append({"region_id": int(region), "region": REGIONS[int(region)-1].replace("\n"," "),
                             "scenario": pathway, "years": 75, "member_years": 1125,
                             "all_members_and_summary_fields_exact_zero": True})
assert omitted == {(2, "ssp585"), (10, "ssp245"), (10, "ssp370")}
absolute_rel = "reporting/primary_member_annual.parquet"
inputs[absolute_rel] = sha(source / absolute_rel)
absolute = pd.read_parquet(source / absolute_rel)
absolute = absolute[absolute.scheme.eq("pooled_centres") & absolute.support.eq("forecast")]
absolute_fields = ["S", "UMRI", "d_contribution", "r_contribution", "g_contribution",
                   "h_contribution", "T_contribution", "absence_fraction"]
for region, pathway in sorted(omitted):
    part = absolute[absolute.region_id.eq(region) & absolute.scenario.eq(pathway)]
    growth = part[part.urban_branch.eq("growth")].set_index(["member_id", "year"])[absolute_fields].sort_index()
    fixed = part[part.urban_branch.eq("fixed2025")].set_index(["member_id", "year"])[absolute_fields].sort_index()
    assert growth.shape == fixed.shape == (1125, 8) and growth.index.equals(fixed.index)
    assert np.isfinite(growth.to_numpy()).all() and np.isfinite(fixed.to_numpy()).all()
    assert np.array_equal(growth.to_numpy(), fixed.to_numpy())
west = members[members.region_id.eq(10)].copy()
west_records = []
for pathway, part in west.groupby("scenario"):
    changed = part[part.delta_UMRI.ne(0)]
    end = annual[annual.region_id.eq(10) & annual.scenario.eq(pathway) & annual.year.eq(2100)].iloc[0]
    west_records.append({
        "scenario": pathway, "member_years": len(part),
        "members": part.member_id.nunique(), "years": part.year.nunique(),
        "first_nonzero_year": int(changed.year.min()) if len(changed) else None,
        "nonzero_member_years": len(changed),
        "minimum_member_effect_UMRI": float(part.delta_UMRI.min()),
        "maximum_member_effect_UMRI": float(part.delta_UMRI.max()),
        "endpoint_q17": float(end.delta_UMRI_q17),
        "endpoint_median": float(end.delta_UMRI_q50),
        "endpoint_q83": float(end.delta_UMRI_q83),
        "omitted_from_display": (10, pathway) in omitted
    })
allocation = read("main_figures/Fig5_allocation_regional_annual.csv")
allocation_west = allocation[allocation.region_id.eq(10)].copy()
allocation_records = []
for pathway, part in allocation_west.groupby("scenario"):
    changed = part[part.primary_realized_growth_km2.ne(0)]
    end = part[part.year.eq(2100)].iloc[0]
    allocation_records.append({
        "scenario": pathway, "years": len(part),
        "first_nonzero_growth_year": int(changed.year.min()) if len(changed) else None,
        "endpoint_primary_growth_km2": float(end.primary_realized_growth_km2),
        "endpoint_in_cell_protected_growth_km2": float(end.in_cell_protected_growth_km2),
        "endpoint_relocated_growth_km2": float(end.relocated_growth_km2),
        "endpoint_unmet_growth_km2": float(end.additional_unmet_growth_km2)
    })
zero_audit = {
    "status": "PASS",
    "all_regional_member_years_checked": 45000,
    "western_asia_member_years_checked": 4500,
    "quantiles_recomputed_max_absolute_error": quantile_error,
    "positive_and_negative_member_counts_match": True,
    "omitted_series": zero_records,
    "absolute_growth_and_fixed_branches_identical_for_omitted_series": {
        "fields": absolute_fields, "rows_per_series": 1125, "comparison": "Exact equality, without tolerance"},
    "western_asia": west_records,
    "western_asia_construction": allocation_records,
    "interpretation": "Omitted series have exact zero regional paired UMRI effects. Western Asia SSP2 has a zero paired UMRI effect alongside 0.0099 km2 of additional construction in 2100.",
    "inputs": {rel: inputs[rel] for rel in [member_rel, absolute_rel, "main_figures/Fig4_urban_paired_annual.csv", "main_figures/Fig5_allocation_regional_annual.csv"]}
}
(qadir / "exact_zero_series_audit.json").write_text(
    json.dumps(zero_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
west.to_csv(qadir / "western_asia_paired_member_annual.csv", index=False, float_format="%.17g")
allocation_west.to_csv(qadir / "western_asia_allocation_annual.csv", index=False, float_format="%.17g")

fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4))
panel(fig, 3, 114.5, "a", "Urban expansion")
handles = [Line2D([0], [0], color=COLORS[p], ls=STYLES[p],
                  marker=MARKERS[p], ms=3, lw=.95,
                  mec="white", mew=.3, label=LABELS[p]) for p in PATHWAYS]
fig.legend(handles=handles, loc="center", bbox_to_anchor=(.55, 111 / HEIGHT),
           ncol=4, handlelength=1.9, columnspacing=1.8, handletextpad=.45)
lefts = [14, 46.75, 79.5, 112.25, 145]
bottoms = [77, 44]
display_rows = []
for region in range(1, 11):
    row, col = divmod(region - 1, 5)
    axis = ax(fig, lefts[col], bottoms[row], 23, 24)
    text(fig, lefts[col] + 11.5, bottoms[row] + 26, REGIONS[region - 1],
         ha="center", va="bottom", fontsize=7.5, linespacing=1.0)
    for pathway in PATHWAYS:
        part = annual[annual.region_id.eq(region) & annual.scenario.eq(pathway)].sort_values("year")
        valid_series(part)
        assert part.delta_UMRI_q17.min() > LIMITS[region][0]
        assert part.delta_UMRI_q83.max() < LIMITS[region][1]
        if (region, pathway) not in omitted:
            axis.fill_between(part.year, part.delta_UMRI_q17, part.delta_UMRI_q83,
                              color=COLORS[pathway], alpha=.16, lw=0, zorder=1)
            axis.plot(part.year, part.delta_UMRI_q50, color=COLORS[pathway],
                      ls=STYLES[pathway], lw=.95, zorder=3)
            axis.plot([2100], [part.delta_UMRI_q50.iloc[-1]], ls="none",
                      marker=MARKERS[pathway], color=COLORS[pathway], ms=2.7,
                      mec="white", mew=.25, zorder=4, clip_on=False)
        display_rows.extend(
            {"region_id": region, "scenario": pathway, "year": int(year),
             "q17": float(q17), "median": float(median), "q83": float(q83),
             "displayed": (region, pathway) not in omitted}
            for year, q17, median, q83 in zip(part.year, part.delta_UMRI_q17,
                                            part.delta_UMRI_q50, part.delta_UMRI_q83))
    axis.axhline(0, color=MID, lw=.4, zorder=2)
    axis.set(xlim=(2026, 2102), ylim=LIMITS[region],
             xticks=[2030, 2100], yticks=TICKS[region])
    if row == 0:
        axis.tick_params(axis="x", labelbottom=False)
    axis.tick_params(labelsize=7.0, pad=2)
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, pos: ("0" if value == 0 else f"{value:g}").replace("-", "−")))
    grid(axis)
text(fig, 2.8, 79, r"$\Delta\mathrm{UMRI}$", rotation=90,
     ha="center", va="center", fontsize=7.4)
text(fig, 91, 37.5, "Year", ha="center", va="center", fontsize=7.4)

panel(fig, 3, 32.5, "b", "Overall protection")
axis = ax(fig, 15, 9.5, 152, 19)
for pathway in ["ssp245", "ssp585"]:
    part = protection[protection.scenario.eq(pathway)].sort_values("year")
    valid_series(part)
    assert part.delta_UMRI_q17.min() > -.005 and part.delta_UMRI_q83.max() < .225
    axis.fill_between(part.year, part.delta_UMRI_q17, part.delta_UMRI_q83,
                      color=COLORS[pathway], alpha=.17, lw=0, zorder=1)
    axis.plot(part.year, part.delta_UMRI_q50, color=COLORS[pathway],
              ls=STYLES[pathway], lw=1.2, zorder=3)
    axis.plot([2100], [part.delta_UMRI_q50.iloc[-1]], ls="none",
              marker=MARKERS[pathway], color=COLORS[pathway], ms=3.0,
              mec="white", mew=.25, zorder=4, clip_on=False)
axis.axhline(0, color=MID, lw=.45, zorder=2)
axis.set(xlim=(2026, 2100.8), ylim=(-.005, .225),
         xticks=[2030, 2050, 2070, 2100], yticks=[0, .1, .2],
         ylabel=r"$\Delta\mathrm{UMRI}$", xlabel="Year")
grid(axis)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
canvas_width, canvas_height = fig.canvas.get_width_height()
outside = []
for item in fig.findobj(mpl.text.Text):
    if item.get_visible() and item.get_text().strip():
        box = item.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > canvas_width + 1 or box.y1 > canvas_height + 1:
            outside.append({"text": item.get_text(), "bbox": list(box.bounds)})
if outside:
    raise RuntimeError(("text outside figure", outside))
stem = "SupplementaryFig2_regional_scenario_trajectories"
for extension in ["pdf", "svg", "png"]:
    fig.savefig(figdir / f"{stem}.{extension}", dpi=300)
fig.savefig(figdir / f"{stem}_600dpi.png", dpi=600)
with Image.open(figdir / f"{stem}_600dpi.png") as raster:
    raster.convert("RGB").save(figdir / f"{stem}.tiff", compression="tiff_lzw", dpi=(600, 600))
fig.savefig(qadir / f"{stem}_preview.png", dpi=180)
plt.close(fig)
pd.DataFrame(display_rows).to_csv(qadir / "S2_annual_regional_display.csv",
                                 index=False, float_format="%.17g")
protection.to_csv(qadir / "S2_annual_protection_display.csv",
                  index=False, float_format="%.17g")
caption = (
    r"Regional urban effects and overall protection benefits through time. "
    r"\textbf{a}, Annual urban-growth minus fixed-2025 differences in UMRI ($\Delta\mathrm{UMRI}$) "
    r"in ten macroregions under four pathways. Lines show medians and shaded bands the "
    r"17th--83rd percentiles across 15 joint members; markers identify 2100 endpoints. "
    r"Regional panels retain original UMRI values on labelled linear axes with different ranges. "
    r"Three trajectories are omitted because every member has exactly zero urban effect "
    r"in all 75 years: SSP5--8.5 in Central America, and SSP2--4.5 and SSP3--7.0 in Western Asia. "
    r"The thin grey line retains zero as a reference. Western Asia under SSP1--2.6 is retained: "
    r"its effect first becomes non-zero in 2092 and reaches a median of $-1.93\times10^{-5}$ "
    r"UMRI in 2100. \textbf{b}, Overall annual protected minus urban-growth "
    r"UMRI differences under SSP2--4.5 and SSP5--8.5. Both panels retain 2026--2100 "
    r"and the same member-quantile definition. Development choices are paired at the same "
    r"locations under common climate, sea-level and ecological-model combinations. "
    r"Regional and overall estimates average the computational intermediate over valid "
    r"equal-area reference centres before applying the fixed UMRI mapping. "
    r"Colours identify SSP pathways consistently across panels. All values, including "
    r"exact-zero trajectories and cumulative effects, are supplied in Source Data.")
(output / "captions_S2.json").write_text(
    json.dumps({"S2": caption}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
(output / "captions_S2.tex").write_text(caption + "\n", encoding="utf-8")
manifest = {
    "renderer": Path(__file__).name,
    "inputs": inputs,
    "width_mm": WIDTH, "height_mm": HEIGHT,
    "layout": {"regional_axes_width_mm": 23, "regional_axes_height_mm": 24, "column_pitch_mm": 32.75, "row_pitch_mm": 33, "region_title_gap_mm": 2, "regional_titles_pt": 7.5, "panel_letters_pt": 8.4, "panel_titles_pt": 7.6, "tick_font_pt": 7.0},
    "TIFF_export": "600 dpi, RGB, lossless LZW compression",
    "figure": stem,
    "annual_regional_median_interval_points_in_source": 3000,
    "annual_regional_median_interval_points_displayed": 2775,
    "omitted_exact_zero_annual_points": 225,
    "omitted_series": zero_records,
    "western_asia_zero_audit": "qa/exact_zero_series_audit.json",
    "annual_protection_median_interval_points": 150,
    "region_ids": list(range(1, 11)), "pathways": PATHWAYS,
    "member_count": 15, "years_inclusive": [2026, 2100],
    "uncertainty": "Input 17th and 83rd member percentiles; no new interval or resampling.",
    "regional_axes": "Original UMRI units with explicit region-specific linear ranges; no regional normalization.",
    "contrasts": {"a": "urban_growth_minus_fixed", "b": "protection_minus_growth"},
    "deleted_panel": "Cumulative regional forest plot removed; complete values remain in unchanged Source Data.",
    "scientific_data_modified": False, "models_fitted": 0,
    "text_inside_canvas": True,
    "outputs": {path.name: sha(path) for path in figdir.iterdir() if path.is_file()}
}
(qadir / "future_process_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"completed": stem, "output": str(output), "input_files": len(inputs),
                  "regional_years_in_source": 3000, "regional_years_displayed": 2775, "protection_years": 150}, ensure_ascii=False))
