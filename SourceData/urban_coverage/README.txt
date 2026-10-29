Global historical habitat coverage

The study covers 934,814 of 1,204,702 equal-area cells: 77.59711530320361%.
Area is 841.3326 / 1084.2318 km2. The support is the GMW v4.1.12
2000-2004 presence union inside GUB2018 urban polygons with urbanArea >100 km2
and their 2 km EPSG:6933 buffers. Buffers include city interiors and reproduce
the original 30 m topology-preserving simplification and quad_segs=2 setting.
The 500 m ground-radius ecological neighbourhood is a separate quantity.

coverage_summary.json: exact counts, areas and coverage.
coverage_by_block.csv: all 671 unique candidate core blocks, including zeros.
reference_population_by_region.csv: the actual ten-region reference population.
reference_cells.npz: unique grid_x, grid_y and region_id for 934814 study cells.
global_baseline_cells.npz: unique grid_x and grid_y for 1204702 global cells.
Coordinates are integer cells on the globally aligned EPSG:6933 lattice;
x=(grid_x+0.5)*30 and y=(grid_y+0.5)*30 metres. Each cell represents 900 m2.
urban_buffers.gpkg: the 152 city buffers intersecting global product tile footprints.
These are candidates, not a count of cities confirmed to contain baseline mangroves.
gmw_v4112_tiles.geojson: the official global tile-footprint catalogue.
coverage_protocol.json: definitions, required tile identities and source checksums.

GMW v4.1.12 source: https://doi.org/10.5281/zenodo.21346457
GUB2018 source: http://data.ess.tsinghua.edu.cn
The GMW tar.gz archive is downloaded from its original provider; it is not
redistributed here. Its MD5 is 2ea47846a2af2de561ee419f151138a0 and its byte size
is 1396716026. The protocol supplies SHA-256 values for all 179 required tiles.
The calculation is global and does not apply the original study mosaic footprint.
The reference and denominator use the same baseline years, buffers and grid.
Overlapping cities and tiles are combined by union, and each core cell is counted once.
All actual study cells belong to the denominator. Annual observation completeness,
2025 extent and the 2000-2025 union are different supports.

Run the coverage verification/reproduction commands in SupplementarySoftware/README.txt.
