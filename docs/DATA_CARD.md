# mentat-sys1-v0.1 Data Card

The registered data product contains four immutable splits:

- train: 87,386 rows
- dev-general: 400 rows
- dev-hard: 400 rows
- calibration: 1,120 rows

Materialized JSONL is not committed to the code repository. The repository
stores split identities, source locks, aggregate statistics, and synthetic
fixtures.

The dataset is mixed-license and receives no blanket relicense. The frozen
training composition is:

- `LicenseRef-generated`: 74,412 rows
- `MIT`: 4,483 rows
- `Apache-2.0`: 4,917 rows
- `CC-BY-4.0`: 3,574 rows

Release remains blocked until the source-by-source inventory, attribution,
teacher lock, and JevBench 8-gram contamination report are complete and
verified.
