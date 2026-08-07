Joins external metadata files (declared via `_metadata` blocks in the event config) onto the
extracted event codes. Each `_metadata` entry maps output column names to dftly expressions
over the raw metadata table; produced columns whose names match the code's component columns
are the join keys. Here, `labs/lab._metadata.lab_descriptions` produces `test_name` (the
code's single component, so it is the join key) and a `description` output from
`lab_descriptions.csv`. The reducer writes the joined table to `metadata/codes.parquet`, with
a `code_template` column preserving the original dftly expression that produced each code
(useful for downstream provenance tracking). The output enumerates every observed code — a
MEDS validity requirement — so codes with no metadata match (`EYE_COLOR//*`, `MEDS_BIRTH`)
appear with null `description` / `code_template`.
