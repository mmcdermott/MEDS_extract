Joins external metadata files (declared via `_metadata` blocks in the event config) onto the
extracted event codes. Here, `labs/lab._metadata.lab_descriptions` declares that
`lab_descriptions.csv` (matched on the `test_name` column, which produces the code) provides a
`description` column. The reducer writes the joined table to `metadata/codes.parquet`, with
a `code_template` column preserving the original dftly expression that produced each code
(useful for downstream provenance tracking).
