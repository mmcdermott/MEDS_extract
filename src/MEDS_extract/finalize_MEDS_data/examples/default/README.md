Takes a merged MEDS cohort (the output of `merge_to_MEDS_cohort`, with auxiliary
`code_components` / `source_block` columns alongside the MEDS core) and returns the same rows
with the mandatory MEDS schema applied to the core columns: `subject_id: Int64`,
`time: Datetime("us")`, `code: String`, `numeric_value: Float32`. Extension columns pass
through unchanged.

This stage should almost always be the last data stage in an extraction pipeline.
