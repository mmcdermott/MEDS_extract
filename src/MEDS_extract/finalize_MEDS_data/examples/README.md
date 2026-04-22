Takes a subject-sharded MEDS cohort with (possibly loosely-typed) columns and returns the same
rows with the mandatory MEDS schema applied: `subject_id: Int64`, `time: Datetime("us")`,
`code: String`, `numeric_value: Float32`. No content changes — this is a schema-alignment pass
intended to run as the final data stage in an extraction pipeline.
