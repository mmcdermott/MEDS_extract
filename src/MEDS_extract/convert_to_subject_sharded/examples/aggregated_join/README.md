Demonstrates the *aggregated* form of `_table.join` (issue #65) — `cols: {deathtime: min}`
groups the right-hand `admissions` table by the join key and reduces each listed column
before the left join, so every `patients` row gains a single `min(deathtime)` value.

The right side is deliberately split across **two** chunk files: subject 111 has admissions
in both chunks, and its earliest death-time (`2020-03-01`) lives in the *second* chunk —
proving the aggregation runs over the concatenated multi-file scan, not per chunk. This is
the MIMIC-IV `fix_static_data` replacement shape: earliest death-time per subject pulled
from the admissions table straight in the MESSY spec.
