"""Generate per-stage documentation pages for MEDS_extract's registered stages.

Delegates to :func:`MEDS_transforms.stages.docgen.generate_stage_docs`, which walks every
``@Stage.register``'d entry point under the given package, renders its docstring, metadata,
default config, schema updates, CLI usage, and per-scenario ``StageExample.render_content(...)``
output into a single Markdown page per stage.

Each generated page lands at ``docs/stages/<stage_name>/index.md`` and the collection is
indexed via ``docs/stages/SUMMARY.md`` so ``mkdocs-literate-nav`` can wire it into the
top-level navigation.
"""

from pathlib import Path

import mkdocs_gen_files
from MEDS_transforms.stages.docgen import generate_stage_docs

nav = mkdocs_gen_files.Nav()

for doc in generate_stage_docs("MEDS_extract"):
    nav_key = Path(doc.path)
    doc_path = nav_key / "index.md"
    full_doc_path = Path("stages") / doc_path

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(doc.content)

    nav[nav_key.parts] = doc_path.as_posix()

    if doc.edit_path:
        mkdocs_gen_files.set_edit_path(full_doc_path, doc.edit_path)

with mkdocs_gen_files.open("stages/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
