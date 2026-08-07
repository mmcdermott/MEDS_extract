"""Mirror the repository's root ``static/`` assets into the built site.

``docs/index.md`` snippet-includes the root ``README.md``, whose logo ``<picture>``
block points at ``static/logo_*.svg`` via repo-root-relative paths. Those resolve on
GitHub (the files live next to the README) but MkDocs only ships files found under
``docs_dir``, so on the built site the logo 404s (#244). Rather than duplicating the
assets or symlinking (fragile on Windows checkouts), copy them into the virtual docs
tree at build time so the home page — rendered at the site root — sees ``static/...``
at the same relative location as on GitHub.
"""

from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent

for path in sorted((root / "static").rglob("*")):
    if not path.is_file():
        continue
    with mkdocs_gen_files.open(path.relative_to(root).as_posix(), "wb") as fd:
        fd.write(path.read_bytes())
