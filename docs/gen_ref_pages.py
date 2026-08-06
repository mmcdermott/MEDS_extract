"""Generate the code reference pages."""

import re
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"

GITHUB_URL = "https://github.com/mmcdermott/MEDS_extract"

# Matches the target of an inline Markdown link/image whose target is relative (skips
# absolute URLs, in-page anchors, and mailto:). Group 1 is the ``[text](`` prefix,
# group 2 the target (with any ``#fragment``), group 3 the closing paren.
_RELATIVE_LINK_RE = re.compile(r"(\[[^\]]*\]\()(?!(?:[a-z][a-z0-9+.-]*:|#))([^)\s]+)(\))")


def absolutize_relative_links(text: str, base_dir: Path) -> str:
    """Rewrite ``text``'s relative Markdown links as absolute GitHub URLs.

    Package READMEs are written to be read in-repo, where relative links to source
    files (``[cli.py](cli.py)``) work. When a README is embedded in a generated API
    page, those targets do not exist among the documentation files, so ``--strict``
    builds flag every one of them (#244). Rewriting them to ``blob``/``tree`` URLs on
    the default branch keeps both renderings working without touching the README.

    Targets that do not resolve to an existing repo path are left alone so that
    MkDocs' link validation still reports them as drift.
    """

    def _rewrite(match: re.Match) -> str:
        target, _, fragment = match.group(2).partition("#")
        resolved = (base_dir / target).resolve()
        if not resolved.is_relative_to(root) or not resolved.exists():
            return match.group(0)
        kind = "tree" if resolved.is_dir() else "blob"
        url = f"{GITHUB_URL}/{kind}/main/{resolved.relative_to(root).as_posix()}"
        if fragment:
            url = f"{url}#{fragment}"
        return f"{match.group(1)}{url}{match.group(3)}"

    return _RELATIVE_LINK_RE.sub(_rewrite, text)


for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = "api" / doc_path

    parts = tuple(module_path.parts)

    md_file_lines = []

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

        readme_path = src / Path(*parts) / "README.md"
        if readme_path.exists():
            readme_text = readme_path.read_text(encoding="utf-8")
            md_file_lines.append(absolutize_relative_links(readme_text, readme_path.parent))
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    ident = ".".join(parts)
    md_file_lines.append(f"::: {ident}")

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write("\n".join(md_file_lines))

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
