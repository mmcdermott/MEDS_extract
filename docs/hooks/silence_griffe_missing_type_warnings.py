"""MkDocs hook: drop griffe's "No type or annotation" docstring warnings.

The codebase's Google-style docstrings put types in signatures, not prose, and
document Hydra config keys as pseudo-parameters (``cfg.stage_cfg.split_fracs``) and
grouped pass-through kwargs (``include, exclude:``) that have no single signature
annotation. griffe warns "No type or annotation for parameter ..." on every one,
which would fail ``mkdocs build --strict`` (#244) despite being deliberate style.

griffe has a first-class ``warn_missing_types: false`` docstring option for exactly
this, but it only became configurable through mkdocstrings-python in 1.17, which
requires mkdocstrings >= 1.0 — a major bump from the 0.28.2 this repo pins. Until
those pins move, filter the message class at the logging layer instead. Delete this
hook (and set ``warn_missing_types: false`` next to ``warn_unknown_params`` in
mkdocs.yml) once the docs toolchain is bumped.

Failure mode is safe: if griffe's message wording or logger routing changes, the
filter stops matching and the warnings resurface as strict-build failures rather
than being silently swallowed.
"""

import logging

# mkdocstrings-python patches griffe's logger to emit under the mkdocs plugin
# namespace (so mkdocs' --strict counter sees it), prefixing messages with
# "griffe: ". Filtering on the emitting logger drops the record before it
# propagates to mkdocs' warning counter.
_GRIFFE_LOGGER_NAME = "mkdocs.plugins.griffe"
_SILENCED_MARKER = "No type or annotation for"


class _MissingTypeWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.levelno == logging.WARNING and _SILENCED_MARKER in record.getMessage())


_FILTER = _MissingTypeWarningFilter()


def on_startup(command: str, dirty: bool) -> None:
    """Install the filter once per mkdocs process (idempotent across rebuilds)."""
    logger = logging.getLogger(_GRIFFE_LOGGER_NAME)
    if _FILTER not in logger.filters:
        logger.addFilter(_FILTER)
