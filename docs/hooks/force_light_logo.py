"""MkDocs hook: always serve the light logo on the docs site.

The README's ``<picture>`` block selects ``logo_dark.svg`` via a
``prefers-color-scheme: dark`` media query. On GitHub that is correct — the page
background follows the same query — but the docs site's Material theme defaults to its
light palette regardless of the visitor's OS scheme, so an OS-dark visitor gets the
dark logo on a light page. Strip the dark ``<source>`` at render time so the
``<img>`` fallback (the light logo) always shows; the README itself keeps the
GitHub-correct markup.

Failure mode is safe: if the README's markup changes shape, the substitution stops
matching and the site falls back to GitHub's behavior rather than breaking the build.
"""

import re

_DARK_SOURCE_RE = re.compile(r"<source[^>]*prefers-color-scheme:\s*dark[^>]*>\s*", re.IGNORECASE)


def on_page_content(html: str, page, config, files) -> str:
    """Drop dark-scheme ``<source>`` tags from any page's rendered HTML (only home has one).

    Runs on the converted HTML rather than the raw markdown because the home page's
    markdown is just a ``--8<--`` snippet-include directive — the README content only
    exists after conversion.
    """
    return _DARK_SOURCE_RE.sub("", html)
