# `MEDS_extract.download`

The shared download layer for MEDS_extract-based ETLs: a small, transport-agnostic API
for staging a dataset's raw files into a local directory before the MEDS_extract stage
pipeline runs.

## Why this submodule exists

Every public MEDS ETL (MIMIC-IV, eICU, AUMCdb, HIRID, …) used to ship its own bespoke
`download.py` — a hand-rolled mix of `requests`/`wget` calls, HTML scraping, ad-hoc
checksum logic, and copy-paste retry loops. Issue
[#81](https://github.com/mmcdermott/MEDS_extract/issues/81) consolidated that into one
layer: every ETL declares its raw-data backends in a `sources:` block in its MESSY
spec, and `meds-extract-download` stages them uniformly.

This is **deliberately not a MEDS-transforms stage**. Download's I/O contract
(network/blob storage, not sharded parquet), parallelism axis (per-file transport
streams, not per-shard workers), failure model (partial-retry, resume), and config
scope all differ from the stage DAG. It sits as a *pipeline-adjacent* hook: same
ergonomic goals as a stage (Hydra-driven, CLI-addressable, override-friendly) without
being forced into the stage machinery.

## The mental model

```
MESSY spec ──► dispatch ──► [Source, Source, …] ──► download_all() ──► raw_input_dir/
  sources:        │              │                      │
   dataset:       │              │                      ├─ files (cached manifest)
    - type: ...   │              │                      ├─ skip / overwrite / error
   common:        │              │                      └─ _fetch() per file
    - type: ...   │              │
                  │              └─ HTTPSource / FsspecSource / PhysioNetSource
                  └─ source_from_config() / sources_from_spec()
```

A **`Source`** is anywhere raw data comes from. It knows two things: *what files it
offers* (`_list_files`) and *how to move one file's bytes to a local path* (`_fetch`).
Everything else — the skip/overwrite/error policy, path-traversal validation,
duplicate-destination detection, sequential-vs-parallel orchestration, error
aggregation — lives once on the `Source` ABC and is shared by every backend.

## Files

| File                    | Responsibility                                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `source.py`             | The `Source` ABC, the `RemoteFile` manifest row, `ChecksumError`, `sha256_of`, and the whole orchestration loop (`download_all` + private helpers).          |
| `backends/http.py`      | `HTTPSource` — explicit list of URLs. tenacity-wrapped client, `.part`-file Range-resume download, `Content-Range` validation. No crawling.                  |
| `backends/physionet.py` | `PhysioNetSource(HTTPSource)` — discovers its file list from the `SHA256SUMS.txt` manifest every PhysioNet release publishes. Overrides only `_list_files`.  |
| `backends/fsspec.py`    | `FsspecSource` — any `fsspec` protocol via `universal_pathlib` (`file://`, `s3://`, `gs://`, …). For re-runs against a pre-downloaded local / cloud mirror.  |
| `dispatch.py`           | `source_from_config` / `sources_from_spec` — turn raw `sources:` YAML entries into concrete `Source` instances. The one place the `type:` → class map lives. |
| `cli.py`                | `meds-extract-download` — the Hydra entry point. Resolves the spec, builds the sources, owns the shared thread pool, drives every source.                    |
| `__init__.py`           | Public surface: `Source`, the three backends, `source_from_config`, `sources_from_spec`.                                                                     |

## Architecture

### `Source` — the ABC

Concrete backends implement exactly two hooks:

```python
@abstractmethod
def _list_files(self) -> Iterable[RemoteFile]: ...  # what files exist
@abstractmethod
def _fetch(self, remote: RemoteFile, dest: Path) -> None: ...  # move one file's bytes
```

and the base class supplies everything users actually call:

- **`download_all(dest_dir, *, pool=None, continue_on_error=False, do_overwrite=False)`**
    — the single public fetch entry point.
- **`files`** — a `cached_property` wrapping `_list_files()`: materializes the manifest
    to a list, validates that every `rel_path` is unique, and caches the result so a
    network-backed manifest (PhysioNet's `SHA256SUMS.txt`) isn't re-fetched on a second
    `download_all` call.
- **`close()` / `__enter__` / `__exit__`** — resource lifecycle. Backends that own a
    network client (`HTTPSource`) override `close()`; the CLI registers every source
    with an `ExitStack` so clients are released deterministically.

### `RemoteFile` — the manifest row

A pure-POD `NamedTuple`, never seen by users — it only crosses the boundary between a
backend's `_list_files` and the orchestrator:

```python
class RemoteFile(NamedTuple):
    rel_path: str  # where it lands under dest_dir (forward slashes)
    sha256: str | None = None  # the only verifier the orchestrator trusts
    source_path: str | None = None  # transport's source-side address (URL / UPath spec)
```

`sha256` is the *only* skip/verify signal. Size was considered and dropped: same-size
files can differ, and a hash already catches every mismatch a size check would. A
`RemoteFile` with no `sha256` can still be downloaded, but it can never be *skipped* on
a re-run — see the overwrite policy below.

### The orchestration loop

`download_all` is one straight pass:

1. `self.files` — get the validated, cached manifest (raises on duplicate `rel_path`).
2. `_attempts(...)` — a generator that yields `(item, callable)` pairs. This is the
    sole sequential-vs-parallel branch point:
    - **no pool** → yields `(item, partial(self._fetch_one, ...))` — runs in the calling
        thread when invoked.
    - **pool given** → submits every `_fetch_one` to the pool up front, yields
        `(item, future.result)` in completion order.
3. A single `for item, run in attempts:` loop calls `run()` and routes any exception
    through the same error-collection path, regardless of which mode produced the pair.
4. With `continue_on_error=True`, per-file errors are collected and raised as one
    `ExceptionGroup` at the end (so the caller sees *every* failure, not just the
    first); otherwise the first failure propagates immediately.

`_fetch_one` is where the per-file **skip / overwrite / error policy** lives:

| `dest` state                               | `do_overwrite=False`  | `do_overwrite=True` |
| ------------------------------------------ | --------------------- | ------------------- |
| doesn't exist                              | fetch                 | fetch               |
| exists, verifies against manifest `sha256` | **skip**              | clear + refetch     |
| exists, sha mismatch *or* no manifest sha  | **`FileExistsError`** | clear + refetch     |

The "exists but can't verify → error" rule is intentional: silently overwriting (or
silently skipping) a file we can't prove matches the manifest is how stale or
half-flushed local copies leak into a pipeline run. The user has to opt into the
ambiguity with `do_overwrite=True`.

### Pool ownership

`download_all` **never builds a thread pool**. Sequential is the default; the caller
passes a `ThreadPoolExecutor` to opt into parallelism and *owns its lifetime*. This
matters because:

- The worker cap is the caller's concern — they know whether they're hitting one
    rate-limited host or ten fast ones. A library-internal `max_workers=4` is a buried
    surprise.
- One pool can be **shared across sources**: the CLI builds a single pool sized to
    `concurrency=` and hands it to every source's `download_all`, so the bound is global
    rather than per-source.
- The CLI shuts the pool down with `shutdown(wait=False, cancel_futures=True)` rather
    than the `with` block's default `wait=True` — a `Ctrl+C` mid-download cancels queued
    work immediately instead of blocking until a multi-GiB transfer drains.

### `dispatch.py` — spec → objects

`source_from_config({"type": "http", "urls": [...]})` is the one `match` statement
mapping a `type:` string to a backend class. `sources_from_spec(spec, key="dataset")`
reads a whole `sources:` block, pulls the selected bucket plus the always-appended
`common:` bucket, and returns the constructed list. New backend = new
`backends/` module + one `case` in `dispatch.py`.

### `cli.py` — the `meds-extract-download` entry point

A Hydra entry point (`DownloadConfig` is a `hydra_registered_dataclass`). It:

1. resolves the spec path against the user's original CWD (Hydra changes CWD);
2. resolves OmegaConf interpolations on **only** the `sources:` subtree — so a combined
    MESSY file's unrelated `${oc.env:...}` interpolations in the event-conversion
    section don't need to be set just to download;
3. builds the sources via `sources_from_spec`;
4. opens an `ExitStack`, creates one shared pool, registers every source for
    `close()`, and calls `download_all` on each;
5. returns `0` on full success, `1` if any source raised.

## Usage

### CLI

```bash
meds-extract-download \
  spec=/path/to/messy.yaml \
  raw_input_dir=/path/to/raw \
  key=dataset \           # which sources: bucket — 'common' is always appended
  concurrency=4 \         # shared thread-pool size across all sources
  continue_on_error=false \
  do_overwrite=false
```

The `sources:` block in the MESSY spec:

```yaml
sources:
  dataset:
    - type: physionet
      base_url: https://physionet.org/files/mimiciv/3.1
      username: ${oc.env:PHYSIONET_USER}
      password: ${oc.env:PHYSIONET_PASS}
  common:
    - type: http
      urls:
        - https://raw.githubusercontent.com/.../concept_map.csv
```

### Library

```python
from concurrent.futures import ThreadPoolExecutor
from MEDS_extract.download import HTTPSource, sources_from_spec

# Single source, sequential:
HTTPSource(urls=["https://example.com/a.csv"]).download_all("raw/")

# Multiple sources sharing one pool:
sources = sources_from_spec(spec, key="dataset")
with ThreadPoolExecutor(max_workers=4) as pool:
    for src in sources:
        with src:  # close() owned clients on exit
            src.download_all("raw/", pool=pool)
```

## Adding a backend

1. Add `backends/<name>.py` with a `class FooSource(Source)` implementing `_list_files`
    and `_fetch`. Honor the `Source` invariants documented in the `source.py` docstring
    — most importantly: `_fetch` stages to a sibling `.part` file and atomic-renames,
    verifies `remote.sha256` when set, and raises (never half-writes `dest`) on failure.
2. Export it from `backends/__init__.py`.
3. Add a `case "<name>":` to `source_from_config` in `dispatch.py`.
4. Cover it with doctests in the backend module (per the project's doctest-first
    convention) and add wire-level tests to `tests/test_download.py` if it needs a real
    transport round-trip.

## Testing

- **Doctests** in each module cover the pure logic: dispatch, URL normalization,
    `SHA256SUMS.txt` parsing, and the `Source.download_all` skip/overwrite/traversal/dup
    paths (via stub sources in the `source.py` docstring).
- **`tests/test_download.py`** covers what doctests can't: `_resumable_download`'s
    wire-level behavior (Range resume, 416/206 mismatch handling) against
    `httpx.MockTransport`, end-to-end `download_all` flows, the CLI subprocess path, and
    the SIGINT-cancellation regression (which needs a real signal in a real subprocess —
    `tests/_fetcher_sigint_child.py`).
- **`tests/test_example.py`** exercises the real PhysioNet path end-to-end (gated
    behind the `integration` marker).
