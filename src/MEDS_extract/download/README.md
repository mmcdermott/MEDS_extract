# `MEDS_extract.download`

The shared download layer for MEDS_extract-based ETLs: a small, transport-agnostic API
for staging a dataset's raw files into a local directory before the MEDS_extract stage
pipeline runs.

## Why this submodule exists

A MEDS ETL can't start until the raw source files are sitting on local disk. Getting
them there is deceptively fiddly — datasets live behind credentialed HTTP endpoints,
PhysioNet release manifests, S3/GCS buckets, or a colleague's pre-downloaded mirror;
they need checksum verification, resumable transfers, and politeness toward
rate-limited hosts.

This submodule exists so that **a downstream ETL never has to write download code at
all**. Instead, it declares *where its raw files live* in a standardized `sources:`
block in its MESSY spec, and `meds-extract-download` turns that declaration into a
deterministic, verifiable local copy. The goals:

- **One specification structure.** Every ETL describes its raw data the same way — a
    `sources:` block of typed backend entries — so the "how do I get this dataset"
    question has a uniform, reviewable answer that lives next to the rest of the spec.
- **One toolchain.** A single CLI (`meds-extract-download`) and a single Python API
    (`Source.download_all`) stage any dataset, regardless of where it's hosted. ETL
    authors compose backends; they don't reimplement transports.
- **Deterministic, verified retrieval.** SHA-256 verification, atomic writes, and a
    strict skip/overwrite policy mean a download either produces exactly the manifest's
    files or fails loudly — no silently-stale local copies leaking into a pipeline run.

The submodule sits *alongside* the MEDS-transforms stage DAG, not inside it: download
I/O is network / blob storage rather than sharded parquet, parallelism is per-file
transport streams rather than per-shard workers, and failures are partial-retry / resume
rather than per-stage. It keeps the same ergonomics as a stage (Hydra-driven,
CLI-addressable, override-friendly) — just with its own machinery.

## Overview

At the highest level, staging a dataset is four steps:

1. A MESSY spec declares where its raw files live in a `sources:` block.
2. `spec.py` (`sources_from_spec`) turns each entry into a `Source` instance —
    `HTTPSource`, `FsspecSource`, or `PhysioNetSource`.
3. `Source.download_all` is called on each source...
4. ...staging every file into one shared `raw_input_dir/`.

A **`Source`** is anywhere raw data comes from. It knows two things: *what files it
offers* (`_list_files`) and *how to stream one file's bytes to a local path* (`_pull`).
Everything else — `.part` staging, SHA-256 verification, atomic rename, the
skip/overwrite/error policy, manifest validation and include/exclude filtering,
duplicate-destination detection, sequential-vs-parallel orchestration, error
aggregation — lives once on the `Source` ABC and is shared by every backend.

### Using it from the CLI

The common case. A MESSY spec declares its backends:

```yaml
sources:
  dataset: # the bucket selected by `key=` (default: "dataset")
    - type: physionet
      base_url: https://physionet.org/files/mimiciv/3.1
      username: ${oc.env:PHYSIONET_USER}
      password: ${oc.env:PHYSIONET_PASS}
      include: # optional fnmatch globs — stage only what the ETL reads
        - hosp/*.csv.gz
        - icu/*.csv.gz
  common: # always appended, regardless of `key=`
    - type: http
      urls:
        - https://raw.githubusercontent.com/.../concept_map.csv
```

and `meds-extract-download` stages it (Hydra dotlist overrides, one command):

```bash
meds-extract-download spec=/path/to/messy.yaml raw_input_dir=/path/to/raw key=dataset concurrency=4
```

The override knobs:

- `key` — which `sources:` bucket to pull; `common` is always appended. A `key`
    that names no bucket in the spec is an error, not a silent no-op.
- `concurrency` — size of the one thread pool shared across all sources.
- `continue_on_error` — collect per-file failures and keep going (all sources are
    attempted; the process exits non-zero at the end if anything failed). With the
    default `False`, the first failing source stops the whole run.
- `do_overwrite` — re-fetch every file even if a verified local copy exists.

Before any fetch, the CLI materializes every source's manifest and rejects
cross-source destination collisions (two sources listing the same `rel_path`), so a
misconfigured spec fails precisely and immediately rather than mid-download. The
process exits `0` only on full success.

### Using it from Python

The CLI is a thin wrapper over the library API. `download_all` is the single entry
point — sequential by default, parallel when handed a pool. The example below is a
runnable doctest (a local `fsspec` source standing in for a remote host):

```python
>>> from concurrent.futures import ThreadPoolExecutor
>>> from MEDS_extract.download import FsspecSource, validate_unique_destinations
>>> mirror = '''
... patients.csv: "patient_id,dob\\n1,2000-01-01\\n"
... labs:
...   vitals.csv: "pid,hr\\n1,80\\n"
... '''
>>> with yaml_disk(mirror) as src_dir, tempfile.TemporaryDirectory() as raw:
...     sources = [FsspecSource(root=str(src_dir))]  # what sources_from_spec returns
...     validate_unique_destinations(sources)  # reject cross-source collisions up-front
...     with ThreadPoolExecutor(max_workers=4) as pool:
...         for src in sources:
...             with src:  # close() owned network clients on exit
...                 src.download_all(raw, pool=pool)
...     print_directory(Path(raw))
├── labs
│   └── vitals.csv
└── patients.csv

```

For a single source the whole dance collapses to
`FsspecSource(root=...).download_all("raw/")` — sequential, no pool, nothing to close
for backends that own no network client (HTTP-backed sources should use `with`).

The rest of this document walks through the pieces behind that API.

## Files

| File                                             | Responsibility                                                                                                                                                                         |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`source.py`](source.py)                         | The `Source` ABC, the `RemoteFile` manifest row, `ChecksumError`, `sha256_of`, `validate_unique_destinations`, and the whole orchestration loop (`download_all` + helpers).            |
| [`backends/http.py`](backends/http.py)           | `HTTPSource` — explicit list of URLs. tenacity-retried client + streaming, `.part`-file Range-resume download, `Content-Range` validation. No crawling.                                |
| [`backends/physionet.py`](backends/physionet.py) | `PhysioNetSource(HTTPSource)` — discovers its file list from the `SHA256SUMS.txt` manifest every PhysioNet release publishes. Overrides `_list_files` (plus its constructor).          |
| [`backends/fsspec.py`](backends/fsspec.py)       | `FsspecSource` — any `fsspec` protocol via `universal_pathlib` (`file://`, `s3://`, `gs://`, …). For re-runs against a pre-downloaded local / cloud mirror.                            |
| [`spec.py`](spec.py)                             | `source_from_config` / `sources_from_spec` — turn raw `sources:` YAML entries into concrete `Source` instances. The one place the `type:` → class registry lives.                      |
| [`cli.py`](cli.py)                               | `meds-extract-download` — the Hydra entry point. Resolves the spec, builds + cross-validates the sources, owns the shared thread pool, drives every source, exits non-zero on failure. |
| [`backends/__init__.py`](backends/__init__.py)   | Lazily re-exports the three backend classes (PEP 562), so the HTTP stack is only imported when actually used.                                                                          |
| [`__init__.py`](__init__.py)                     | Public surface: `Source`, `RemoteFile`, `ChecksumError`, the three backends, `source_from_config`, `sources_from_spec`, `validate_unique_destinations`.                                |

## Architecture

### `Source` — the ABC

Concrete backends implement exactly two hooks:

```python
@abstractmethod
def _list_files(self) -> Iterable[RemoteFile]: ...  # enumerate files
@abstractmethod
def _pull(self, source_path: str, target: Path) -> None: ...  # stream bytes
```

The base class supplies everything else. `_fetch_one(item, dest_dir, do_overwrite)`
is the per-file pipeline (orchestrator-facing): resolve dest, apply the
skip/overwrite/error policy, derive `.part`, call `_pull`, verify SHA-256,
atomic-rename.

The user-facing entry points:

- **`download_all(dest_dir, *, pool=None, continue_on_error=False, do_overwrite=False)`**
    — the single public fetch entry point.
- **`files`** — a `cached_property` wrapping `_list_files()`: materializes the manifest
    to a list, applies the constructor's `include` / `exclude` globs, validates that
    every `rel_path` is unique, and caches the result so a network-backed manifest
    (PhysioNet's `SHA256SUMS.txt`) isn't re-fetched on a second `download_all` call.
    `n_files` is the corresponding count.
- **`close()` / `__enter__` / `__exit__`** — resource lifecycle. Backends that own a
    network client (`HTTPSource`) override `close()`; the CLI registers every source
    with an `ExitStack` so clients are released deterministically.

Every backend constructor also accepts **`include`** / **`exclude`** — `fnmatch`-style
globs over the (normalized) `rel_path` — so an ETL can stage only the subset of a large
release it actually reads (e.g. `include: ["hosp/*.csv.gz"]` against a release that also
bundles waveforms or images).

### `RemoteFile` — the manifest row

A frozen, self-validating dataclass. Every `_list_files` implementation — in-repo
backend or downstream `Source` subclass — constructs these, so it is exported from
`MEDS_extract.download`:

```python
@dataclass(frozen=True)
class RemoteFile:
    rel_path: str  # where it lands under dest_dir (forward slashes)
    source_path: str  # transport's source-side address (URL / UPath spec)
    sha256: str | None = None  # the only verifier the orchestrator trusts
```

Validation runs in `__post_init__`, so a malformed row fails the instant it is built:
`rel_path` must be relative, forward-slash, and must not escape the destination
directory after normalization; `sha256` (when set) must be 64 hex chars and is
normalized to lowercase. The cross-*row* check — no two rows resolving to the same
destination — lives in `Source.files`, and the cross-*source* variant in
`validate_unique_destinations`.

`sha256` is the only verifier the orchestrator trusts to skip a re-fetch. A
`RemoteFile` with no `sha256` can still be downloaded, but on a re-run it can't be
*skipped* — see the overwrite policy below.

### The orchestration loop

`download_all` is one straight pass: get the validated manifest, turn it into a stream
of fetch *attempts*, and run them through a single error-collection loop.

1. **`self.files`** — the validated, filtered, cached manifest.
2. **`_iter_attempts(...)`** pairs each row with the zero-arg thunk that fetches it;
    **`_attempts(...)`** dispatches those pairs by mode:
    - **no pool** → the pairs pass through; the `callable` runs `_fetch_one` in the
        calling thread when invoked.
    - **pool given** → every thunk is submitted up front; pairs come back as
        `(item, future.result)` in completion order.
3. A single `for item, run in attempts:` loop calls `run()`. On a per-file exception:
    with `continue_on_error=True` the error is collected and the loop continues;
    otherwise it propagates immediately (and in pooled mode the still-queued futures
    are cancelled on the way out).
4. If any errors were collected, they are raised together as one `ExceptionGroup`;
    otherwise `download_all` returns `None`.

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

Two `.part`-level refinements: a leftover `.part` that already verifies against the
manifest sha is promoted to `dest` directly (a prior run died between the last byte
and the rename — no re-fetch needed), and a leftover `.part` with *no* manifest sha to
verify against is discarded (resume-without-verification is unsafe).

### Pool ownership

`download_all` runs sequentially by default; pass an `Executor` (typically a
`ThreadPoolExecutor`) to opt into parallelism, and the caller *owns its lifetime*.
That gives:

- **Caller-controlled worker cap** — sized to whatever the transport tolerates (one
    rate-limited host vs. ten fast ones is a per-deployment decision).
- **One pool shared across sources** — the CLI builds a single pool sized to
    `concurrency=` and hands it to every source's `download_all`, so the bound is global.
- **Deterministic teardown** — the CLI shuts the pool down with
    `shutdown(wait=False, cancel_futures=True)`, so a `Ctrl+C` mid-download cancels all
    *queued* work immediately. Worker threads are not daemons (Python ≥3.9 joins them
    at interpreter exit), so *in-flight* transfers end when their transport is torn
    down: the CLI closes each HTTP source's client on the way out, which kills live
    streams promptly; fsspec copies have no abort path and run to completion.

### `spec.py` — spec → objects

`source_from_config({"type": "http", "urls": [...]})` looks the `type:` string up in
the `_SOURCE_TYPES` registry — the one dict mapping each type to its backend module
and class (the "supported types" error message derives from the same dict, so the two
can't drift). Backend modules import lazily, per selected type.
`sources_from_spec(spec, key="dataset")` reads a whole `sources:` block, pulls the
selected bucket plus the always-appended `common:` bucket, and returns the constructed
list. New backend = new `backends/` module + one registry row.

### `cli.py` — the `meds-extract-download` entry point

A Hydra entry point (`DownloadConfig` is a `hydra_registered_dataclass`). It:

1. resolves the spec path against the user's original CWD (Hydra changes CWD);
2. resolves OmegaConf interpolations on **only** the `sources:` subtree — so a combined
    MESSY file's unrelated `${oc.env:...}` interpolations in the event-conversion
    section don't need to be set just to download;
3. validates `key=` against the buckets the spec actually declares, then builds the
    sources via `sources_from_spec`;
4. opens an `ExitStack`, creates one shared pool, registers every source for
    `close()`, rejects cross-source destination collisions via
    `validate_unique_destinations`, and calls `download_all` on each — stopping at the
    first failed source unless `continue_on_error=true`;
5. exits `0` on full success and `1` otherwise (via explicit `sys.exit` — Hydra
    discards the task function's return value, so returning an exit code would not
    work).

## Adding a backend

1. Add `backends/<name>.py` with a `class FooSource(Source)` implementing `_list_files`
    and `_pull`. The base class wraps `_pull` with `.part` staging, SHA-256 verification,
    and atomic rename — `_pull`'s only contract is "produce a complete file at `target`
    or raise." Accept `include` / `exclude` in your constructor and forward them to
    `super().__init__` so manifest filtering works uniformly.
2. Add one row to `_SOURCE_TYPES` in `spec.py`.
3. Cover it with doctests in the backend module (per the project's doctest-first
    convention) and add wire-level tests to `tests/test_download.py` if it needs a real
    transport round-trip.

## Testing

- **Doctests** in each module cover the pure logic: spec dispatch, URL normalization,
    `RemoteFile` validation, `SHA256SUMS.txt` parsing, manifest filtering, and the
    `Source.download_all` skip/overwrite/traversal/dup paths (via stub sources in the
    `source.py` docstrings). This README's Python-usage example is itself a collected
    doctest.
- **`tests/test_download.py`** covers what doctests can't: `_resumable_stream`'s
    wire-level behavior (Range resume, 416/206 mismatch handling, identity
    content-coding) against `httpx.MockTransport`, streaming retry behavior, the
    `Source._fetch_one` staging pipeline (sha verify + atomic rename + `.part`
    promotion/discard), end-to-end `download_all` flows (sequential and pooled), the
    CLI subprocess paths (success, failure exit codes, key validation, collision
    rejection), and the SIGINT-cancellation regression (which needs a real signal
    in a real subprocess — `tests/_fetcher_sigint_child.py`).
- **`tests/test_example.py`** exercises the real PhysioNet path end-to-end (gated
    behind the `integration` marker).
