"""Shared console-script plumbing for the two Hydra CLIs.

One job: validate that the required dotlist arguments are present BEFORE Hydra owns
the process. Both CLIs anchor their Hydra run dir at ``${output_dir}/...``, so Hydra
cannot even start without ``output_dir`` — and the error it produces for a missing
interpolation names OmegaConf internals, not the flag the user forgot. Checking
``sys.argv`` first turns the most common new-user mistake (a bare invocation) into a
one-line usage message, and guarantees a failed invocation creates no directories
anywhere.
"""

from __future__ import annotations

import sys

# Flags under which Hydra should run without the required args (help/introspection
# modes; ``--cfg`` and ``--info`` take their value as a separate token).
_PASSTHROUGH_FLAGS = frozenset(
    {"-h", "--help", "--hydra-help", "--cfg", "-c", "--info", "-i", "--shell-completion", "-sc"}
)


def require_dotlist_args(prog: str, required: dict[str, str], *, local_only: tuple[str, ...] = ()) -> None:
    """Exit 1 with a usage line unless every required ``key=`` argument is present.

    ``local_only`` keys are additionally rejected when their argv value is a URL.
    This duplicates the authoritative in-config validation (``user_local_path``) for
    one reason: Hydra's run dir is anchored at ``${output_dir}/...`` and is created
    BEFORE the task function runs, so a URL-shaped ``output_dir`` would otherwise
    make Hydra itself materialize a literal ``s3:/...`` directory tree (#213) before
    any validation could fire.

    Args:
        prog: Console-script name for the usage message.
        required: Maps each required dotlist key to the placeholder shown in the
            usage line (e.g. ``{"spec": "<name|pkg://…|/path|./path>"}``).
        local_only: Dotlist keys whose values must be local paths, not URLs.

    Examples:
        >>> import pytest
        >>> argv = sys.argv
        >>> sys.argv = ["meds-extract-run"]
        >>> with pytest.raises(SystemExit) as exc_info:
        ...     require_dotlist_args("meds-extract-run", {"spec": "<spec>", "output_dir": "<dir>"})
        >>> exc_info.value.code
        1
        >>> sys.argv = ["meds-extract-run", "spec=X", "output_dir=/tmp/o", "dataset_key=demo"]
        >>> require_dotlist_args("meds-extract-run", {"spec": "<spec>", "output_dir": "<dir>"})
        >>> sys.argv = ["meds-extract-run", "spec=X", "output_dir=s3://bucket/raw"]
        >>> with pytest.raises(SystemExit) as exc_info:
        ...     require_dotlist_args(
        ...         "meds-extract-run", {"spec": "<spec>", "output_dir": "<dir>"},
        ...         local_only=("output_dir",),
        ...     )
        >>> exc_info.value.code
        1
        >>> sys.argv = ["meds-extract-run", "--help"]
        >>> require_dotlist_args("meds-extract-run", {"spec": "<spec>", "output_dir": "<dir>"})
        >>> sys.argv = argv
    """
    args = sys.argv[1:]
    if any(a in _PASSTHROUGH_FLAGS for a in args):
        return
    given = dict(a.split("=", 1) for a in args if "=" in a)
    missing = [k for k in required if k not in given]
    if missing:
        usage = " ".join(f"{k}={placeholder}" for k, placeholder in required.items())
        print(
            f"{prog}: missing required argument(s): {', '.join(f'{k}=' for k in missing)}\n"
            f"usage: {prog} {usage} [key=value ...]   (see {prog} --help for all options)",
            file=sys.stderr,
        )
        sys.exit(1)
    for k in local_only:
        if k in given and "://" in given[k]:
            print(
                f"{prog}: {k}={given[k]!r} is a URL, but {k} must be a local filesystem path. "
                "Remote data enters through the download layer (e.g. a `type: fsspec` source in "
                f"the spec's `sources:` block): fetch to local disk first, then point {k} at the "
                "local copy.",
                file=sys.stderr,
            )
            sys.exit(1)
