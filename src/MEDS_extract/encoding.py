"""Text encodings of csv sources: the ``etl.encoding`` option, detection, and transcoding.

Polars streams only UTF-8 csv, so a source in another encoding is decoded here into a
UTF-8 copy on disk (:func:`transcode_to_utf8`) before it is scanned. Under the default
``etl.encoding: auto`` its codec is detected first (:func:`detect_encoding`).
"""

from __future__ import annotations

import codecs
import gzip
import logging
import re
import shutil
import uuid
from collections.abc import Mapping
from contextlib import ExitStack
from io import TextIOWrapper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from upath import UPath

logger = logging.getLogger(__name__)


def is_utf8(encoding: str) -> bool:
    """Whether ``encoding`` names UTF-8, under any of Python's aliases for it.

    Raises ``LookupError`` for a name Python does not know.

    Examples:
        >>> is_utf8("utf8"), is_utf8("UTF-8"), is_utf8("latin-1"), is_utf8("cp1252")
        (True, True, False, False)
        >>> is_utf8("not-a-codec")
        Traceback (most recent call last):
            ...
        LookupError: unknown encoding: not-a-codec
    """
    return codecs.lookup(encoding).name == "utf-8"


# The ``etl.encoding`` value (and default) under which each csv source's codec is
# detected rather than declared.
AUTO_ENCODING = "auto"


def source_encoding(encoding: str | Mapping[str, str], prefix: str) -> str:
    """The codec to decode source ``prefix`` with, from the ``etl.encoding`` option.

    The option is either one codec for every source or a mapping of source prefix to
    codec; a prefix the mapping does not list is :data:`AUTO_ENCODING`, detected per file
    by :func:`resolve_encoding`.

    Examples:
        >>> source_encoding("latin-1", "labs")
        'latin-1'
        >>> source_encoding({"labs": "latin-1"}, "labs"), source_encoding({"labs": "latin-1"}, "vitals")
        ('latin-1', 'auto')
    """
    if isinstance(encoding, Mapping):
        return encoding.get(prefix, AUTO_ENCODING)
    return encoding


# Byte-order marks that name their codec outright. UTF-32 precedes UTF-16 because a
# UTF-32-LE mark begins with the UTF-16-LE one. A UTF-8 mark needs no entry: it is
# valid UTF-8, and polars strips it.
_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

# The five bytes cp1252 leaves undefined; a non-UTF-8 source containing any of them
# cannot be cp1252, so it falls back to latin-1, which defines every byte.
_CP1252_UNDEFINED = (b"\x81", b"\x8d", b"\x8f", b"\x90", b"\x9d")

# One well-formed multi-byte UTF-8 sequence (RFC 3629: no overlong forms, surrogates,
# or code points past U+10FFFF). A non-UTF-8 source that contains one is broken UTF-8.
_UTF8_MULTIBYTE = re.compile(
    rb"[\xc2-\xdf][\x80-\xbf]"
    rb"|\xe0[\xa0-\xbf][\x80-\xbf]|[\xe1-\xec\xee\xef][\x80-\xbf]{2}|\xed[\x80-\x9f][\x80-\xbf]"
    rb"|\xf0[\x90-\xbf][\x80-\xbf]{2}|[\xf1-\xf3][\x80-\xbf]{3}|\xf4[\x80-\x8f][\x80-\xbf]{2}"
)


def _open_bytes(src: Path | UPath, stack: ExitStack):
    """Open ``src`` for binary reading on ``stack``, transparently decompressing ``.gz``."""
    raw = stack.enter_context(src.open("rb"))
    if "".join(src.suffixes).lower().endswith(".gz"):
        raw = stack.enter_context(gzip.GzipFile(fileobj=raw))
    return raw


def detect_encoding(src: Path | UPath, *, chunk_bytes: int = 1 << 22) -> str:
    """Detect the text encoding of a csv / csv.gz file, deterministically.

    In order: a UTF-16/32 byte-order mark names its codec; a file that is valid UTF-8
    is UTF-8; any other file is cp1252, or latin-1 if it holds a byte cp1252 leaves
    undefined. Real non-ASCII text in another encoding almost never forms valid UTF-8,
    and cp1252 is what Western European exports use. General-purpose statistical
    detectors are deliberately avoided: on short code and unit strings they mistake
    cp1252 for cp1250, which garbles ``°`` and ``µ`` without raising. Other codecs
    (cp1250, Shift-JIS, ...) and BOM-less UTF-16 are not detected and must be
    declared. Memory use is bounded by ``chunk_bytes``.

    A file that is not valid UTF-8 but contains well-formed multi-byte UTF-8 sequences
    is not guessed at all: it is evidently UTF-8 with a stray byte or some rows in
    another encoding, and decoding it as cp1252 would garble every UTF-8 character in
    it. Rarely, cp1252 text forms such a sequence by chance (an accented capital or
    ``ß`` followed by a byte in 0x80-0xBF, e.g. ``ß“``); that file is rejected too, and
    its codec must be declared.

    Raises:
        ValueError: If ``src`` is not valid UTF-8 but contains valid multi-byte UTF-8
            sequences. The message names the offset of the first invalid byte.

    Examples:
        >>> import tempfile
        >>> text = "item,unit\\nTemperatuur,\\u00b0C\\nCaf\\u00e9\\u00efne,\\u00b5g\\n"
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "items.csv"
        ...     for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-32", "cp1252", "latin-1"):
        ...         _ = fp.write_bytes(text.encode(enc))
        ...         print(enc, "->", detect_encoding(fp))
        utf-8 -> utf8
        utf-8-sig -> utf8
        utf-16 -> utf-16
        utf-32 -> utf-32
        cp1252 -> cp1252
        latin-1 -> cp1252

        Latin-1 text that uses a byte cp1252 leaves undefined (here a C1 control
        character) is still decoded strictly, as latin-1:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "items.csv"
        ...     _ = fp.write_bytes("item\\nCaf\\u00e9\\x81\\n".encode("latin-1"))
        ...     detect_encoding(fp)
        'latin-1'

        A UTF-8 sequence split across a chunk boundary is still valid UTF-8:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "items.csv"
        ...     _ = fp.write_bytes("ab\\u00b5g".encode("utf-8"))
        ...     detect_encoding(fp, chunk_bytes=3)
        'utf8'

        Mixed UTF-8 and latin-1 fails loudly, in either order and across chunks:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "labs.csv"
        ...     _ = fp.write_bytes("unit\\nCaf\\u00e9\\n".encode("latin-1") + "\\u20ac\\n".encode("utf-8"))
        ...     detect_encoding(fp, chunk_bytes=1)
        Traceback (most recent call last):
            ...
        ValueError: ...labs.csv is not valid UTF-8 (byte 0xe9 at offset 8), but it contains
        valid multi-byte UTF-8 sequences, so it is mixed or corrupted UTF-8 ...
    """
    with ExitStack() as stack:
        raw = _open_bytes(src, stack)
        chunk = raw.read(chunk_bytes)
        for bom, codec in _BOMS:
            if chunk.startswith(bom):
                return codec
        decoder = codecs.getincrementaldecoder("utf-8")()
        # ``offset`` counts the bytes before ``chunk``; ``tail`` carries the 3 bytes before
        # it, so a multi-byte sequence split across chunks still matches.
        offset, tail = 0, b""
        invalid, multibyte, cp1252_undefined = None, False, False
        while True:
            final = not chunk
            if invalid is None:
                pending = len(decoder.getstate()[0])
                try:
                    decoder.decode(chunk, final=final)
                except UnicodeDecodeError as e:
                    invalid = (offset - pending + e.start, e.object[e.start])
            multibyte = multibyte or _UTF8_MULTIBYTE.search(tail + chunk) is not None
            if invalid is not None and multibyte:
                at, byte = invalid
                raise ValueError(
                    f"{src} is not valid UTF-8 (byte 0x{byte:02x} at offset {at}), but it contains "
                    f"valid multi-byte UTF-8 sequences, so it is mixed or corrupted UTF-8 and its "
                    f"encoding cannot be detected. Re-export it as UTF-8, or, if it is in fact a "
                    f"single-byte codec, declare it in the MESSY config, e.g. "
                    f"`etl: {{encoding: {{<prefix>: cp1252}}}}`."
                )
            cp1252_undefined = cp1252_undefined or any(b in chunk for b in _CP1252_UNDEFINED)
            if final:
                break
            offset, tail = offset + len(chunk), (tail + chunk)[-3:]
            chunk = raw.read(chunk_bytes)
    if invalid is None:
        return "utf8"
    return "latin-1" if cp1252_undefined else "cp1252"


def resolve_encoding(src: Path | UPath, encoding: str) -> str:
    """The codec to decode ``src`` with: ``encoding`` itself, unless it is ``"auto"``.

    Under :data:`AUTO_ENCODING` the codec is detected (:func:`detect_encoding`), and a
    detected non-UTF-8 codec is logged as a warning naming the file, so a wrong guess
    is visible and can be pinned in the MESSY config.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "items.csv"
        ...     _ = fp.write_bytes("item\\nCaf\\u00e9\\n".encode("cp1252"))
        ...     resolve_encoding(fp, "auto"), resolve_encoding(fp, "latin-1")
        ('cp1252', 'latin-1')
    """
    if encoding != AUTO_ENCODING:
        return encoding
    codec = detect_encoding(src)
    if not is_utf8(codec):
        logger.warning(
            f"{src} is not valid UTF-8; decoding it as detected encoding {codec}. If that is "
            f"wrong, declare the source's codec in the MESSY config, e.g. "
            f"`etl: {{encoding: {{<prefix>: <codec>}}}}`."
        )
    return codec


def transcode_to_utf8(src: Path | UPath, dest: Path, encoding: str, *, chunk_chars: int = 1 << 22) -> None:
    """Stream a csv / csv.gz file decoded as ``encoding`` into an uncompressed UTF-8 copy.

    Polars' CSV reader only understands UTF-8 (strictly, or with lossy replacement of
    invalid bytes), so a source in any other encoding is re-encoded on disk before it is
    scanned. Decoding is strict: a byte sequence that is invalid in ``encoding`` raises
    rather than being replaced, so nothing is silently corrupted. Memory use is bounded
    by ``chunk_chars``, and ``newline=""`` keeps line endings (including any inside quoted
    fields) byte-for-byte.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     src, dest = Path(d) / "items.csv", Path(d) / "items.utf8.csv"
        ...     text = "item,unit\\nTemperatuur,\\u00b0C\\nCaf\\u00e9\\u00efne,mg\\n"
        ...     _ = src.write_bytes(text.encode("latin-1"))
        ...     transcode_to_utf8(src, dest, "latin-1")
        ...     dest.read_bytes().decode("utf-8").splitlines()
        ['item,unit', 'Temperatuur,°C', 'Caféïne,mg']

        Bytes that are invalid in the declared encoding fail loudly:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     src = Path(d) / "items.csv"
        ...     _ = src.write_bytes(b"item\\n\\xff\\n")
        ...     transcode_to_utf8(src, Path(d) / "out.csv", "ascii")
        Traceback (most recent call last):
            ...
        UnicodeDecodeError: 'ascii' codec can't decode byte 0xff in position 5: ordinal not in range(128)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        raw = _open_bytes(src, stack)
        fin = stack.enter_context(TextIOWrapper(raw, encoding=encoding, newline=""))
        fout = stack.enter_context(dest.open("w", encoding="utf-8", newline=""))
        shutil.copyfileobj(fin, fout, chunk_chars)


def utf8_copy(src: Path | UPath, dest_dir: Path, encoding: str) -> Path:
    """Transcode a csv / csv.gz metadata source decoded as ``encoding`` into ``dest_dir``.

    Every worker writes the same deterministic copy, so each one transcodes into a
    private temp file and atomically renames it into place: a reader never sees a
    partial file, and the last writer wins harmlessly.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     src = Path(d) / "items.csv"
        ...     _ = src.write_bytes("item,unit\\n1,\\u00b5g\\n".encode("latin-1"))
        ...     out = utf8_copy(src, Path(d) / "utf8", "latin-1")
        ...     out.name, pl.read_csv(out)["unit"].to_list()
        ('items.csv', ['µg'])
    """
    name = src.name[: -len(".gz")] if src.name.lower().endswith(".gz") else src.name
    dest = dest_dir / name
    tmp = dest_dir / f".{name}.{uuid.uuid4().hex}.tmp"
    transcode_to_utf8(src, tmp, encoding)
    tmp.replace(dest)
    return dest
