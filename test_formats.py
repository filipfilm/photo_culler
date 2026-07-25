#!/usr/bin/env python3
"""Tests for which file formats the culler accepts.

The list of extensions lives in two places -- batch.py for the library default and
config.py for the shipped config.yaml default -- and a format added to one but not the
other fails silently: the files are simply never picked up, and the run reports a
smaller folder than you have. These tests keep the two honest.

Run with pytest, or directly: python test_formats.py
"""

from pathlib import Path
import sys

try:
    from .batch import DEFAULT_EXTENSIONS as BATCH_EXTENSIONS
    from .config import DEFAULT_EXTENSIONS as CONFIG_EXTENSIONS, Config
    from .extractor import HEIF_EXTENSIONS, RAW_EXTENSIONS
except ImportError:
    from batch import DEFAULT_EXTENSIONS as BATCH_EXTENSIONS
    from config import DEFAULT_EXTENSIONS as CONFIG_EXTENSIONS, Config
    from extractor import HEIF_EXTENSIONS, RAW_EXTENSIONS

PACKAGE = Path(__file__).parent


def test_extension_defaults_agree():
    """batch.py and config.py must offer the same formats."""
    assert set(BATCH_EXTENSIONS) == set(CONFIG_EXTENSIONS), (
        "DEFAULT_EXTENSIONS differ between modules:\n"
        f"  only in batch.py : {sorted(set(BATCH_EXTENSIONS) - set(CONFIG_EXTENSIONS))}\n"
        f"  only in config.py: {sorted(set(CONFIG_EXTENSIONS) - set(BATCH_EXTENSIONS))}"
    )


def test_shipped_config_agrees_with_code():
    """config.yaml as shipped must not silently drop a supported format."""
    config = Config.load(PACKAGE / "config.yaml")
    assert set(config.extensions) == set(BATCH_EXTENSIONS), (
        "config.yaml extensions differ from the code default:\n"
        f"  only in config.yaml: {sorted(set(config.extensions) - set(BATCH_EXTENSIONS))}\n"
        f"  only in code       : {sorted(set(BATCH_EXTENSIONS) - set(config.extensions))}"
    )


def test_heif_is_accepted():
    """iPhone photographs must be picked up by default."""
    for extension in (".heic", ".heif"):
        assert extension in BATCH_EXTENSIONS
        assert extension in HEIF_EXTENSIONS


def test_heif_is_not_treated_as_raw():
    """HEIF goes through Pillow, not rawpy; overlap would route it to the wrong decoder."""
    assert not (HEIF_EXTENSIONS & RAW_EXTENSIONS)


def test_extensions_are_lowercase_and_dotted():
    """find_image_files lowercases the suffix it compares, so the list must match."""
    for extension in set(BATCH_EXTENSIONS) | set(CONFIG_EXTENSIONS):
        assert extension.startswith("."), f"{extension} is missing its leading dot"
        assert extension == extension.lower(), f"{extension} is not lowercase"


def test_uppercase_files_are_found(tmp_path=None):
    """Cameras write IMG_1234.HEIC and DSC_0001.NEF; matching is case-insensitive."""
    import tempfile

    try:
        from .batch import BatchCuller
    except ImportError:
        from batch import BatchCuller

    with tempfile.TemporaryDirectory() as raw_dir:
        folder = Path(raw_dir)
        for name in ("a.HEIC", "b.heic", "c.NEF", "d.JPG", "ignore.txt", "movie.MOV"):
            (folder / name).write_bytes(b"")

        culler = BatchCuller.__new__(BatchCuller)  # no Ollama needed for path matching
        found = {p.name for p in culler.find_image_files(folder, BATCH_EXTENSIONS)}

    assert found == {"a.HEIC", "b.heic", "c.NEF", "d.JPG"}, f"got {found}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []

    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}\n        {e}")
            failures.append(test.__name__)
        except Exception as e:
            print(f"  ERROR {test.__name__}: {e}")
            failures.append(test.__name__)

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
