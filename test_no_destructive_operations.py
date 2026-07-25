#!/usr/bin/env python3
"""Guarantee the culler cannot move, rename or remove a photograph.

A culling tool has exactly one job it must never do. Documenting that in a README is
not a guarantee; this is. The check reads the source of every module and fails if a
destructive filesystem call appears anywhere in the package.

If you are adding a feature and this test fails, that is the test working. Do not add
an exemption for photo files -- write the verdict to a sidecar and let the photographer
act on it in their catalogue, where it can be undone.

Run with pytest, or directly: python test_no_destructive_operations.py
"""

import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).parent

# Modules exempt from the scan, with the reason. Keep this list tiny and justified.
EXEMPT = {
    # Builds throwaway test images in a directory it creates itself under the system
    # temp dir, and cleans that directory up. It never touches the photographs it reads.
    "eval_harness.py",
    "test_no_destructive_operations.py",
}

# Calls that can destroy or relocate a file on disk.
BANNED_ATTRS = {
    "rename",       # Path.rename / os.rename
    "replace",      # Path.replace -- silently overwrites the destination
    "unlink",       # Path.unlink
    "rmdir",
    "remove",       # os.remove
    "removedirs",
    "rmtree",       # shutil.rmtree
    "move",         # shutil.move
}

# str.replace and dict/list .remove are extremely common and harmless. We only care
# about these names when they are called on something filesystem-shaped, so a call is
# reported when the receiver looks like a path or a filesystem module.
FILESYSTEM_RECEIVERS = {"os", "shutil", "path", "filepath", "file", "dest", "destination",
                        "target", "src", "source", "p", "f", "sidecar", "companion",
                        "folder", "directory", "dir", "trash"}


def _receiver_name(node: ast.Attribute) -> str:
    value = node.value
    if isinstance(value, ast.Name):
        return value.id.lower()
    if isinstance(value, ast.Attribute):
        return value.attr.lower()
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return value.func.id.lower()
    return ""


def scan(source_path: Path):
    """Return a list of (line, description) for banned calls in one file."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    findings = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in BANNED_ATTRS:
            continue

        receiver = _receiver_name(func)

        # str.replace("a", "b") is not a filesystem operation. Path.replace(other) is.
        # Distinguish by receiver name, and by string-literal arguments.
        if func.attr == "replace":
            args_are_strings = node.args and all(
                isinstance(a, ast.Constant) and isinstance(a.value, str) for a in node.args
            )
            if args_are_strings or receiver not in FILESYSTEM_RECEIVERS:
                continue

        if func.attr == "remove" and receiver not in FILESYSTEM_RECEIVERS:
            continue  # list.remove(x)

        findings.append((node.lineno, f"{receiver}.{func.attr}()" if receiver else f"{func.attr}()"))

    return findings


def test_no_destructive_filesystem_calls():
    offenders = {}

    for source_path in sorted(PACKAGE.glob("*.py")):
        if source_path.name in EXEMPT:
            continue
        findings = scan(source_path)
        if findings:
            offenders[source_path.name] = findings

    assert not offenders, (
        "This tool must never move, rename or remove a photograph. Found:\n"
        + "\n".join(
            f"  {name}:{line}  {what}"
            for name, findings in offenders.items()
            for line, what in findings
        )
    )


def test_no_move_deletes_flag():
    """The old --move-deletes flag must stay gone."""
    cli_source = (PACKAGE / "cli.py").read_text(encoding="utf-8")
    assert "move-deletes" not in cli_source
    assert "move_deletions" not in cli_source
    assert "_culled_deletes" not in cli_source


def test_delete_is_only_ever_a_keyword():
    """The Delete verdict must reach the photo as metadata and nothing else."""
    sidecars = (PACKAGE / "sidecars.py").read_text(encoding="utf-8")
    assert "PhotoCuller:{result.decision}" in sidecars

    for name in ("cli.py", "batch.py", "decision.py", "grouping.py", "sidecars.py"):
        source = (PACKAGE / name).read_text(encoding="utf-8")
        assert "send2trash" not in source
        assert "Trash" not in source


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []

    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}\n{e}")
            failures.append(test.__name__)

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
