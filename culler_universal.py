#!/usr/bin/env python3
"""Photo culler writing standard XMP sidecars.

Works with Lightroom, Bridge, Capture One and ON1 alike. Everything else lives in cli.py.
"""

import sys

import click

try:
    from .cli import common_options, run_cull
except ImportError:
    from cli import common_options, run_cull


@click.command()
@common_options
def cull_universal(**kwargs):
    """Analyse a folder and write the results into .xmp sidecars.

    \b
    Examples:
      python culler_universal.py ~/Photos/Shoot
      python culler_universal.py ~/Photos/Shoot --dry-run --detail
      python culler_universal.py ~/Photos/Shoot --fast
      python culler_universal.py ~/Photos/Shoot --no-tags --workers 8

    \b
    Sidecars are written as photo.NEF.xmp next to each file. Existing ratings, keywords
    and descriptions are preserved unless --override is passed.
    """
    sys.exit(run_cull("xmp", **kwargs))


if __name__ == "__main__":
    cull_universal()
