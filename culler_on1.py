#!/usr/bin/env python3
"""Photo culler for ON1 Photo RAW.

Updates the .on1 sidecars ON1 has already created, so keywords, descriptions and the
culling verdict show up in the catalogue. Everything else lives in cli.py.
"""

import sys

import click

try:
    from .cli import common_options, run_cull
except ImportError:
    from cli import common_options, run_cull


@click.command()
@common_options
def cull_on1(**kwargs):
    """Analyse a folder and write the results into ON1 .on1 sidecars.

    \b
    Examples:
      python culler_on1.py ~/Photos/Shoot
      python culler_on1.py ~/Photos/Shoot --dry-run --detail
      python culler_on1.py ~/Photos/Shoot --fast
      python culler_on1.py ~/Photos/Shoot --model qwen3-vl:30b-a3b-instruct

    \b
    ON1 has to have created a .on1 file for a photo before its metadata can be updated;
    browse the folder in ON1 once first. Restart ON1 afterwards to see the keywords.
    """
    sys.exit(run_cull("on1", **kwargs))


if __name__ == "__main__":
    cull_on1()
