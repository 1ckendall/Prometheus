#!/usr/bin/env python3
"""CLI for building thermodynamic databases from raw source files.

Entry point registered in pyproject.toml as ``Prometheus-build``.

Subcommands
-----------
``nasa7``
    Parse a Burcat7 .thr file → ``nasa7.json``
``nasa9``
    Parse Burcat9 .thr and/or CEA thermo.inp → unified ``nasa9.json``
``janaf``
    Parse a JANAF .jnf file → ``janaf.csv``
``shomate``
    Validate and merge a hand-authored ``shomate.json``
``all``
    Run all of the above from a source directory

Modes
-----
``--mode overwrite``
    Rebuild from scratch; existing output is replaced.
``--mode append``
    Add new species only; keep existing records on conflict (default for nasa9).
``--mode interactive``
    Prompt for each conflicting species (default for ``all``).

Source labels
-------------
Every subcommand asks for a ``--source`` label (or ``--burcat-source`` /
``--cea-source`` for the nasa9 dual-source case).  If the flag is omitted
the script **prompts** for it interactively, because the label is embedded
in every compiled record and is shown during conflict resolution.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAW = os.path.normpath(os.path.join(_HERE, "..", "thermo_data", "raw"))
_OUT = os.path.normpath(os.path.join(_HERE, "..", "thermo_data"))


# ---------------------------------------------------------------------------
# Source-label prompt helper
# ---------------------------------------------------------------------------


def _require_source(flag_value: str | None, prompt_name: str) -> str:
    """Return *flag_value* if provided, otherwise ask the user interactively.

    Args:
        flag_value:  Value of the ``--source`` (or similar) CLI flag, or None.
        prompt_name: Human-readable name shown in the prompt,
                     e.g. ``"Burcat9"`` or ``"CEA"``.

    Returns:
        A non-empty source label string.
    """
    if flag_value:
        return flag_value
    while True:
        label = input(
            f"Enter a label for the {prompt_name} source\n"
            f"  (e.g. 'Burcat-2024', 'CEA-NRL2002', 'JANAF-4th-Ed'): "
        ).strip()
        if label:
            return label
        print("  Label cannot be empty — please try again.")


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_nasa7(args: argparse.Namespace) -> None:
    from prometheus.thermo_data.compiler import ThermoCompiler

    source = _require_source(args.source, "Burcat7 / NASA-7")
    ThermoCompiler().compile_nasa7(args.src, args.out, source=source, mode=args.mode)


def _cmd_nasa9(args: argparse.Namespace) -> None:
    from prometheus.thermo_data.compiler import ThermoCompiler

    burcat_source = cea_source = None

    if args.burcat and os.path.exists(args.burcat):
        burcat_source = _require_source(args.burcat_source, "Burcat9 / NASA-9")

    if args.cea and os.path.exists(args.cea):
        cea_source = _require_source(args.cea_source, "CEA / NASA-9")

    if burcat_source is None and cea_source is None:
        sys.exit(
            "error: at least one of --burcat or --cea must point to an existing file."
        )

    ThermoCompiler().compile_nasa9(
        args.out,
        burcat_src=args.burcat if burcat_source else None,
        burcat_source=burcat_source or "",
        cea_src=args.cea if cea_source else None,
        cea_source=cea_source or "",
        mode=args.mode,
    )


def _cmd_janaf(args: argparse.Namespace) -> None:
    from prometheus.thermo_data.compiler import ThermoCompiler

    source = _require_source(args.source, "JANAF")
    ThermoCompiler().compile_janaf(args.src, args.out, source=source, mode=args.mode)


def _cmd_shomate(args: argparse.Namespace) -> None:
    from prometheus.thermo_data.compiler import ThermoCompiler

    ThermoCompiler().compile_shomate(args.src, args.out, mode=args.mode)


def _cmd_all(args: argparse.Namespace) -> None:
    from prometheus.thermo_data.compiler import ThermoCompiler

    src_dir = args.src_dir

    # Prompt for each source label only if the corresponding file exists.
    def _maybe_source(filename: str, name: str, provided: str | None) -> str:
        if os.path.exists(os.path.join(src_dir, filename)):
            return _require_source(provided, name)
        return ""

    burcat_source = _maybe_source(
        "burcat7.thr", "Burcat / NASA-7 and NASA-9", args.burcat_source
    )
    cea_source = _maybe_source("cea_thermo.inp", "CEA / NASA-9", args.cea_source)
    janaf_source = _maybe_source("JANAF.jnf", "JANAF", args.janaf_source)

    ThermoCompiler().compile_all(
        src_dir,
        args.out_dir,
        burcat_source=burcat_source,
        cea_source=cea_source,
        janaf_source=janaf_source,
        mode=args.mode,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Prometheus-build",
        description="Build thermodynamic databases from raw source files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")

    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # ---- nasa7 ----
    p7 = sub.add_parser("nasa7", help="Build nasa7.json from burcat7.thr")
    p7.add_argument(
        "--src",
        default=os.path.join(_RAW, "burcat7.thr"),
        metavar="FILE",
        help="Input burcat7.thr (default: thermo_data/raw/burcat7.thr)",
    )
    p7.add_argument(
        "--out",
        default=os.path.join(_OUT, "nasa7.json"),
        metavar="FILE",
        help="Output nasa7.json (default: thermo_data/nasa7.json)",
    )
    p7.add_argument(
        "--mode", choices=["overwrite", "append", "interactive"], default="overwrite"
    )
    p7.add_argument(
        "--source",
        metavar="LABEL",
        default=None,
        help="Source label (prompted if omitted)",
    )
    p7.set_defaults(func=_cmd_nasa7)

    # ---- nasa9 ----
    p9 = sub.add_parser(
        "nasa9", help="Build nasa9.json from burcat9.thr and/or cea_thermo.inp"
    )
    p9.add_argument(
        "--burcat", default=os.path.join(_RAW, "burcat9.thr"), metavar="FILE"
    )
    p9.add_argument(
        "--cea", default=os.path.join(_RAW, "cea_thermo.inp"), metavar="FILE"
    )
    p9.add_argument("--out", default=os.path.join(_OUT, "nasa9.json"), metavar="FILE")
    p9.add_argument(
        "--mode", choices=["overwrite", "append", "interactive"], default="append"
    )
    p9.add_argument(
        "--burcat-source",
        metavar="LABEL",
        default=None,
        dest="burcat_source",
        help="Label for the Burcat9 source (prompted if omitted)",
    )
    p9.add_argument(
        "--cea-source",
        metavar="LABEL",
        default=None,
        dest="cea_source",
        help="Label for the CEA source (prompted if omitted)",
    )
    p9.set_defaults(func=_cmd_nasa9)

    # ---- janaf ----
    pj = sub.add_parser("janaf", help="Build janaf.csv from JANAF.jnf")
    pj.add_argument("--src", default=os.path.join(_RAW, "JANAF.jnf"), metavar="FILE")
    pj.add_argument("--out", default=os.path.join(_OUT, "janaf.csv"), metavar="FILE")
    pj.add_argument(
        "--mode", choices=["overwrite", "append", "interactive"], default="overwrite"
    )
    pj.add_argument(
        "--source",
        metavar="LABEL",
        default=None,
        help="Source label (prompted if omitted)",
    )
    pj.set_defaults(func=_cmd_janaf)

    # ---- shomate ----
    ps = sub.add_parser("shomate", help="Validate/merge shomate.json")
    ps.add_argument("--src", default=os.path.join(_OUT, "shomate.json"), metavar="FILE")
    ps.add_argument("--out", default=os.path.join(_OUT, "shomate.json"), metavar="FILE")
    ps.add_argument(
        "--mode", choices=["overwrite", "append", "interactive"], default="append"
    )
    ps.set_defaults(func=_cmd_shomate)

    # ---- all ----
    pa = sub.add_parser("all", help="Build all databases from a raw source directory")
    pa.add_argument("--src-dir", default=_RAW, metavar="DIR", dest="src_dir")
    pa.add_argument("--out-dir", default=_OUT, metavar="DIR", dest="out_dir")
    pa.add_argument(
        "--mode", choices=["overwrite", "append", "interactive"], default="interactive"
    )
    pa.add_argument(
        "--burcat-source", metavar="LABEL", default=None, dest="burcat_source"
    )
    pa.add_argument("--cea-source", metavar="LABEL", default=None, dest="cea_source")
    pa.add_argument(
        "--janaf-source", metavar="LABEL", default=None, dest="janaf_source"
    )
    pa.set_defaults(func=_cmd_all)

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
