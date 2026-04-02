"""ThermoCompiler — build and merge thermodynamic JSON/CSV databases.

Typical workflow::

    from prometheus_equilibrium.thermo_data.compiler import ThermoCompiler

    c = ThermoCompiler()
    c.compile_nasa7("raw/burcat7.thr",  "nasa7.json", source="Burcat-2024")
    c.compile_nasa9("nasa9.json",
                    burcat_src="raw/burcat9.thr",  burcat_source="Burcat-2024",
                    cea_src="raw/cea_thermo.inp",  cea_source="CEA-2026",
                    mode="append")
    c.compile_janaf("raw/JANAF.jnf",   "janaf.csv",  source="JANAF-4th-Ed")

Merge modes
-----------
``overwrite``
    Rebuild the output file from scratch; the existing file is ignored.
``append``
    Add new species to the existing file.  When a species ID already exists,
    keep the existing record (the incoming one is discarded).
``interactive``
    Same as *append*, but when a conflict is detected the user is shown a
    summary of both records and asked which to keep.
"""

from __future__ import annotations

import csv
import difflib
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from .parsers import Burcat7Parser, Burcat9Parser, CEAParser, JANAFParser, ShomateParser
from .parsers.janaf import JANAF_COLUMNS

log = logging.getLogger(__name__)

# Tolerance for deciding whether two floating-point coefficients are equal.
_COEFF_TOL = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(db: Dict[str, dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d records to %s", len(db), path)


def _write_csv(rows: List[List[str]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    log.info("Wrote %d data rows to %s", max(0, len(rows) - 1), path)


def _record_summary(sp_id: str, rec: dict) -> str:
    """One-line summary of a species record for conflict display."""
    source = rec.get("source") or "?"
    fmt = rec.get("format", "?")
    if fmt == "NASA-9":
        segs = rec.get("segments", [])
        t_min = segs[0]["t_low"] if segs else "?"
        t_max = segs[-1]["t_high"] if segs else "?"
        return (
            f"[{source}]  NASA-9  {len(segs)} segments  " f"T=[{t_min}, …, {t_max}] K"
        )
    if fmt == "NASA-7":
        return (
            f"[{source}]  NASA-7  "
            f"T=[{rec.get('t_low','?')}, {rec.get('t_mid','?')}, "
            f"{rec.get('t_high','?')}] K"
        )
    return f"[{source}]  {fmt}"


def _max_coeff_delta(a: dict, b: dict) -> float:
    """Return the maximum absolute difference between any pair of coefficients."""

    def _flat_coeffs(rec: dict) -> List[float]:
        if rec.get("format") == "NASA-7":
            c = rec.get("coeffs", {})
            return c.get("low", []) + c.get("high", [])
        return [
            v
            for seg in rec.get("segments", [])
            for v in seg.get("coeffs", []) + [seg.get("b1", 0.0), seg.get("b2", 0.0)]
        ]

    ca, cb = _flat_coeffs(a), _flat_coeffs(b)
    if not ca or not cb or len(ca) != len(cb):
        return float("inf")
    return max(abs(x - y) for x, y in zip(ca, cb))


# ---------------------------------------------------------------------------
# ThermoCompiler
# ---------------------------------------------------------------------------


class ThermoCompiler:
    """Build and merge thermodynamic databases from raw source files.

    All ``compile_*`` methods accept a *mode* argument:

    * ``"overwrite"`` — rebuild from scratch, discarding any existing file.
    * ``"append"``    — add new species only; keep existing on conflict.
    * ``"interactive"`` — prompt the user for each conflicting species.
    """

    # ------------------------------------------------------------------
    # Public compile methods
    # ------------------------------------------------------------------

    def compile_nasa7(
        self,
        src: str,
        out: str,
        source: str,
        mode: str = "overwrite",
    ) -> int:
        """Build (or update) *out* from a single Burcat7 ``.thr`` source file.

        Args:
            src:    Path to the ``burcat7.thr`` file.
            out:    Path to the output ``nasa7.json``.
            source: Label for this data source, embedded in every record.
            mode:   ``"overwrite"`` / ``"append"`` / ``"interactive"``.

        Returns:
            Total number of species in the written file.
        """
        log.info("Parsing NASA-7: %s  [source: %s]", src, source)
        incoming = Burcat7Parser().parse(src, source=source)

        if mode == "overwrite" or not os.path.exists(out):
            result = incoming
        else:
            existing = _load_json(out)
            result, added, updated = self._merge_json(
                existing, incoming, mode, label="NASA-7"
            )
            log.info("NASA-7 merge: %d added, %d updated.", added, updated)

        _write_json(result, out)
        return len(result)

    def compile_nasa9(
        self,
        out: str,
        *,
        burcat_src: Optional[str] = None,
        burcat_source: str = "",
        cea_src: Optional[str] = None,
        cea_source: str = "",
        mode: str = "append",
    ) -> int:
        """Build (or update) *out* from Burcat9 and/or CEA source files.

        When both sources are provided they are merged before touching the
        existing output file.  Within that internal merge, CEA takes
        priority: CEA species override any conflicting BURCAT9 record so that
        well-validated CEA ground-state data is not shadowed by BURCAT9 exotic
        isomers or excited states that share the same canonical ID.  BURCAT9
        species absent from CEA are kept as supplementary data.

        Args:
            out:           Path to the output ``nasa9.json``.
            burcat_src:    Path to ``burcat9.thr`` (optional).
            burcat_source: Label for the Burcat9 source.
            cea_src:       Path to ``cea_thermo.inp`` (optional).
            cea_source:    Label for the CEA source.
            mode:          How to handle conflicts with the *existing* output file.

        Returns:
            Total number of species in the written file.
        """
        # 1. Parse each raw source.
        new_db: Dict[str, dict] = {}

        if burcat_src:
            log.info(
                "Parsing NASA-9 (Burcat9): %s  [source: %s]", burcat_src, burcat_source
            )
            new_db.update(Burcat9Parser().parse(burcat_src, source=burcat_source))

        if cea_src:
            log.info("Parsing NASA-9 (CEA): %s  [source: %s]", cea_src, cea_source)
            cea_db = CEAParser().parse(cea_src, source=cea_source)
            # Internal merge: CEA wins on conflict (CEA ground-state data takes
            # priority over BURCAT9 exotic isomers/excited states that may have
            # claimed the same canonical ID).  BURCAT9 species not in CEA are kept.
            cea_added = cea_replaced = 0
            for sp_id, rec in cea_db.items():
                if sp_id not in new_db:
                    cea_added += 1
                else:
                    cea_replaced += 1
                new_db[sp_id] = rec
            log.info(
                "CEA → Burcat9 merge: %d new species added, %d Burcat9 entries replaced by CEA.",
                cea_added,
                cea_replaced,
            )

        if not new_db:
            log.warning("compile_nasa9: no source files provided or parseable.")
            return 0

        # 2. Merge against existing output file (if any).
        if mode == "overwrite" or not os.path.exists(out):
            result = new_db
        else:
            existing = _load_json(out)
            result, added, updated = self._merge_json(
                existing, new_db, mode, label="NASA-9"
            )
            log.info("NASA-9 merge: %d added, %d updated.", added, updated)

        _write_json(result, out)
        return len(result)

    def compile_janaf(
        self,
        src: str,
        out: str,
        source: str,
        mode: str = "overwrite",
    ) -> int:
        """Build (or update) *out* from a JANAF ``.jnf`` source file.

        Args:
            src:    Path to ``JANAF.jnf``.
            out:    Path to the output ``janaf.csv``.
            source: Label for this data source.
            mode:   ``"overwrite"`` / ``"append"`` / ``"interactive"``.

        Returns:
            Number of data rows written (excluding the header).
        """
        log.info("Parsing JANAF: %s  [source: %s]", src, source)
        incoming_rows = JANAFParser().parse(src, source=source)  # row[0] = header

        if mode == "overwrite" or not os.path.exists(out):
            rows = incoming_rows
        else:
            rows = self._merge_csv(out, incoming_rows, mode, label="JANAF")

        _write_csv(rows, out)
        return max(0, len(rows) - 1)

    def compile_shomate(
        self,
        src: str,
        out: str,
        mode: str = "append",
    ) -> int:
        """Load, validate, and merge a ``shomate.json`` file.

        Because Shomate data is hand-authored there is no separate "raw"
        source — the JSON *is* the source.  This method validates the file
        and merges it into *out* (which may be the same path).

        Args:
            src:  Path to the source ``shomate.json``.
            out:  Path to the output ``shomate.json`` (may equal *src*).
            mode: ``"overwrite"`` / ``"append"`` / ``"interactive"``.

        Returns:
            Total number of species in the written file.
        """
        log.info("Loading Shomate: %s", src)
        incoming = ShomateParser().parse(src)

        if mode == "overwrite" or not os.path.exists(out) or src == out:
            result = incoming
        else:
            existing = _load_json(out)
            result, added, updated = self._merge_json(
                existing, incoming, mode, label="Shomate"
            )
            log.info("Shomate merge: %d added, %d updated.", added, updated)

        _write_json(result, out)
        return len(result)

    def compile_all(
        self,
        src_dir: str,
        out_dir: str,
        *,
        burcat_source: str = "",
        cea_source: str = "",
        janaf_source: str = "",
        mode: str = "interactive",
    ) -> None:
        """Build all databases from *src_dir* into *out_dir*.

        Looks for ``burcat7.thr``, ``burcat9.thr``, ``cea_thermo.inp``,
        and ``JANAF.jnf`` in *src_dir*.  Missing files are skipped with a
        warning.

        Args:
            src_dir:       Directory containing raw source files.
            out_dir:       Directory to write compiled databases.
            burcat_source: Label for Burcat7 and Burcat9 data.
            cea_source:    Label for CEA data.
            janaf_source:  Label for JANAF data.
            mode:          Merge mode applied to all databases.
        """

        def _src(name: str) -> Optional[str]:
            p = os.path.join(src_dir, name)
            if os.path.exists(p):
                return p
            log.warning("Source file not found, skipping: %s", p)
            return None

        b7 = _src("burcat7.thr")
        b9 = _src("burcat9.thr")
        cea = _src("cea_thermo.inp")
        jnf = _src("JANAF.jnf")

        if b7:
            self.compile_nasa7(
                b7, os.path.join(out_dir, "nasa7.json"), source=burcat_source, mode=mode
            )
        if b9 or cea:
            self.compile_nasa9(
                os.path.join(out_dir, "nasa9.json"),
                burcat_src=b9,
                burcat_source=burcat_source,
                cea_src=cea,
                cea_source=cea_source,
                mode=mode,
            )
        if jnf:
            self.compile_janaf(
                jnf, os.path.join(out_dir, "janaf.csv"), source=janaf_source, mode=mode
            )

    # ------------------------------------------------------------------
    # JSON merge
    # ------------------------------------------------------------------

    def _merge_json(
        self,
        existing: Dict[str, dict],
        incoming: Dict[str, dict],
        mode: str,
        label: str = "",
    ) -> Tuple[Dict[str, dict], int, int]:
        """Merge *incoming* into *existing* according to *mode*.

        Returns:
            ``(result, n_added, n_updated)`` where *result* is the merged dict,
            *n_added* is the count of new species, and *n_updated* is the count
            of species whose record was replaced by the incoming version.
        """
        result = dict(existing)
        n_added = n_updated = 0

        for sp_id, new_rec in incoming.items():
            if sp_id not in result:
                result[sp_id] = new_rec
                n_added += 1
                continue

            old_rec = result[sp_id]
            if self._records_equal(old_rec, new_rec):
                continue  # identical — nothing to do

            # Conflict
            if mode == "append":
                pass  # keep existing silently

            elif mode == "overwrite":
                result[sp_id] = new_rec
                n_updated += 1

            elif mode == "interactive":
                winner = self._resolve_conflict_interactive(
                    sp_id, old_rec, new_rec, label
                )
                if winner is not old_rec:
                    result[sp_id] = winner
                    n_updated += 1

        return result, n_added, n_updated

    # ------------------------------------------------------------------
    # CSV merge (JANAF)
    # ------------------------------------------------------------------

    def _merge_csv(
        self,
        existing_path: str,
        incoming_rows: List[List[str]],
        mode: str,
        label: str = "",
    ) -> List[List[str]]:
        """Merge incoming CSV rows into an existing CSV file.

        Keyed by the first column (species ID).  Whole-species granularity:
        all temperature rows for a given ID are kept or replaced together.

        Returns:
            Merged row list (including header row).
        """
        # Load existing file grouped by species ID
        existing_species: Dict[str, List[List[str]]] = {}
        existing_order: List[str] = []
        try:
            with open(existing_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if not row:
                        continue
                    sp_id = row[0]
                    if sp_id not in existing_species:
                        existing_species[sp_id] = []
                        existing_order.append(sp_id)
                    existing_species[sp_id].append(row)
        except FileNotFoundError:
            pass

        # Group incoming rows by species ID (skip header)
        incoming_species: Dict[str, List[List[str]]] = {}
        incoming_order: List[str] = []
        for row in incoming_rows[1:]:
            if not row:
                continue
            sp_id = row[0]
            if sp_id not in incoming_species:
                incoming_species[sp_id] = []
                incoming_order.append(sp_id)
            incoming_species[sp_id].append(row)

        result_species: Dict[str, List[List[str]]] = dict(existing_species)
        result_order: List[str] = list(existing_order)
        n_added = n_updated = 0

        for sp_id in incoming_order:
            new_rows = incoming_species[sp_id]
            if sp_id not in result_species:
                result_species[sp_id] = new_rows
                result_order.append(sp_id)
                n_added += 1
                continue

            old_rows = result_species[sp_id]
            if old_rows == new_rows:
                continue

            if mode == "append":
                pass  # keep existing

            elif mode == "overwrite":
                result_species[sp_id] = new_rows
                n_updated += 1

            elif mode == "interactive":
                winner = self._resolve_conflict_csv_interactive(
                    sp_id, old_rows, new_rows
                )
                if winner is not old_rows:
                    result_species[sp_id] = winner
                    n_updated += 1

        log.info("%s CSV merge: %d added, %d updated.", label, n_added, n_updated)

        # Reconstruct rows in order
        out = [JANAF_COLUMNS]
        for sp_id in result_order:
            out.extend(result_species[sp_id])
        return out

    # ------------------------------------------------------------------
    # Equality check
    # ------------------------------------------------------------------

    @staticmethod
    def _records_equal(a: dict, b: dict, tol: float = _COEFF_TOL) -> bool:
        """Return True if two NASA-7/9 records are effectively identical."""
        if a.get("format") != b.get("format"):
            return False
        if a.get("elements") != b.get("elements"):
            return False

        def _near(x: float, y: float) -> bool:
            return abs(x - y) <= tol

        if a.get("format") == "NASA-7":
            for side in ("low", "high"):
                ca = a.get("coeffs", {}).get(side, [])
                cb = b.get("coeffs", {}).get(side, [])
                if len(ca) != len(cb) or not all(_near(x, y) for x, y in zip(ca, cb)):
                    return False
            return (
                _near(a.get("t_low", 0.0), b.get("t_low", 0.0))
                and _near(a.get("t_mid", 0.0), b.get("t_mid", 0.0))
                and _near(a.get("t_high", 0.0), b.get("t_high", 0.0))
            )

        if a.get("format") == "NASA-9":
            sa, sb = a.get("segments", []), b.get("segments", [])
            if len(sa) != len(sb):
                return False
            for seg_a, seg_b in zip(sa, sb):
                if not (
                    _near(seg_a["t_low"], seg_b["t_low"])
                    and _near(seg_a["t_high"], seg_b["t_high"])
                ):
                    return False
                ca = seg_a.get("coeffs", [])
                cb = seg_b.get("coeffs", [])
                if len(ca) != len(cb) or not all(_near(x, y) for x, y in zip(ca, cb)):
                    return False
            return True

        return False  # unknown format — treat as different

    # ------------------------------------------------------------------
    # Interactive conflict resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_conflict_interactive(
        sp_id: str,
        old_rec: dict,
        new_rec: dict,
        label: str = "",
    ) -> dict:
        """Prompt the user to resolve a conflict between two JSON records.

        Returns either *old_rec* or *new_rec* (the object, not a copy).
        """
        header = f"\n{'─'*70}"
        if label:
            header += f"\nConflict in {label}"
        header += f"\nSpecies: {sp_id}"
        print(header)
        print(f"  old: {_record_summary(sp_id, old_rec)}")
        print(f"  new: {_record_summary(sp_id, new_rec)}")
        delta = _max_coeff_delta(old_rec, new_rec)
        if delta < float("inf"):
            print(f"  Max coefficient delta: {delta:.3e}")

        while True:
            choice = (
                input("  [k]eep old / [u]se new / [d]iff / [s]kip (keep old): ")
                .strip()
                .lower()
            )

            if choice in ("k", "s", ""):
                return old_rec
            if choice == "u":
                return new_rec
            if choice == "d":
                old_json = json.dumps(old_rec, indent=2).splitlines(keepends=True)
                new_json = json.dumps(new_rec, indent=2).splitlines(keepends=True)
                old_src = old_rec.get("source") or "old"
                new_src = new_rec.get("source") or "new"
                diff = difflib.unified_diff(
                    old_json,
                    new_json,
                    fromfile=f"{sp_id} [{old_src}]",
                    tofile=f"{sp_id}   [{new_src}]",
                    n=3,
                )
                print("".join(diff))

    @staticmethod
    def _resolve_conflict_csv_interactive(
        sp_id: str,
        old_rows: List[List[str]],
        new_rows: List[List[str]],
    ) -> List[List[str]]:
        """Prompt the user to resolve a JANAF CSV conflict."""
        print(f"\n{'─'*70}")
        print(f"JANAF conflict: {sp_id}")
        print(f"  old: {len(old_rows)} temperature rows")
        print(f"  new: {len(new_rows)} temperature rows")

        while True:
            choice = (
                input("  [k]eep old / [u]se new / [s]kip (keep old): ").strip().lower()
            )
            if choice in ("k", "s", ""):
                return old_rows
            if choice == "u":
                return new_rows
