"""Tests for the PEPCODED.DAF -> propellants.toml converter."""

from __future__ import annotations

import tomllib
from pathlib import Path

from prometheus_equilibrium.tools.build_propellants import (
    _assign_roles,
    _parse_fixed_record,
    build_propellants_toml,
)


def test_parse_fixed_record_ap_line() -> None:
    line = "F     30 AMMONIUM PERCHLORATE            1CL  4H   1N   4O   0    0   -601 .0704]"
    rec = _parse_fixed_record(line)
    assert rec is not None
    assert rec.seq == 30
    assert rec.flags == "F"
    assert rec.name == "AMMONIUM PERCHLORATE"
    assert rec.elements == {"Cl": 1.0, "H": 4.0, "N": 1.0, "O": 4.0}
    assert rec.heat_cal_per_g == -601.0
    assert rec.density_lb_per_in3 == 0.0704


def test_role_heuristic_oxidizer_vs_fuel() -> None:
    ap_roles = _assign_roles(
        "AMMONIUM PERCHLORATE", {"Cl": 1.0, "H": 4.0, "N": 1.0, "O": 4.0}, "F"
    )
    assert "oxidizer" in ap_roles

    ch4_roles = _assign_roles("METHANE", {"C": 1.0, "H": 4.0}, "")
    assert "fuel" in ch4_roles
    assert "oxidizer" not in ch4_roles


def test_build_propellants_toml_skips_deuterium(tmp_path: Path) -> None:
    daf = tmp_path / "PEPCODED.DAF"
    out = tmp_path / "propellants.toml"

    daf.write_text(
        "\n".join(
            [
                "* header",
                "F     30 AMMONIUM PERCHLORATE            1CL  4H   1N   4O   0    0   -601 .0704]",
                "      31 DEUTERATED TEST                 1D   1O   0    0    0    0   -120 .0400]",
                "E     29 ALUMINUM (PURE CRYSTALINE)      1AL  0    0    0    0    0      0 .0975]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    count = build_propellants_toml(daf, out)
    assert count == 2

    with open(out, "rb") as f:
        data = tomllib.load(f)

    ingredients = data["ingredient"]
    names = {item["name"] for item in ingredients}
    assert "DEUTERATED TEST" not in names
    assert "AMMONIUM PERCHLORATE" in names
    assert "ALUMINUM (PURE CRYSTALINE)" in names
