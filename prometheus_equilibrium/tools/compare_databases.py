import json
import math
import sys
from pathlib import Path

from prometheus_equilibrium.equilibrium.species import (
    NASANineCoeff,
    NASASevenCoeff,
    R,
    ShomateCoeff,
)


def load_db(path, cls):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    species = {}
    for sp_id, rec in data.items():
        if sp_id.startswith("_"):
            continue
        try:
            if cls == NASASevenCoeff:
                sp = NASASevenCoeff(
                    elements=rec["elements"],
                    state=rec.get("phase", "G"),
                    temperature=(rec["t_low"], rec["t_mid"], rec["t_high"]),
                    coefficients=(rec["coeffs"]["low"], rec["coeffs"]["high"]),
                    phase=rec.get("phase"),
                )
            elif cls == NASANineCoeff:
                segs = rec.get("segments", [])
                temps = []
                for seg in segs:
                    if not temps:
                        temps.append(seg["t_low"])
                    temps.append(seg["t_high"])
                exponents = tuple(tuple(seg["exponents"]) for seg in segs)
                coefficients = tuple(
                    tuple(seg["coeffs"]) + (seg["b1"], seg["b2"]) for seg in segs
                )
                sp = NASANineCoeff(
                    elements=rec["elements"],
                    state=rec.get("phase", "G"),
                    temperatures=tuple(temps),
                    exponents=exponents,
                    coefficients=coefficients,
                    phase=rec.get("phase"),
                )
            elif cls == ShomateCoeff:
                segs = rec.get("segments", [])
                temps = []
                for seg in segs:
                    if not temps:
                        temps.append(seg["t_low"])
                    temps.append(seg["t_high"])
                coefficients = tuple(tuple(seg["coefficients"]) for seg in segs)
                sp = ShomateCoeff(
                    elements=rec["elements"],
                    state=rec.get("phase", "G"),
                    temperatures=tuple(temps),
                    coefficients=coefficients,
                    phase=rec.get("phase"),
                )
            species[sp_id] = sp
        except:
            continue
    return species


def main():
    _thermo = Path(__file__).resolve().parent.parent / "thermo_data"
    print("Loading databases...")
    n7 = load_db(str(_thermo / "nasa7.json"), NASASevenCoeff)
    n9 = load_db(str(_thermo / "nasa9.json"), NASANineCoeff)
    terra = load_db(str(_thermo / "terra.json"), ShomateCoeff)

    test_species = ["H2_G", "O2_G", "H2O_G", "CO2_G", "N2_G"]
    T = 298.15

    print(f"\nComparison at T = {T} K:")
    print(
        f"{'Species':<10} | {'Source':<10} | {'H (kJ/mol)':>12} | {'G (kJ/mol)':>12} | {'S (J/mol*K)':>12} | {'Cp (J/mol*K)':>12}"
    )
    print("-" * 85)

    for sid in test_species:
        for name, db in [("NASA-7", n7), ("NASA-9", n9), ("TERRA", terra)]:
            sp = db.get(sid)
            if sp:
                h = sp.enthalpy(T) / 1000.0
                g = sp.gibbs_free_energy(T) / 1000.0
                s = sp.entropy(T)
                cp = sp.specific_heat_capacity(T)
                print(
                    f"{sid:<10} | {name:<10} | {h:>12.3f} | {g:>12.3f} | {s:>12.3f} | {cp:>12.3f}"
                )
        print("-" * 85)


if __name__ == "__main__":
    main()
