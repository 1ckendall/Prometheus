# Prometheus: An open-source combustion equilibrium solver in Python.
# Copyright (C) 2026 Charles Kendall
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

from scipy.constants import R

REFERENCE_TEMPERATURE = 298.15  # Kelvin
REFERENCE_PRESSURE = 1e5  # Standard reference pressure [Pa] (1 bar, IUPAC convention)
UNIVERSAL_GAS_CONSTANT = R

# We pretend that e- is an element as this is the easiest way to keep track of charge in equilibrium
# calculations.
ELEMENTS_MOLAR_MASSES = {
    "e-": 5.486e-4,  # Electron
    "H": 1.008,  # Hydrogen
    "D": 2.014,  # Deuterium (Isotope of Hydrogen)
    "T": 3.016,  # Tritium (Isotope of Hydrogen)
    "He": 4.002602,  # Helium
    "Li": 6.94,  # Lithium
    "Be": 9.0121831,  # Beryllium
    "B": 10.81,  # Boron
    "C": 12.011,  # Carbon
    "N": 14.007,  # Nitrogen
    "O": 15.999,  # Oxygen
    "F": 18.998403163,  # Fluorine
    "Ne": 20.1797,  # Neon
    "Na": 22.98976928,  # Sodium
    "Mg": 24.305,  # Magnesium
    "Al": 26.9815384,  # Aluminum
    "Si": 28.085,  # Silicon
    "P": 30.973761998,  # Phosphorus
    "S": 32.06,  # Sulfur
    "Cl": 35.45,  # Chlorine
    "Ar": 39.948,  # Argon
    "K": 39.0983,  # Potassium
    "Ca": 40.078,  # Calcium
    "Sc": 44.955908,  # Scandium
    "Ti": 47.867,  # Titanium
    "V": 50.9415,  # Vanadium
    "Cr": 51.9961,  # Chromium
    "Mn": 54.938043,  # Manganese
    "Fe": 55.845,  # Iron
    "Co": 58.933194,  # Cobalt
    "Ni": 58.6934,  # Nickel
    "Cu": 63.546,  # Copper
    "Zn": 65.38,  # Zinc
    "Ga": 69.723,  # Gallium
    "Ge": 72.630,  # Germanium
    "As": 74.921595,  # Arsenic
    "Se": 78.971,  # Selenium
    "Br": 79.904,  # Bromine
    "Kr": 83.798,  # Krypton
    "Rb": 85.4678,  # Rubidium
    "Sr": 87.62,  # Strontium
    "Y": 88.90584,  # Yttrium
    "Zr": 91.224,  # Zirconium
    "Nb": 92.90637,  # Niobium
    "Mo": 95.95,  # Molybdenum
    "Tc": 97,  # Technetium
    "Ru": 101.07,  # Ruthenium
    "Rh": 102.90549,  # Rhodium
    "Pd": 106.42,  # Palladium
    "Ag": 107.8682,  # Silver
    "Cd": 112.414,  # Cadmium
    "In": 114.818,  # Indium
    "Sn": 118.710,  # Tin
    "Sb": 121.760,  # Antimony
    "Te": 127.60,  # Tellurium
    "I": 126.90447,  # Iodine
    "Xe": 131.293,  # Xenon
    "Cs": 132.90545196,  # Cesium
    "Ba": 137.327,  # Barium
    "La": 138.90547,  # Lanthanum
    "Ce": 140.116,  # Cerium
    "Pr": 140.90766,  # Praseodymium
    "Nd": 144.242,  # Neodymium
    "Pm": 145,  # Promethium
    "Sm": 150.36,  # Samarium
    "Eu": 151.964,  # Europium
    "Gd": 157.25,  # Gadolinium
    "Tb": 158.925354,  # Terbium
    "Dy": 162.500,  # Dysprosium
    "Ho": 164.930328,  # Holmium
    "Er": 167.259,  # Erbium
    "Tm": 168.934218,  # Thulium
    "Yb": 173.045,  # Ytterbium
    "Lu": 174.9668,  # Lutetium
    "Hf": 178.486,  # Hafnium
    "Ta": 180.94788,  # Tantalum
    "W": 183.84,  # Tungsten
    "Re": 186.207,  # Rhenium
    "Os": 190.23,  # Osmium
    "Ir": 192.217,  # Iridium
    "Pt": 195.084,  # Platinum
    "Au": 196.966570,  # Gold
    "Hg": 200.592,  # Mercury
    "Tl": 204.38,  # Thallium
    "Pb": 207.2,  # Lead
    "Bi": 208.98040,  # Bismuth
    "Po": 209,  # Polonium
    "At": 210,  # Astatine
    "Rn": 222,  # Radon
    "Fr": 223,  # Francium
    "Ra": 226,  # Radium
    "Ac": 227,  # Actinium
    "Th": 232.0377,  # Thorium
    "Pa": 231.03588,  # Protactinium
    "U": 238.02891,  # Uranium
    "Np": 237,  # Neptunium
    "Pu": 244,  # Plutonium
    "Am": 243,  # Americium
    "Cm": 247,  # Curium
    "Bk": 247,  # Berkelium
    "Cf": 251,  # Californium
    "Es": 252,  # Einsteinium
    "Fm": 257,  # Fermium
    "Md": 258,  # Mendelevium
    "No": 259,  # Nobelium
    "Lr": 262,  # Lawrencium
    "Rf": 267,  # Rutherfordium
    "Db": 270,  # Dubnium
    "Sg": 269,  # Seaborgium
    "Bh": 270,  # Bohrium
    "Hs": 270,  # Hassium
    "Mt": 278,  # Meitnerium
    "Ds": 281,  # Darmstadtium
    "Rg": 281,  # Roentgenium
    "Cn": 285,  # Copernicium
    "Nh": 286,  # Nihonium
    "Fl": 289,  # Flerovium
    "Mc": 289,  # Moscovium
    "Lv": 293,  # Livermorium
    "Ts": 293,  # Tennessine
    "Og": 294,  # Oganesson
}
