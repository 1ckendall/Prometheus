"""Numba JIT kernels for NASA-7 and NASA-9 thermodynamic property evaluation.

These kernels provide fast scalar evaluation of:
  - G°/RT  (reduced Gibbs free energy)
  - H°/RT  (reduced enthalpy)
  - Cp°/R  (reduced specific heat capacity)

for both the NASA-7 piecewise polynomial and the NASA-9 multi-segment polynomial
formats. They are called from NASASevenCoeff and NASANineCoeff when a scalar
temperature is passed in the hot path of the equilibrium solver.

NASA-9 array layout:
  t_bounds   : float64[n_segs+1]   — segment boundary temperatures
  exps_mat   : float64[n_segs, max_exps]      — exponents, zero-padded
  coeffs_mat : float64[n_segs, max_exps+2]    — a1..an, b1, b2
  n_segs     : int                            — number of segments
  n_exps_arr : int64[n_segs]                  — actual exponent count per segment

NASA-7 array layout:
  T_common   : float64 scalar
  lo_coeffs  : float64[7]   — [a1..a5, a6, a7]
  hi_coeffs  : float64[7]
"""

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

import math

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# NASA-9 kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _gibbs_n9(T, t_bounds, exps_mat, coeffs_mat, n_segs, n_exps_arr):
    """Return G°/RT = H°/RT - S°/R for NASA-9 polynomial at scalar T."""
    # Find segment
    k = -1
    for s in range(n_segs):
        t_lo = t_bounds[s]
        t_hi = t_bounds[s + 1]
        if s == n_segs - 1:
            if t_lo <= T <= t_hi:
                k = s
                break
        else:
            if t_lo <= T < t_hi:
                k = s
                break
    if k < 0:
        return math.nan

    n = n_exps_arr[k]
    log_T = math.log(T)
    h_rt = 0.0
    s_r = 0.0
    for i in range(n):
        e = exps_mat[k, i]
        c = coeffs_mat[k, i]
        te = T**e
        if abs(e + 1.0) < 1e-9:  # e == -1 branch for H
            h_rt += c * log_T / T
        else:
            h_rt += c * te / (e + 1.0)
        if abs(e) < 1e-9:  # e == 0 branch for S
            s_r += c * log_T
        else:
            s_r += c * te / e
    h_rt += coeffs_mat[k, n] / T  # b1/T
    s_r += coeffs_mat[k, n + 1]  # b2
    return h_rt - s_r


@njit(cache=True)
def _enthalpy_n9(T, t_bounds, exps_mat, coeffs_mat, n_segs, n_exps_arr):
    """Return H°/RT for NASA-9 polynomial at scalar T."""
    k = -1
    for s in range(n_segs):
        t_lo = t_bounds[s]
        t_hi = t_bounds[s + 1]
        if s == n_segs - 1:
            if t_lo <= T <= t_hi:
                k = s
                break
        else:
            if t_lo <= T < t_hi:
                k = s
                break
    if k < 0:
        return math.nan

    n = n_exps_arr[k]
    log_T = math.log(T)
    h_rt = 0.0
    for i in range(n):
        e = exps_mat[k, i]
        c = coeffs_mat[k, i]
        if abs(e + 1.0) < 1e-9:  # e == -1
            h_rt += c * log_T / T
        else:
            h_rt += c * T**e / (e + 1.0)
    h_rt += coeffs_mat[k, n] / T  # b1/T
    return h_rt


@njit(cache=True)
def _cp_n9(T, t_bounds, exps_mat, coeffs_mat, n_segs, n_exps_arr):
    """Return Cp°/R for NASA-9 polynomial at scalar T."""
    k = -1
    for s in range(n_segs):
        t_lo = t_bounds[s]
        t_hi = t_bounds[s + 1]
        if s == n_segs - 1:
            if t_lo <= T <= t_hi:
                k = s
                break
        else:
            if t_lo <= T < t_hi:
                k = s
                break
    if k < 0:
        return math.nan

    n = n_exps_arr[k]
    cp_r = 0.0
    for i in range(n):
        cp_r += coeffs_mat[k, i] * T ** exps_mat[k, i]
    return cp_r


# ---------------------------------------------------------------------------
# NASA-7 kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _gibbs_n7(T, T_common, lo_coeffs, hi_coeffs):
    """Return G°/RT = H°/RT - S°/R for NASA-7 polynomial at scalar T."""
    if T <= T_common:
        c = lo_coeffs
    else:
        c = hi_coeffs
    log_T = math.log(T)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    h_rt = (
        c[0]
        + c[1] * T / 2.0
        + c[2] * T2 / 3.0
        + c[3] * T3 / 4.0
        + c[4] * T4 / 5.0
        + c[5] / T
    )
    s_r = (
        c[0] * log_T
        + c[1] * T
        + c[2] * T2 / 2.0
        + c[3] * T3 / 3.0
        + c[4] * T4 / 4.0
        + c[6]
    )
    return h_rt - s_r


@njit(cache=True)
def _enthalpy_n7(T, T_common, lo_coeffs, hi_coeffs):
    """Return H°/RT for NASA-7 polynomial at scalar T."""
    if T <= T_common:
        c = lo_coeffs
    else:
        c = hi_coeffs
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    return (
        c[0]
        + c[1] * T / 2.0
        + c[2] * T2 / 3.0
        + c[3] * T3 / 4.0
        + c[4] * T4 / 5.0
        + c[5] / T
    )


@njit(cache=True)
def _cp_n7(T, T_common, lo_coeffs, hi_coeffs):
    """Return Cp°/R for NASA-7 polynomial at scalar T."""
    if T <= T_common:
        c = lo_coeffs
    else:
        c = hi_coeffs
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    return c[0] + c[1] * T + c[2] * T2 + c[3] * T3 + c[4] * T4


# ---------------------------------------------------------------------------
# Shomate kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _gibbs_shomate(T, t_bounds, coeffs_mat, n_segs, R):
    """Return G°/RT = H°/RT - S°/R for Shomate polynomial at scalar T."""
    k = -1
    for s in range(n_segs):
        t_lo = t_bounds[s]
        t_hi = t_bounds[s + 1]
        if s == n_segs - 1:
            if t_lo <= T <= t_hi:
                k = s
                break
        else:
            if t_lo <= T < t_hi:
                k = s
                break
    if k < 0:
        return math.nan

    t = T * 1e-3
    c = coeffs_mat[k]
    H = (
        c[0] * t
        + c[1] * t * t / 2.0
        + c[2] * t * t * t / 3.0
        + c[3] * t * t * t * t / 4.0
        - c[4] / t
        + c[5]
    ) * 1000.0
    S = (
        c[0] * math.log(t)
        + c[1] * t
        + c[2] * t * t / 2.0
        + c[3] * t * t * t / 3.0
        - c[4] / (2.0 * t * t)
        + c[6]
    )
    return (H - S * T) / (R * T)


@njit(cache=True)
def _enthalpy_shomate(T, t_bounds, coeffs_mat, n_segs, R):
    """Return H°/RT for Shomate polynomial at scalar T."""
    k = -1
    for s in range(n_segs):
        t_lo = t_bounds[s]
        t_hi = t_bounds[s + 1]
        if s == n_segs - 1:
            if t_lo <= T <= t_hi:
                k = s
                break
        else:
            if t_lo <= T < t_hi:
                k = s
                break
    if k < 0:
        return math.nan

    t = T * 1e-3
    c = coeffs_mat[k]
    H = (
        c[0] * t
        + c[1] * t * t / 2.0
        + c[2] * t * t * t / 3.0
        + c[3] * t * t * t * t / 4.0
        - c[4] / t
        + c[5]
    ) * 1000.0
    return H / (R * T)


@njit(cache=True)
def _cp_shomate(T, t_bounds, coeffs_mat, n_segs, R):
    """Return Cp°/R for Shomate polynomial at scalar T."""
    k = -1
    for s in range(n_segs):
        t_lo = t_bounds[s]
        t_hi = t_bounds[s + 1]
        if s == n_segs - 1:
            if t_lo <= T <= t_hi:
                k = s
                break
        else:
            if t_lo <= T < t_hi:
                k = s
                break
    if k < 0:
        return math.nan

    t = T * 1e-3
    c = coeffs_mat[k]
    cp = c[0] + c[1] * t + c[2] * t * t + c[3] * t * t * t + c[4] / (t * t)
    return cp / R


# ---------------------------------------------------------------------------
# AFCESIC kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _gibbs_afcesic(T, T_low, T_mid, T_high, lo, hi, rf, ch, h0_hi, s0_hi, R_cal):
    """Return G°/RT = H°/RT - S°/R for AFCESIC polynomial at scalar T."""
    if T < T_low or T > T_high:
        return math.nan

    th = T * 1e-3
    if T <= T_mid:
        h = (
            1000.0
            * (
                lo[0] * th
                + lo[1] * th**2 / 2.0
                + lo[2] * th**3 / 3.0
                + lo[3] * th**4 / 4.0
                + lo[4] * th**5 / 5.0
            )
            + rf
        )
        s = (
            lo[0] * math.log(th)
            + lo[1] * th
            + lo[2] * th**2 / 2.0
            + lo[3] * th**3 / 3.0
            + lo[4] * th**4 / 4.0
            + ch
        )
    else:
        inv = 1.0 / th
        h = (
            1000.0
            * (
                hi[0] * th
                + hi[1] * math.log(th)
                - hi[2] * inv
                - hi[3] * inv**2 / 2.0
                - hi[4] * inv**3 / 3.0
            )
            + h0_hi
        )
        s = (
            hi[0] * math.log(th)
            - hi[1] * inv
            - hi[2] * inv**2 / 2.0
            - hi[3] * inv**3 / 3.0
            - hi[4] * inv**4 / 4.0
            + s0_hi
        )
    return h / (R_cal * T) - s / R_cal


@njit(cache=True)
def _enthalpy_afcesic(T, T_low, T_mid, T_high, lo, hi, rf, h0_hi, R_cal):
    """Return H°/RT for AFCESIC polynomial at scalar T."""
    if T < T_low or T > T_high:
        return math.nan

    th = T * 1e-3
    if T <= T_mid:
        h = (
            1000.0
            * (
                lo[0] * th
                + lo[1] * th**2 / 2.0
                + lo[2] * th**3 / 3.0
                + lo[3] * th**4 / 4.0
                + lo[4] * th**5 / 5.0
            )
            + rf
        )
    else:
        inv = 1.0 / th
        h = (
            1000.0
            * (
                hi[0] * th
                + hi[1] * math.log(th)
                - hi[2] * inv
                - hi[3] * inv**2 / 2.0
                - hi[4] * inv**3 / 3.0
            )
            + h0_hi
        )
    return h / (R_cal * T)


@njit(cache=True)
def _cp_afcesic(T, T_low, T_mid, T_high, lo, hi, R_cal):
    """Return Cp°/R for AFCESIC polynomial at scalar T."""
    if T < T_low or T > T_high:
        return math.nan

    th = T * 1e-3
    if T <= T_mid:
        cp = lo[0] + lo[1] * th + lo[2] * th**2 + lo[3] * th**3 + lo[4] * th**4
    else:
        inv = 1.0 / th
        cp = hi[0] + hi[1] * inv + hi[2] * inv**2 + hi[3] * inv**3 + hi[4] * inv**4
    return cp / R_cal
