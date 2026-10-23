"""
Solvers — runs every positioning algorithm in the real ``ultragps_position``
library on one frame of ticks.

The library is a copy of ``libs/ultragps_position`` from UltraGPS-Control
with two changes to its receiver filter: it drops distances <= 0, and its
upper bound is an adjustable top-band reject, ``max_distance`` (see README).

Algorithms shown:

  * ``OLS``  — the linear least-squares estimate the library uses as its seed
               (``_ols``; private, but it is the first stage of both solvers).
  * ``LM``   — OLS seed refined by Levenberg-Marquardt (``lm_position``).
  * ``CEP``  — best receiver subset by Circular Error Probable
               (``cep_position``).  The control program caps the search at
               15 subsets; the simulator passes a cap above the number of
               subsets that exist, so every subset of 3 or more kept
               receivers is tried and the lowest-CEP one wins.

  * ``CEP4`` — simulator-only variant of CEP for comparison.  It differs
               from the library's CEP in three ways:
                 - only subsets of 4 or more receivers are tried, so one bad
                   reading can't hide inside an exact 3-receiver fit;
                 - fixes outside the receivers' bounding box are kept;
                 - CEP is the textbook one: covariance = s² (JᵀJ)⁻¹ with
                   s² = sum of squared residuals / (m − 2).  The library
                   instead weights each receiver by 1 / (|residual| + 0.1),
                   which rewards a fit where some residuals are near zero
                   and lets a bad reading win.
               Each subset is solved with the library's own OLS + LM.  With
               fewer than 4 kept receivers it holds its last fix.

All four see the same receivers: the ones that pass the library's bounds and
jump (differential) filter, run once per frame by ``get_position_full`` —
exactly as UltraGPS-Control's ``windows/position_window.py`` calls it.
"""

from __future__ import annotations

import os
from itertools import combinations
from math import comb

import numpy as np

from ultragps_position import UltraGPSPositionLib, load_config

_REPO = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG  = os.path.join(_REPO, "config", "config.toml")
DEFAULT_SENSORS = os.path.join(_REPO, "config", "sensors.toml")
ALGORITHMS     = ("OLS", "LM", "CEP", "CEP4")


def load_sensors(config_path: str, sensors_path: str) -> tuple[np.ndarray, ...]:
    """Return ``(coords, slopes, intercepts, facing_deg)`` in receiver-id order:
    positions from config.toml; true tick response and facing direction from
    sensors.toml (``facing_deg`` is ``None`` if any receiver lacks one)."""
    receivers  = sorted(load_config(config_path)["receivers"], key=lambda r: r["id"])
    truth      = {int(r["id"]): r for r in load_config(sensors_path)["receivers"]}
    coords     = np.array([r["position"] for r in receivers], dtype=float)
    slopes     = np.array([truth[r["id"]]["slope"] for r in receivers], dtype=float)
    intercepts = np.array([truth[r["id"]]["intercept"] for r in receivers], dtype=float)
    facing     = [truth[r["id"]].get("facing_deg") for r in receivers]
    facing     = None if None in facing else np.array(facing, dtype=float)
    return coords, slopes, intercepts, facing


class Solvers:
    def __init__(self, config_path: str = DEFAULT_CONFIG, max_differential: float = 50.0,
                 max_distance: float | None = None):
        self.lib = UltraGPSPositionLib(config_path, max_differential=max_differential,
                                       max_distance=max_distance)
        # Subsets of size 3..n of n receivers, plus one: every subset is tried
        n = self.lib.receiver_count
        self.max_subsets = 2 ** n - sum(comb(n, k) for k in range(3)) + 1
        self.cep4_last = None

    def reset(self) -> None:
        self.lib.reset_state()
        self.cep4_last = None

    def _standard_cep(self, distances, subset) -> tuple:
        d      = np.array(distances, dtype=float)
        idx    = np.array(subset)
        pos, r = self.lib._nlls(d, self.lib._ols(d, idx), idx)
        J      = r.jac
        s2     = float(np.sum(r.fun ** 2)) / (len(subset) - 2)
        try:
            cov = s2 * np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            return pos, float("inf")
        return pos, 0.59 * (np.sqrt(abs(cov[0, 0])) + np.sqrt(abs(cov[1, 1])))

    def cep4(self, distances, sane) -> tuple:
        """Lowest-CEP subset of 4+ kept receivers, no bounding-box check.
        Returns ``(position, subset)``; subset is ``[]`` when holding."""
        best_pos, best_cep, best_idx = None, float("inf"), []
        for k in range(4, len(sane) + 1):
            for subset in combinations(sane, k):
                pos, cep = self._standard_cep(distances, list(subset))
                if cep < best_cep:
                    best_pos, best_cep, best_idx = pos, cep, list(subset)
        if best_pos is None:
            return self.cep4_last, []
        self.cep4_last = np.asarray(best_pos, dtype=float).copy()
        return self.cep4_last, best_idx

    def solve(self, ticks: list[int]) -> dict:
        """Return ``{"positions": {name: xy or None}, "success": {name: bool},
        "distances": [...], "sane": [...], "cep_subset": [...]}``."""
        r    = self.lib.get_position_full(ticks, max_subsets=self.max_subsets)
        sane = r["sane_indices"]

        ols = None
        if len(sane) >= 3:
            ols = self.lib._ols(np.array(r["distances"], dtype=float), np.array(sane))

        cep4_pos, cep4_subset = self.cep4(r["distances"], list(sane))

        return {
            "positions":  {"OLS": ols, "LM": r["lm_position"], "CEP": r["cep_position"],
                           "CEP4": cep4_pos},
            "success":    {"OLS": ols is not None, "LM": r["lm_success"], "CEP": r["cep_success"],
                           "CEP4": bool(cep4_subset)},
            "distances":  list(r["distances"]),
            "sane":       list(sane),
            # On a failed frame the library returns the last good fix and the
            # subset that produced it; report no subset so it is not mistaken
            # for one used this frame.
            "cep_subset":  list(r["best_indices"] or []) if r["cep_success"] else [],
            "cep4_subset": cep4_subset,
        }
