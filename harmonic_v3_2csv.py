#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Harmonic Grand Unified Theory (GUT)
harmonic_v3_2csv.py — loto 7/39

coherence_score, peak na mreži 1..39, next = top 7

Optimizacije:
  1) fuzija T1+T2+T3 mreže 
  2) ćelija (Z,N) × ko-pojava sa celog CSV
  3) blagi boost ako je Z ili N u poslednjem kolu (prelaz)

SEED=39; cela dva CSV; bez randoma
Dva odvojena CSV-a (Loto / Loto Plus).
"""


from pathlib import Path
from typing import List

import numpy as np

MAGIC_NUMBERS = np.array([5, 10, 15, 20, 25, 30, 35])

SEED = 39
MAX_NUM = 39
N_PICK = 7
# fuzija grana 
W_T1, W_T2, W_T3 = 0.50, 0.25, 0.25
# prelaz sa poslednjeg kola
LAST_BOOST = 1.15

ROOT = Path(__file__).resolve().parent
CSV_LOTO = ROOT.parent / "data" / "loto7_4654_k58_loto_2949.csv"
CSV_PLUS = ROOT.parent / "data" / "loto7_4654_k58_loto_plus_1705.csv"
CSV_PATH = CSV_LOTO
POS_LO = np.arange(1, N_PICK + 1, dtype=int)
POS_HI = POS_LO + (MAX_NUM - N_PICK)

anchor_default = 27.0
ratio_default = 19.0 / 13.0
cutoff_default = anchor_default / 230.0
n_default = 3.54
t3_default = 0.0


def coherence_score(Z, N, anchor_hz, ratio_19_13, cutoff_hz, exponent_n,
                    branch_mix, t3_coupling):
    """funkcija za računanje koherencije"""
    A = Z + N
    f_scale = A * anchor_hz / 27.0
    resonance = np.exp(-5.0 * np.abs(f_scale - np.round(f_scale)))
    Z_magic_dist = np.min(np.abs(Z - MAGIC_NUMBERS)) / 40.0
    N_magic_dist = np.min(np.abs(N - MAGIC_NUMBERS)) / 40.0
    if N > 0:
        ratio = Z / N
        ratio_penalty = np.abs(ratio - ratio_19_13) / ratio_19_13
    else:
        ratio_penalty = 1.0
    n_eff = A / 230.0
    n_penalty = np.exp(-10.0 * np.abs(n_eff - np.round(n_eff)))
    n_val = np.log(A + 1.0) / np.log(anchor_hz) if anchor_hz > 1 else 0.0
    if branch_mix >= 0.5:
        n_target = exponent_n
        n_res = np.exp(-2.0 * np.abs(np.sin(np.pi * (n_val - n_target))))
    elif branch_mix <= -0.5:
        n_target = -exponent_n
        n_res = np.exp(-2.0 * np.abs(np.sin(np.pi * (n_val - n_target))))
    else:
        n_target_T3 = exponent_n * (13.0 / 19.0)
        n_res = np.exp(-2.0 * np.abs(np.sin(np.pi * (n_val - n_target_T3))))
        n_eff_T3 = A * (13.0 / 19.0) / 230.0
        n_penalty = np.exp(-10.0 * np.abs(n_eff_T3 - np.round(n_eff_T3)))
    coherence = resonance * n_penalty * n_res * np.exp(
        -(Z_magic_dist + N_magic_dist + ratio_penalty)
    )
    if t3_coupling > 0 and abs(branch_mix) < 0.5:
        coherence = coherence * (1.0 - t3_coupling) + t3_coupling * 0.8
    return np.clip(coherence, 0.0, 1.0)


def load_draws(path: Path = CSV_PATH) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", dtype=int)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    assert raw.shape[1] == N_PICK
    return raw


def cooccurrence(draws: np.ndarray) -> np.ndarray:
    """C[i,j] = broj kola gde se zajedno pojave i+1 i j+1 (ceo CSV)."""
    C = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
    for row in draws:
        idx = [int(x) - 1 for x in row]
        for a in range(N_PICK):
            for b in range(a + 1, N_PICK):
                i, j = idx[a], idx[b]
                C[i, j] += 1.0
                C[j, i] += 1.0
    return C


def fused_grid() -> np.ndarray:
    """Mreža 1..39: W_T1·T1 + W_T2·T2 + W_T3·T3."""
    vals = np.arange(1, MAX_NUM + 1, dtype=np.float64)
    g1 = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
    g2 = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
    g3 = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
    for i, Z in enumerate(vals):
        for j, N in enumerate(vals):
            g1[i, j] = coherence_score(
                Z, N, anchor_default, ratio_default,
                cutoff_default, n_default, 1.0, t3_default,
            )
            g2[i, j] = coherence_score(
                Z, N, anchor_default, ratio_default,
                cutoff_default, n_default, -1.0, t3_default,
            )
            g3[i, j] = coherence_score(
                Z, N, anchor_default, ratio_default,
                cutoff_default, n_default, 0.0, t3_default,
            )
    return W_T1 * g1 + W_T2 * g2 + W_T3 * g3


def predict_next(draws: np.ndarray) -> List[int]:
    grid = fused_grid()
    C = cooccurrence(draws)
    c_max = float(np.max(C)) if np.max(C) > 0 else 1.0
    # coherence × ko-pojava
    W = grid * (C / c_max)
    last = set(int(x) for x in draws[-1])

    # peak na W — svi indeksi 0..38 (uključujući brojeve 1 i 39)
    peak_score = np.zeros(MAX_NUM, dtype=np.float64)
    for i in range(MAX_NUM):
        for j in range(MAX_NUM):
            if i == j:
                continue
            w = W[i, j]
            if w <= 0.0:
                continue
            is_peak = True
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < MAX_NUM and 0 <= nj < MAX_NUM and w < W[ni, nj]:
                    is_peak = False
                    break
            if is_peak:
                peak_score[i] += w
                peak_score[j] += w

    freq = np.zeros(MAX_NUM, dtype=np.float64)
    for row in draws:
        for x in row:
            freq[int(x) - 1] += 1.0
    freq_max = float(np.max(freq)) if np.max(freq) > 0 else 1.0
    scores = peak_score * (0.35 + 0.65 * (freq / freq_max))
    for n in last:
        scores[n - 1] *= LAST_BOOST
    scores = scores + 1e-12 * ((np.arange(1, MAX_NUM + 1) * SEED) % 97)

    used = set()
    out: List[int] = []
    for slot in range(N_PICK):
        lo, hi = int(POS_LO[slot]), int(POS_HI[slot])
        best_n, best_key = None, None
        for n in range(lo, hi + 1):
            if n in used:
                continue
            key = (-scores[n - 1], n)
            if best_key is None or key < best_key:
                best_key = key
                best_n = n
        used.add(best_n)
        out.append(best_n)
    return sorted(out)


def main() -> None:
    draws_loto = load_draws(CSV_LOTO)
    draws_plus = load_draws(CSV_PLUS)
    next_loto = predict_next(draws_loto)
    next_loto_plus = predict_next(draws_plus)
    print("next_loto:      ", next_loto)
    print("next_loto_plus: ", next_loto_plus)


if __name__ == "__main__":
    main()


"""
next_loto:       [2, 3, 10, 15, 16, 19, 22]
next_loto_plus:  [2, 3, 11, 18, 20, 23, 25]
"""
########################################################################################