#!/usr/bin/env python3
"""
Stima delle statistiche WEART ricavabili SENZA ground truth.

Input:
    un file di testo contenente righe come:

    ts=1788171631772 | closure=0.000 | opening=1.000 | abduction=0.525 |
    adduction=0.475 | ACC[g]=(0.087, 0.928, -0.286) |
    GYRO[deg/s]=(2.030, -7.000, -5.740) | ToF[mm]=83

Cosa stima:
    1) bias del giroscopio sull'asse di flessione
    2) rumore del giroscopio a mano ferma
    3) angolo accelerometrico medio nella posa aperta
    4) ripetibilita' dell'angolo accelerometrico a mano ferma
    5) valore ToF medio/mediano nella posa aperta
    6) rumore statico del ToF
    7) statistiche del periodo di campionamento dt
    8) norma dell'accelerometro

Cosa NON puo' stimare senza ground truth:
    - errore assoluto di theta
    - THETA_SIGMA_RAD reale dell'intero filtro ACC+GYRO
    - rapporto reale DIP/PIP
    - sigma del coupling DIP/PIP
    - errore assoluto di q1,q2,q3
    - errore geometrico effettivo del modello ToF

Uso:
    python3 estimate_weart_static_stats.py weart.log

Opzionale:
    python3 estimate_weart_static_stats.py weart.log --samples 200

Le righe utilizzate devono corrispondere a:
    indice aperto + mano ferma.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GYRO_FLEX_AXIS = 2
GYRO_FLEX_SIGN = +1.0

ACC_ANGLE_NUM_AXIS = 0
ACC_ANGLE_DEN_AXIS = 1
ACC_ANGLE_NUM_SIGN = +1.0
ACC_ANGLE_DEN_SIGN = +1.0

MAX_CLOSURE = 0.10
MAX_GYRO_NORM_DEG_S = 15.0
MIN_ACC_NORM_G = 0.75
MAX_ACC_NORM_G = 1.25

SAMPLE_RE = re.compile(
    r"ts=(?P<ts>\d+)\s*\|\s*"
    r"closure=(?P<closure>[-+0-9.eE]+)\s*\|\s*"
    r"opening=(?P<opening>[-+0-9.eE]+)\s*\|\s*"
    r"abduction=(?P<abduction>[-+0-9.eE]+)\s*\|\s*"
    r"adduction=(?P<adduction>[-+0-9.eE]+)\s*\|\s*"
    r"ACC\[g\]=\(\s*(?P<ax>[-+0-9.eE]+)\s*,\s*"
    r"(?P<ay>[-+0-9.eE]+)\s*,\s*(?P<az>[-+0-9.eE]+)\s*\)\s*\|\s*"
    r"GYRO\[deg/s\]=\(\s*(?P<gx>[-+0-9.eE]+)\s*,\s*"
    r"(?P<gy>[-+0-9.eE]+)\s*,\s*(?P<gz>[-+0-9.eE]+)\s*\)\s*\|\s*"
    r"ToF\[mm\]=(?P<tof>[-+0-9.eE]+)"
)

@dataclass
class WeartSample:
    ts_ms: int
    closure: float
    opening: float
    abduction: float
    adduction: float
    acc_g: np.ndarray
    gyro_deg_s: np.ndarray
    tof_mm: float


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(angles_rad: np.ndarray) -> float:
    return float(math.atan2(float(np.mean(np.sin(angles_rad))), float(np.mean(np.cos(angles_rad)))))


def sample_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def robust_sigma_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def raw_acc_angle(acc_g: np.ndarray) -> float:
    num = ACC_ANGLE_NUM_SIGN * float(acc_g[ACC_ANGLE_NUM_AXIS])
    den = ACC_ANGLE_DEN_SIGN * float(acc_g[ACC_ANGLE_DEN_AXIS])
    return math.atan2(num, den)


def parse_file(path: Path) -> list[WeartSample]:
    samples = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SAMPLE_RE.search(line)
            if m is None:
                continue
            d = m.groupdict()
            samples.append(
                WeartSample(
                    ts_ms=int(d["ts"]),
                    closure=float(d["closure"]),
                    opening=float(d["opening"]),
                    abduction=float(d["abduction"]),
                    adduction=float(d["adduction"]),
                    acc_g=np.array([float(d["ax"]), float(d["ay"]), float(d["az"])], dtype=float),
                    gyro_deg_s=np.array([float(d["gx"]), float(d["gy"]), float(d["gz"])], dtype=float),
                    tof_mm=float(d["tof"]),
                )
            )
    if not samples:
        raise ValueError(f"Nessun campione WEART riconosciuto nel file: {path}")
    samples.sort(key=lambda s: s.ts_ms)
    return samples


def is_static_open_sample(s: WeartSample) -> bool:
    gyro_norm = float(np.linalg.norm(s.gyro_deg_s))
    acc_norm = float(np.linalg.norm(s.acc_g))
    return (
        s.closure <= MAX_CLOSURE
        and gyro_norm <= MAX_GYRO_NORM_DEG_S
        and MIN_ACC_NORM_G <= acc_norm <= MAX_ACC_NORM_G
    )


def select_samples(samples: list[WeartSample], max_samples: int | None) -> list[WeartSample]:
    selected = [s for s in samples if is_static_open_sample(s)]
    if max_samples is not None:
        selected = selected[:max_samples]
    if len(selected) < 5:
        raise ValueError("Troppi pochi campioni validi. Registra il dito aperto e la mano ferma piu' a lungo.")
    return selected


def estimate_stats(samples: list[WeartSample]) -> dict:
    gyro_flex = np.array([
        GYRO_FLEX_SIGN * float(s.gyro_deg_s[GYRO_FLEX_AXIS])
        for s in samples
    ], dtype=float)
    gyro_bias = float(np.mean(gyro_flex))
    gyro_residual = gyro_flex - gyro_bias

    acc_angles = np.array([raw_acc_angle(s.acc_g) for s in samples], dtype=float)
    acc_open_angle = circular_mean(acc_angles)
    acc_angle_error = np.asarray(wrap_pi(acc_angles - acc_open_angle), dtype=float)
    acc_norm = np.array([float(np.linalg.norm(s.acc_g)) for s in samples], dtype=float)

    tof = np.array([float(s.tof_mm) for s in samples], dtype=float)
    tof_mean = float(np.mean(tof))
    tof_median = float(np.median(tof))

    ts = np.array([s.ts_ms for s in samples], dtype=np.int64)
    dt_s = np.diff(ts).astype(float) * 1e-3
    valid_dt = dt_s[np.isfinite(dt_s) & (dt_s > 0.0)]

    if len(valid_dt):
        dt_mean = float(np.mean(valid_dt))
        dt_std = sample_std(valid_dt)
        dt_median = float(np.median(valid_dt))
        freq_mean = 1.0 / dt_mean if dt_mean > 0 else float("nan")
    else:
        dt_mean = dt_std = dt_median = freq_mean = float("nan")

    return {
        "n_samples": len(samples),
        "gyro": {
            "axis": GYRO_FLEX_AXIS,
            "sign": GYRO_FLEX_SIGN,
            "bias_deg_s": gyro_bias,
            "sigma_deg_s": sample_std(gyro_residual),
            "robust_sigma_deg_s": robust_sigma_mad(gyro_residual),
        },
        "accelerometer": {
            "open_angle_rad": acc_open_angle,
            "open_angle_deg": math.degrees(acc_open_angle),
            "static_angle_sigma_rad": sample_std(acc_angle_error),
            "static_angle_sigma_deg": math.degrees(sample_std(acc_angle_error)),
            "static_angle_robust_sigma_rad": robust_sigma_mad(acc_angle_error),
            "static_angle_robust_sigma_deg": math.degrees(robust_sigma_mad(acc_angle_error)),
            "norm_mean_g": float(np.mean(acc_norm)),
            "norm_sigma_g": sample_std(acc_norm),
        },
        "tof": {
            "open_mean_mm": tof_mean,
            "open_median_mm": tof_median,
            "static_sigma_mm": sample_std(tof - tof_mean),
            "static_robust_sigma_mm": robust_sigma_mad(tof - tof_median),
        },
        "timing": {
            "mean_dt_s": dt_mean,
            "median_dt_s": dt_median,
            "sigma_dt_s": dt_std,
            "mean_frequency_hz": freq_mean,
        },
    }


def print_report(stats: dict) -> None:
    g = stats["gyro"]
    a = stats["accelerometer"]
    t = stats["tof"]
    tm = stats["timing"]

    print("\n============================================")
    print(" WEART STATIC CALIBRATION REPORT")
    print("============================================")
    print(f"Campioni usati: {stats['n_samples']}\n")

    print("GYRO")
    print("--------------------------------------------")
    print(f"bias asse flessione      = {g['bias_deg_s']:.6f} deg/s")
    print(f"sigma rumore             = {g['sigma_deg_s']:.6f} deg/s")
    print(f"sigma robusta (MAD)      = {g['robust_sigma_deg_s']:.6f} deg/s\n")

    print("ACCELEROMETRO")
    print("--------------------------------------------")
    print(f"angolo posa aperta       = {a['open_angle_deg']:.6f} deg")
    print(f"sigma angolo statico     = {a['static_angle_sigma_deg']:.6f} deg")
    print(f"sigma angolo robusta     = {a['static_angle_robust_sigma_deg']:.6f} deg")
    print(f"norma media              = {a['norm_mean_g']:.6f} g")
    print(f"sigma norma              = {a['norm_sigma_g']:.6f} g\n")

    print("ToF")
    print("--------------------------------------------")
    print(f"ToF medio open           = {t['open_mean_mm']:.6f} mm")
    print(f"ToF mediano open         = {t['open_median_mm']:.6f} mm")
    print(f"sigma statica            = {t['static_sigma_mm']:.6f} mm")
    print(f"sigma robusta (MAD)      = {t['static_robust_sigma_mm']:.6f} mm\n")

    print("TIMING")
    print("--------------------------------------------")
    print(f"dt medio                 = {tm['mean_dt_s']:.6f} s")
    print(f"dt mediano               = {tm['median_dt_s']:.6f} s")
    print(f"sigma dt                 = {tm['sigma_dt_s']:.6f} s")
    print(f"frequenza media          = {tm['mean_frequency_hz']:.3f} Hz\n")

    print("============================================")
    print(" VALORI UTILIZZABILI")
    print("============================================")
    print(f"GYRO_BIAS_DEG_S = {g['bias_deg_s']:.8f}")
    print(f"GYRO_NOISE_SIGMA_DEG_S = {g['robust_sigma_deg_s']:.8f}")
    print(f"TOF_STATIC_SIGMA_MM = {t['static_robust_sigma_mm']:.8f}")
    print(f"ACC_STATIC_ANGLE_SIGMA_RAD = {a['static_angle_robust_sigma_rad']:.10f}")

    print("\nATTENZIONE:")
    print("- TOF_STATIC_SIGMA_MM e' solo il rumore statico del sensore.")
    print("  Non comprende l'errore del modello geometrico ToF.")
    print("- ACC_STATIC_ANGLE_SIGMA_RAD e' solo la ripetibilita' statica dell'ACC.")
    print("  Non e' THETA_SIGMA_RAD dell'intero filtro ACC+GYRO.")
    print("- Senza ground truth NON si possono identificare correttamente:")
    print("  THETA_SIGMA totale, DIP/PIP ratio, coupling sigma, errore q1/q2/q3.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weart_log", type=Path, help="File di testo contenente i campioni WEART")
    parser.add_argument("--samples", type=int, default=None, help="Numero massimo di campioni statici/open da usare")
    parser.add_argument("--json", type=Path, default=None, help="Salva anche il report completo in JSON")
    args = parser.parse_args()

    all_samples = parse_file(args.weart_log)
    selected = select_samples(all_samples, args.samples)
    stats = estimate_stats(selected)
    print_report(stats)

    if args.json is not None:
        args.json.write_text(json.dumps(stats, indent=2, allow_nan=True), encoding="utf-8")
        print(f"Report JSON salvato in: {args.json}")


if __name__ == "__main__":
    main()