#!/usr/bin/env python3
"""Verify published primary results from frozen TEST predictions.

Requires numpy and scikit-learn. Does not retrain or refit models.
Writes only to reproducibility/outputs/.
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/final_evaluation/final_test_primary_v1"
OUTPUT = ROOT / "reproducibility/outputs"

EXPECTED_HASHES = {
    "paired_test_labels.npy":
        "0b96c78c74aad53b52bc5ae5670429e6c0b7cacdd8d03017a376f65c6d844012",
    "paired_sequence_probs.npy":
        "d625b8ee115979b44abc8314abfc28db8c715d11bcec10a30e44477e3b568512",
    "paired_structure_probs.npy":
        "f1f40070d842f6dd2c90b2f5cc7c150d889575410646cc3d761c82091266822f",
    "structure_failure_target.npy":
        "f7ea830936703596403c8d3f8f493e5f4eda2cd7ee8e45c8739cbc8d3febadc6",
    "MA_binary_predictions.npy":
        "5e1c47075a2cf90b356289bee32b2f435c7aa86453dddf8d6923d0b14de2e6b3",
    "MB_binary_predictions.npy":
        "9b8a0da0f02f4f0e83b8f58d7be1a5b9b532db34b1ed40b50f15f52d43105503",
    "A1_predictions.npy":
        "9f3024da1eac34275e3eb655ff96080fcafc6bc09e9ae33b46644572966eadeb",
    "A2_predictions.npy":
        "2188b90eef326f14751a56afb26fd9d5bb7dabed7932e666b7763df13c107aef",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(name):
    path = INPUT / name

    if not path.is_file():
        raise FileNotFoundError(path)

    if name in EXPECTED_HASHES:
        actual = sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise RuntimeError(f"SHA256 mismatch: {name}")

    array = np.load(path, allow_pickle=False)

    if array.shape != (4731, 494):
        raise RuntimeError(
            f"Unexpected shape for {name}: {array.shape}"
        )

    if not np.isfinite(array).all():
        raise RuntimeError(f"Nonfinite values: {name}")

    return array


def ap(y, p):
    return float(average_precision_score(y.ravel(), p.ravel()))


def macro_ap(y, p):
    supported = y.sum(axis=0) > 0
    return float(
        average_precision_score(
            y[:, supported],
            p[:, supported],
            average="macro",
        )
    )


def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    parser = argparse.ArgumentParser(
        description="Verify frozen primary TEST results."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Verify point estimates only (default).",
    )
    mode.add_argument(
        "--bootstrap",
        action="store_true",
        help="Also reproduce 1,000 bootstrap replicates.",
    )
    args = parser.parse_args()
    y = load("paired_test_labels.npy")
    seq = load("paired_sequence_probs.npy")
    struct = load("paired_structure_probs.npy")
    failure = load("structure_failure_target.npy")
    ma = load("MA_binary_predictions.npy")
    mb = load("MB_binary_predictions.npy")
    a1 = load("A1_predictions.npy")
    a2 = load("A2_predictions.npy")

    if not np.array_equal(
        failure.astype(bool),
        np.abs(struct - y) >= 0.5,
    ):
        raise RuntimeError("Failure target definition mismatch")

    primary = {
        "MA_AP": ap(failure, ma),
        "MB_AP": ap(failure, mb),
    }
    primary["delta_AP"] = primary["MB_AP"] - primary["MA_AP"]

    arbitration = {
        "A0_micro_AP": ap(y, seq),
        "A1_micro_AP": ap(y, a1),
        "A2_micro_AP": ap(y, a2),
        "A0_macro_AP": macro_ap(y, seq),
        "A1_macro_AP": macro_ap(y, a1),
        "A2_macro_AP": macro_ap(y, a2),
    }
    arbitration["delta_micro_AP"] = (
        arbitration["A2_micro_AP"] - arbitration["A1_micro_AP"]
    )
    arbitration["delta_macro_AP"] = (
        arbitration["A2_macro_AP"] - arbitration["A1_macro_AP"]
    )

    if not args.bootstrap:
        report = {
            "mode": "quick",
            "population": {
                "proteins": 4731,
                "go_mf_labels": 494,
            },
            "primary_reliability": primary,
            "arbitration": arbitration,
        }

        checks = {
            "primary_delta": abs(
                primary["delta_AP"] - 0.111614032
            ) < 1e-6,
            "arbitration_micro_delta": abs(
                arbitration["delta_micro_AP"] - 0.034784218
            ) < 1e-6,
            "arbitration_macro_delta": abs(
                arbitration["delta_macro_AP"] - 0.045425730
            ) < 1e-6,
        }

        report["checks"] = checks

        if not all(checks.values()):
            raise RuntimeError(
                f"Published result mismatch: {checks}"
            )

        OUTPUT.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT / "primary_quick_report.json"
        output_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        print(json.dumps(report, indent=2))
        print("\nReport:", output_path)
        print("QUICK REPRODUCTION PASSED")
        return

    rng = np.random.default_rng(42)
    n = y.shape[0]

    reliability_deltas = []
    arbitration_micro_deltas = []
    arbitration_macro_deltas = []

    for i in range(1000):
        idx = rng.integers(0, n, size=n)

        reliability_deltas.append(
            ap(failure[idx], mb[idx])
            - ap(failure[idx], ma[idx])
        )
        arbitration_micro_deltas.append(
            ap(y[idx], a2[idx])
            - ap(y[idx], a1[idx])
        )
        arbitration_macro_deltas.append(
            macro_ap(y[idx], a2[idx])
            - macro_ap(y[idx], a1[idx])
        )

        if (i + 1) % 200 == 0:
            print(f"Bootstrap: {i + 1}/1000")

    report = {
        "population": {
            "proteins": 4731,
            "go_mf_labels": 494,
            "bootstrap_replicates": 1000,
            "bootstrap_seed": 42,
            "bootstrap_unit": "protein",
        },
        "primary_reliability": {
            **primary,
            "delta_AP_95CI": interval(reliability_deltas),
        },
        "arbitration": {
            **arbitration,
            "delta_micro_AP_95CI": interval(
                arbitration_micro_deltas
            ),
            "delta_macro_AP_95CI": interval(
                arbitration_macro_deltas
            ),
        },
    }

    expected = {
        "primary_delta": 0.111614032,
        "arbitration_micro_delta": 0.034784218,
        "arbitration_macro_delta": 0.045425730,
    }

    checks = {
        "primary_delta": abs(
            primary["delta_AP"] - expected["primary_delta"]
        ) < 1e-6,
        "arbitration_micro_delta": abs(
            arbitration["delta_micro_AP"]
            - expected["arbitration_micro_delta"]
        ) < 1e-6,
        "arbitration_macro_delta": abs(
            arbitration["delta_macro_AP"]
            - expected["arbitration_macro_delta"]
        ) < 1e-6,
    }

    report["checks"] = checks

    if not all(checks.values()):
        raise RuntimeError(f"Published result mismatch: {checks}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "primary_reproduction_report.json"

    path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nREPRODUCTION RESULTS")
    print("=" * 72)
    print(json.dumps(report, indent=2))
    print("\nReport:", path)
    print("All published point-estimate checks passed.")


if __name__ == "__main__":
    main()
