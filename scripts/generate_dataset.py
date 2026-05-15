#!/usr/bin/env python3
"""
generate_dataset.py — Generate synthetic CPS/ICS flow datasets.

Examples
--------
python scripts/generate_dataset.py
python scripts/generate_dataset.py --n-flows 10000 --attack-prob 0.35
python scripts/generate_dataset.py --augment --output data/processed/augmented
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cps_defender.core.logging_setup import setup_logging
from cps_defender.core.models import FEATURE_NAMES
from cps_defender.testbed.traffic_sim import TrafficSimulator
from cps_defender.ids.feature_extractor import FeatureExtractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic CPS/ICS flow datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-flows",     type=int,   default=5_000)
    p.add_argument("--attack-prob", type=float, default=0.30)
    p.add_argument("--output",      default="data/processed/dataset")
    p.add_argument("--augment",     action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--format",      choices=["csv","npz","both"], default="both")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO" if args.verbose else "WARNING")
    np.random.seed(args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[+] Generating {args.n_flows:,} flows (attack_prob={args.attack_prob}) …")
    sim   = TrafficSimulator(seed=args.seed, attack_probability=args.attack_prob)
    flows = sim.generate(n_flows=args.n_flows)

    print(f"[+] Extracting features …")
    ext = FeatureExtractor()
    X   = ext.fit_transform(flows)
    y   = np.array([f.label for f in flows])

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u:25s}: {c:5d}  ({c/len(y)*100:.1f}%)")

    if args.augment:
        print("\n[+] Applying GenAI augmentation …")
        try:
            from cps_defender.genai.augmenter import AugmentationPipeline
            aug = AugmentationPipeline(target_per_class=500, seed=args.seed)
            X, y = aug.fit_transform(X, y)
            print(f"    Augmented size: {len(y):,}")
        except Exception as e:
            print(f"    [!] Augmentation failed ({e}), skipping.")

    if args.format in ("csv", "both"):
        df  = pd.DataFrame(X, columns=FEATURE_NAMES)
        df["label"] = y
        path = output.with_suffix(".csv")
        df.to_csv(path, index=False)
        print(f"\n[+] CSV → {path}")

    if args.format in ("npz", "both"):
        path = output.with_suffix(".npz")
        np.savez_compressed(path, X=X, y=y, feature_names=np.array(FEATURE_NAMES))
        print(f"[+] NPZ → {path}")

    ext_path = output.parent / "feature_extractor.joblib"
    ext.save(str(ext_path))
    print(f"[+] FeatureExtractor → {ext_path}\nDone.")


if __name__ == "__main__":
    main()
