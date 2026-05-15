#!/usr/bin/env python3
"""
train_detector.py — Train the ensemble IDS pipeline.

Examples
--------
python scripts/train_detector.py
python scripts/train_detector.py --dataset data/processed/dataset.npz --report
python scripts/train_detector.py --n-flows 8000 --attack-prob 0.35 --n-estimators 300
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cps_defender.core.logging_setup import setup_logging
from cps_defender.core.models import FEATURE_NAMES
from cps_defender.ids.pipeline import IDSPipeline
from cps_defender.ids.feature_extractor import FeatureExtractor
from cps_defender.testbed.traffic_sim import TrafficSimulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CPS/ICS Defender IDS pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",      default=None,
                   help=".npz or .csv dataset (auto-generate if omitted)")
    p.add_argument("--n-flows",      type=int,   default=5_000)
    p.add_argument("--attack-prob",  type=float, default=0.30)
    p.add_argument("--test-split",   type=float, default=0.20)
    p.add_argument("--model-out",    default="data/models/ids_pipeline.joblib")
    p.add_argument("--n-estimators", type=int,   default=200)
    p.add_argument("--report",       action="store_true")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_dataset(path: str) -> tuple:
    p = Path(path)
    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        return d["X"].astype(np.float32), d["y"].astype(str)
    df = pd.read_csv(p)
    return df[FEATURE_NAMES].values.astype(np.float32), df["label"].values.astype(str)


def main() -> None:
    args = parse_args()
    setup_logging("INFO" if args.verbose else "WARNING")
    np.random.seed(args.seed)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────
    if args.dataset:
        print(f"[+] Loading dataset: {args.dataset}")
        X, y_labels = load_dataset(args.dataset)
        ext   = FeatureExtractor()
        flows = None  # raw flows not available from pre-built dataset
    else:
        print(f"[+] Generating {args.n_flows:,} synthetic flows …")
        sim   = TrafficSimulator(seed=args.seed, attack_probability=args.attack_prob)
        flows = sim.generate(n_flows=args.n_flows)
        ext   = FeatureExtractor()
        X     = ext.fit_transform(flows)
        y_labels = np.array([f.label for f in flows])

    print(f"    Dataset: {X.shape[0]:,} flows × {X.shape[1]} features")
    unique, counts = np.unique(y_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u:25s}: {c}")

    # ── Split ──────────────────────────────────────────────────────
    if args.test_split > 0:
        rng   = np.random.default_rng(args.seed)
        idx   = rng.permutation(len(X))
        split = int(len(X) * (1 - args.test_split))
        idx_tr, idx_te = idx[:split], idx[split:]
        print(f"    Train: {len(idx_tr):,}   Test: {len(idx_te):,}")
    else:
        idx_tr = np.arange(len(X)); idx_te = np.array([], dtype=int)

    # ── Train ──────────────────────────────────────────────────────
    print(f"\n[+] Training IDSPipeline (RF n_estimators={args.n_estimators}) …")
    pipeline = IDSPipeline()

    # Override n_estimators on the ML sub-detector
    pipeline.ml_detector.model.estimators_ if hasattr(
        pipeline.ml_detector, 'model') else None
    # Directly set n_estimators before training
    pipeline.ml_detector.n_estimators = args.n_estimators

    t0 = time.perf_counter()
    if flows is not None:
        # Train on FlowRecord objects (preferred path — uses internal extractor)
        train_flows = [flows[i] for i in idx_tr]
        pipeline.train(train_flows)
    else:
        # Train from pre-extracted arrays — wrap in minimal FlowRecord-like objects
        from cps_defender.ids.feature_extractor import FeatureExtractor as FE
        train_flows = _make_mock_flows(X[idx_tr], y_labels[idx_tr])
        pipeline.train(train_flows)
    elapsed = time.perf_counter() - t0
    print(f"    Training time: {elapsed:.2f}s")

    # ── Evaluate ───────────────────────────────────────────────────
    if len(idx_te) > 0:
        if flows is not None:
            test_flows = [flows[i] for i in idx_te]
        else:
            test_flows = _make_mock_flows(X[idx_te], y_labels[idx_te])

        print(f"\n[+] Evaluating on {len(test_flows):,} held-out flows …")
        tp = fp = fn = tn = 0
        for flow in test_flows:
            alert = pipeline.analyze(flow)
            is_attack = flow.label != "normal"
            if alert:
                if is_attack: tp += 1
                else:          fp += 1
            else:
                if is_attack: fn += 1
                else:          tn += 1

        acc = (tp + tn) / max(1, tp + fp + fn + tn)
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        prec = tp / max(1, tp + fp)
        f1  = 2 * prec * tpr / max(1e-9, prec + tpr)
        print(f"    Accuracy  : {acc:.4f}")
        print(f"    TPR       : {tpr:.4f}   Precision: {prec:.4f}")
        print(f"    FPR       : {fpr:.4f}   F1:        {f1:.4f}")
        print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        if args.report:
            # Feature importance from ML sub-detector
            fi = pipeline.ml_detector.feature_importance()
            if fi:
                top = sorted(fi.items(), key=lambda x: -x[1])[:12]
                print("\n    Top feature importances:")
                for feat, score in top:
                    bar = "█" * int(score * 40)
                    print(f"      {feat:30s} {score:.4f}  {bar}")

    # ── Save ───────────────────────────────────────────────────────
    pipeline.save(args.model_out)
    print(f"\n[+] Saved IDSPipeline → {args.model_out}")
    print("Done.  Run 'python scripts/demo.py --ml-model "
          + args.model_out + "' to test.")


def _make_mock_flows(X: np.ndarray, y: np.ndarray):
    """Create minimal FlowRecord-like objects from a feature matrix."""
    import datetime
    from cps_defender.core.models import FlowRecord, FEATURE_NAMES
    import uuid
    flows = []
    for i, (row, label) in enumerate(zip(X, y)):
        feat = dict(zip(FEATURE_NAMES, row.tolist()))
        flow = FlowRecord(
            uid=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(),
            src_ip="10.0.1.1",
            dst_ip="10.0.0.1",
            src_port=int(feat.get("src_port", 20001)),
            dst_port=int(feat.get("dst_port", 20000)),
            protocol="DNP3",
            flow_duration=float(feat.get("flow_duration", 1.0)),
            pkt_count=int(feat.get("pkt_count", 10)),
            byte_count=int(feat.get("byte_count", 500)),
            bytes_per_pkt=float(feat.get("bytes_per_pkt", 50.0)),
            function_code=int(feat.get("function_code", 1)),
            unique_fc_count=int(feat.get("unique_fc_count", 1)),
            req_resp_ratio=float(feat.get("req_resp_ratio", 1.0)),
            inter_arrival_mean=float(feat.get("inter_arrival_mean", 0.1)),
            inter_arrival_std=float(feat.get("inter_arrival_std", 0.01)),
            is_broadcast=bool(feat.get("is_broadcast", 0) > 0.5),
            direction=int(feat.get("direction", 0)),
            burst_count=int(feat.get("burst_count", 0)),
            error_rate=float(feat.get("error_rate", 0.0)),
            label=str(label),
            label_id=0,
        )
        flows.append(flow)
    return flows


if __name__ == "__main__":
    main()
