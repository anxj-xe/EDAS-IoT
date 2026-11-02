"""Model evaluation module and script for EDAS-IoT.

Evaluates anomaly detection performance against a labelled dataset with controlled anomalies.
Computes Precision, Recall, F1-Score, Confusion Matrix, False Positive Rate (FPR),
and per-anomaly-type breakdown.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from src.common.config import get_config
from src.ingestion.csv_reader import CSVReader


def evaluate_pipeline_on_dataset(
    dataset_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full pipeline evaluation on a labelled dataset."""
    config = get_config()
    root = config.root_dir

    eval_file = dataset_path or (root / "data" / "samples" / "evaluation_labeled.csv")
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found at {eval_file}. Run scripts/generate_data.py first.")

    print(f"Loading labelled evaluation dataset from: {eval_file}")
    df = pd.read_csv(eval_file)
    if "anomaly_type" not in df.columns:
        raise ValueError("Evaluation dataset must contain an 'anomaly_type' column with ground-truth labels.")

    reader = CSVReader()
    results_df, readiness, summary = reader.run_csv_pipeline(eval_file)

    # Ground truth binary: NORMAL -> 0, anything else -> 1
    y_true = (df["anomaly_type"] != "NORMAL").astype(int).to_numpy()

    # Predictions
    # 1. Overall Pipeline Anomaly (Validation + Statistical + ML)
    y_pred_overall = results_df["is_overall_anomaly"].astype(int).to_numpy()

    # 2. ML Only (Isolation Forest)
    y_pred_ml = results_df["is_ml_anomaly"].astype(int).to_numpy()

    # 3. Statistical Only
    y_pred_stat = results_df["is_statistical_anomaly"].astype(int).to_numpy()

    def compute_metrics(y_t: np.ndarray, y_p: np.ndarray) -> Dict[str, Any]:
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }

    overall_metrics = compute_metrics(y_true, y_pred_overall)
    ml_metrics = compute_metrics(y_true, y_pred_ml)
    stat_metrics = compute_metrics(y_true, y_pred_stat)

    # Per-anomaly-type recall breakdown
    type_breakdown = {}
    for anomaly_type in sorted(df["anomaly_type"].unique()):
        mask = (df["anomaly_type"] == anomaly_type)
        total_type = int(mask.sum())
        detected_overall = int(results_df.loc[mask, "is_overall_anomaly"].sum())
        detected_ml = int(results_df.loc[mask, "is_ml_anomaly"].sum())
        detected_stat = int(results_df.loc[mask, "is_statistical_anomaly"].sum())

        type_breakdown[anomaly_type] = {
            "total_samples": total_type,
            "overall_detected": detected_overall,
            "overall_detection_rate": round(detected_overall / max(1, total_type), 3),
            "ml_detected": detected_ml,
            "stat_detected": detected_stat,
        }

    report = {
        "dataset": eval_file.name,
        "total_records": len(df),
        "normal_records": int((df["anomaly_type"] == "NORMAL").sum()),
        "anomalous_records": int((df["anomaly_type"] != "NORMAL").sum()),
        "overall_pipeline": overall_metrics,
        "isolation_forest_ml": ml_metrics,
        "statistical_detector": stat_metrics,
        "anomaly_type_breakdown": type_breakdown,
    }

    # Save report
    out_file = output_path or (root / "data" / "processed" / "evaluation_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=======================================================")
    print("           EDAS-IoT MODEL EVALUATION REPORT           ")
    print("=======================================================")
    print(f"Dataset: {eval_file.name} ({len(df)} total samples)")
    print(f"Normal Samples: {report['normal_records']} | Anomalous Samples: {report['anomalous_records']}")
    print("\n--- OVERALL PIPELINE (Validation + Statistical + ML) ---")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall:    {overall_metrics['recall']:.4f}")
    print(f"  F1-Score:  {overall_metrics['f1_score']:.4f}")
    print(f"  FPR:       {overall_metrics['false_positive_rate']:.4f}")
    print(f"  Confusion Matrix: TP={overall_metrics['confusion_matrix']['true_positives']}, FP={overall_metrics['confusion_matrix']['false_positives']}, FN={overall_metrics['confusion_matrix']['false_negatives']}, TN={overall_metrics['confusion_matrix']['true_negatives']}")

    print("\n--- ISOLATION FOREST ML ALONE ---")
    print(f"  Precision: {ml_metrics['precision']:.4f} | Recall: {ml_metrics['recall']:.4f} | F1: {ml_metrics['f1_score']:.4f}")

    print("\n--- STATISTICAL DETECTOR ALONE ---")
    print(f"  Precision: {stat_metrics['precision']:.4f} | Recall: {stat_metrics['recall']:.4f} | F1: {stat_metrics['f1_score']:.4f}")

    print("\n--- ANOMALY TYPE DETECTION RATES ---")
    for atype, stats in type_breakdown.items():
        if atype != "NORMAL":
            print(f"  • {atype:25s}: {stats['overall_detected']}/{stats['total_samples']} ({stats['overall_detection_rate']*100:.1f}%) [ML: {stats['ml_detected']}, Stat: {stats['stat_detected']}]")

    print(f"\nDetailed evaluation report saved to: {out_file}")
    return report


if __name__ == "__main__":
    evaluate_pipeline_on_dataset()
