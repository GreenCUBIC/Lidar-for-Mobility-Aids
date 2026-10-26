from pathlib import Path
import re
import statistics as stats

MODES = [
    "walking",
    "wheelchair",
    "rollatorshuffling",
    "rollatorwalking",
    "walker",
    "cane",
    "crutches",
]

SEEDS = [0, 1, 2, 3, 4]

ROOT = Path(".")
RESULT_NAME = "multiquery_test_summary.txt"
OUTPUT_FILE = ROOT / "2_person_200_end_PRETRAINED_combined_multiquery_results_5e4.txt"


def parse_summary(path):
    metrics = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                metrics[key.strip()] = value.strip()

    return {
        "n_rounds": int(metrics["n_rounds"]),
        "queries_per_id": int(metrics["queries_per_id"]),
        "avg_top1": float(metrics["avg_top1"]),
        "avg_top3": float(metrics["avg_top3"]),
        "avg_mAP": float(metrics["avg_mAP"]),
    }


rows = []
missing = []

for mode in MODES:
    for seed in SEEDS:
        result_path = (
            ROOT
            / f"{mode}_2_person_200_end_PRETRAINEDckpt_cv_5e4"
            / f"class_{mode}_200_end"
            / f"seed_{seed}"
            / "lr0.0005_optAdamW"
            / "eval_multiquery"
            / RESULT_NAME
        )

        if not result_path.exists():
            missing.append(result_path)
            continue

        metrics = parse_summary(result_path)

        rows.append({
            "mode": mode,
            "seed": seed,
            **metrics,
        })


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("Per-seed multi-query results\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Mode':<22}{'Seed':<8}{'Top-1':<12}{'Top-3':<12}{'mAP':<12}\n")
    f.write("-" * 80 + "\n")

    for row in rows:
        f.write(
            f"{row['mode']:<22}"
            f"{row['seed']:<8}"
            f"{row['avg_top1']:<12.6f}"
            f"{row['avg_top3']:<12.6f}"
            f"{row['avg_mAP']:<12.6f}\n"
        )

    f.write("\n\nMean +/- std across seeds\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Mode':<22}{'Top-1':<20}{'Top-3':<20}{'mAP':<20}\n")
    f.write("-" * 80 + "\n")

    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]

        if not mode_rows:
            continue

        top1 = [r["avg_top1"] for r in mode_rows]
        top3 = [r["avg_top3"] for r in mode_rows]
        mAP = [r["avg_mAP"] for r in mode_rows]

        def mean_std(values):
            mean = stats.mean(values)
            std = stats.stdev(values) if len(values) > 1 else 0.0
            return f"{mean:.4f} +/- {std:.4f}"

        f.write(
            f"{mode:<22}"
            f"{mean_std(top1):<20}"
            f"{mean_std(top3):<20}"
            f"{mean_std(mAP):<20}\n"
        )

    if missing:
        f.write("\n\nMissing result files\n")
        f.write("=" * 80 + "\n")
        for path in missing:
            f.write(str(path) + "\n")


print(f"Found {len(rows)} result files.")
print(f"Missing {len(missing)} result files.")
print(f"Saved summary to: {OUTPUT_FILE}")