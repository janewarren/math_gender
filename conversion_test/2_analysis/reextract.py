"""Re-extract model_answer and recompute loss from raw_response in all result files."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "1_run_inference"))

import pandas as pd
import numpy as np
from pathlib import Path
from extractors import extract_answer, determine_answer_type, calculate_loss
REASONING_MODELS = {"gpt-5.2", "deepseek-v3.1", "qwen3-235b-thinking", "qwen3-next-thinking", "deepseek-r1"}

BASE = Path(__file__).parent / "full_results"
TOLERANCE_PCT = 0.1
TOLERANCE_MIN = 1.0

changed_total = 0
fixed_total = 0
broken_total = 0
files_processed = 0

for subdir in sorted(BASE.glob("results*")):
    for model_dir in sorted(subdir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        is_reasoning = model_name in REASONING_MODELS

        for fpath in sorted(model_dir.glob("*_converted.tsv")):
            try:
                df = pd.read_csv(fpath, sep="\t")
            except Exception as e:
                print(f"SKIP {fpath}: {e}")
                continue

            if "raw_response" not in df.columns:
                continue

            files_processed += 1
            old_ma = df["model_answer"].copy()
            old_loss = df["loss"].copy()
            n_changed = 0

            for idx, row in df.iterrows():
                raw = str(row.get("raw_response", ""))
                if not raw or raw in ("nan", "null", "None") or raw.startswith("ERROR:"):
                    continue

                domain = row["domain"]
                correct = row["answer"]
                answer_type = determine_answer_type(domain, correct)

                new_ma = extract_answer(raw, domain, correct, is_reasoning)
                new_loss = calculate_loss(new_ma, correct, answer_type, TOLERANCE_PCT, TOLERANCE_MIN)

                old_m = old_ma.iloc[idx]
                old_l = old_loss.iloc[idx]

                ma_changed = str(new_ma) != str(old_m)
                loss_changed = False
                try:
                    if pd.isna(new_loss) != pd.isna(old_l):
                        loss_changed = True
                    elif not pd.isna(new_loss) and abs(float(new_loss) - float(old_l)) > 0.001:
                        loss_changed = True
                except (ValueError, TypeError):
                    loss_changed = str(new_loss) != str(old_l)

                if ma_changed or loss_changed:
                    df.at[idx, "model_answer"] = new_ma
                    df.at[idx, "loss"] = new_loss
                    n_changed += 1

                    was_wrong = pd.notna(old_l) and float(old_l) > 0
                    now_correct = pd.notna(new_loss) and float(new_loss) == 0.0
                    was_correct = pd.notna(old_l) and float(old_l) == 0.0
                    now_wrong = pd.notna(new_loss) and float(new_loss) > 0

                    if was_wrong and now_correct:
                        fixed_total += 1
                    elif was_correct and now_wrong:
                        broken_total += 1

            if n_changed > 0:
                df.to_csv(fpath, sep="\t", index=False)
                changed_total += n_changed
                cond = subdir.name
                print(f"  {cond}/{model_name}/{fpath.name}: {n_changed} rows changed")

print(f"\n{'='*60}")
print(f"Files processed: {files_processed}")
print(f"Total rows changed: {changed_total}")
print(f"  Fixed (wrong→correct): {fixed_total}")
print(f"  Broken (correct→wrong): {broken_total}")
print(f"  Other changes: {changed_total - fixed_total - broken_total}")
