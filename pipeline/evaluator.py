# pipeline/evaluator.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pipeline.data_loader import load_and_clean

def run_evaluation(exp, verbose=False):
    """
    For an Experiment with:
      - exp.trained_base: dict[name → (model, feature_list)]
      - exp.trained_meta: dict[name → model] (may be empty)
      - exp.test_files: List[Path]
    Compute metrics for each model/combo, save to JSON.
    """
    results = []
    for fp in exp.test_files:
        if verbose: print(f"[EVAL] Loading test set {fp.name}")
        df = load_and_clean(fp)
        for name,(mdl,features) in exp.trained_base.items():
            X = df[features]; y = df["Label"]
            y_pred = mdl.predict(X)
            results.append({
                "experiment": exp.name,
                "data_file": fp.name,
                "model_type": "base",
                "combo": name,
                "accuracy": accuracy_score(y,y_pred),
                "f1":       f1_score(y,y_pred),
                "precision":precision_score(y,y_pred),
                "recall":   recall_score(y,y_pred)
            })
            if verbose:
                print(f"  [base] {name}: F1={results[-1]['f1']:.3f}")

        # meta predictions
        if exp.trained_meta:
            # assemble meta‐DataFrame for this test df
            base_preds = {
                nm: mdl.predict_proba(df[feat])[:,1]
                for nm,(mdl,feat) in exp.trained_base.items()
            }
            meta_df = pd.DataFrame(base_preds)
            # if raw meta‐features included during training, add them here
            if exp.config["include_meta_features"]:
                for cat in exp.config["meta_models"][list(exp.config["meta_models"].keys())[0]]:
                    meta_df[cat] = df[cat].values
            y = df["Label"]
            for mn,mdl in exp.trained_meta.items():
                y_pred = mdl.predict(meta_df.values if hasattr(meta_df,"values") else meta_df)
                results.append({
                    "experiment": exp.name,
                    "data_file": fp.name,
                    "model_type": "meta",
                    "combo": mn,
                    "accuracy": accuracy_score(y,y_pred),
                    "f1":       f1_score(y,y_pred),
                    "precision":precision_score(y,y_pred),
                    "recall":   recall_score(y,y_pred)
                })
                if verbose:
                    print(f"  [meta] {mn}: F1={results[-1]['f1']:.3f}")

    # write out
    out_dir = Path("results") / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"[EVAL] Saved metrics → {out_file}")
    return results
