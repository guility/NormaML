#!/usr/bin/env python3
"""Quick data inspection script for examples/basic_usage

This script:
- loads ./data/sample.csv with automatic separator detection (pandas.read_csv sep=None, engine='python')
  and falls back to sep=',' on failure;
- writes first 20 rows preview to examples/basic_usage/data_preview.txt;
- writes a detailed summary to examples/basic_usage/data_summary.txt;
- logs progress to stdout.
Requires: pandas, numpy. If missing, prints installation instruction and exits with code 1.
"""
import sys
import os

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Required packages not found. Install with: pip install pandas numpy", flush=True)
    sys.exit(1)

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "sample.csv"))
PREVIEW_OUT = os.path.join(BASE_DIR, "data_preview.txt")
SUMMARY_OUT = os.path.join(BASE_DIR, "data_summary.txt")

def read_csv_autosep(path):
    """Try to read CSV with auto-detected separator, fallback to comma."""
    print(f"Loading data from {path} ...", flush=True)
    try:
        df = pd.read_csv(path, sep=None, engine='python')
        print("Auto-detected separator and loaded with engine='python'", flush=True)
        return df
    except Exception as e:
        print(f"Auto-detect failed ({e}), trying comma as fallback...", flush=True)
        df = pd.read_csv(path, sep=',')
        print("Loaded with sep=','", flush=True)
        return df

def detect_target_column(df):
    """Return list of columns whose names look like target/label/y/class/outcome."""
    candidates = []
    for col in df.columns:
        ncol = str(col).lower()
        if any(k in ncol for k in ('target', 'label', 'y', 'class', 'outcome')):
            candidates.append(col)
    return candidates

def write_preview(df, out_path, n=20):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("Data preview (first {} rows):\n".format(n))
        # to_string without index for nicer preview
        f.write(df.head(n).to_string(index=False))
        f.write("\n\n")
        f.write("Basic info: rows={}, columns={}\n".format(len(df), len(df.columns)))
    print(f"Saved data preview to {out_path}", flush=True)

def write_summary(df, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("Data summary\n")
        f.write("=" * 40 + "\n\n")
        f.write("Columns and dtypes:\n")
        try:
            f.write(df.dtypes.to_string())
        except Exception:
            f.write(str(df.dtypes))
        f.write("\n\nMissing values per column:\n")
        try:
            f.write(df.isna().sum().to_string())
        except Exception:
            f.write(str(df.isna().sum()))
        f.write("\n\n")

        # Numeric statistics
        num = df.select_dtypes(include=[np.number])
        if not num.empty:
            f.write("Numeric columns basic statistics (count, mean, std, min, 25%, 50%, 75%, max):\n")
            try:
                f.write(num.describe().transpose().to_string())
            except Exception:
                # Fallback to simpler representation
                for col in num.columns:
                    f.write(f"\nColumn: {col}\n")
                    f.write(str(num[col].describe()))
            f.write("\n\n")
        else:
            f.write("No numeric columns detected.\n\n")

        # Categorical top-5
        cat = df.select_dtypes(include=['object', 'category', 'bool'])
        if not cat.empty:
            f.write("Categorical columns top-5 frequent values:\n")
            for col in cat.columns:
                f.write(f"\nColumn: {col}\n")
                try:
                    vc = df[col].value_counts(dropna=False).head(5)
                    f.write(vc.to_string())
                    f.write("\n")
                except Exception as e:
                    f.write(f"Could not compute value_counts for {col}: {e}\n")
        else:
            f.write("No categorical columns detected.\n")

        # Target detection
        candidates = detect_target_column(df)
        f.write("\nDetected target-like columns:\n")
        if candidates:
            for c in candidates:
                f.write(f"- {c}\n")
            f.write("\nRecommendation: consider using the first candidate above as a target column.\n")
        else:
            f.write("None found (no column names matching 'target','label','y','class','outcome').\n")
    print(f"Saved data summary to {out_path}", flush=True)

def write_experiment_config(base_dir: str, data_path: str, targets: list[str]) -> str:
    """
    Create a simple NormaML experiment config (YAML) in the examples directory.
    Returns the path to the written config file.
    Note: This is a minimal example intended to be compatible with the project's CLI;
    adjust depending on your local NormaML config schema.
    """
    import textwrap

    cfg_path = os.path.join(base_dir, "experiment.yml")
    # Minimal example config: adjust keys as required by your NormaML installation
    targets_yaml = "\n".join(f"  - \"{t}\"" for t in targets)
    cfg = textwrap.dedent(f"""\
        # Example NormaML experiment generated by inspect_data.py
        experiment:
          name: examples_basic_usage_experiment

        data:
          path: {data_path}
          format: csv

        targets:
{targets_yaml}

        # Trainer / pipeline section below is intentionally minimal; replace with your actual pipeline config.
        pipeline:
          trainer:
            name: default
            # further trainer configuration can be added here
    """)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg)
    print(f"Wrote experiment config to {cfg_path}", flush=True)
    return cfg_path

def run_experiment_cli(config_path: str) -> int:
    """
    Run the NormaML CLI to start the experiment using the generated config.
    Returns the subprocess return code.
    """
    import subprocess
    # Use the same Python interpreter to run the package as a module
    cmd = [sys.executable, "-m", "normaml.cli", "run", "--config", config_path]
    print(f"Launching NormaML experiment: {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd, check=False)
        print(f"NormaML CLI exited with return code {proc.returncode}", flush=True)
        return proc.returncode
    except FileNotFoundError as e:
        print(f"Failed to run NormaML CLI: {e}", flush=True)
        return 127
    except Exception as e:
        print(f"Error while running NormaML CLI: {e}", flush=True)
        return 1

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Ensure './data/sample.csv' exists.", flush=True)
        sys.exit(1)

    df = read_csv_autosep(DATA_PATH)

    print("Inspecting dataframe...", flush=True)
    try:
        write_preview(df, PREVIEW_OUT, n=20)
        write_summary(df, SUMMARY_OUT)
    except Exception as e:
        print(f"Error while writing outputs: {e}", flush=True)
        sys.exit(1)

    print("Inspection completed successfully.", flush=True)

    # --- Launch NormaML experiment (targets specified by user) ---
    targets = ["Прод.тр. Всего", "Попереч. трещ."]

    try:
        cfg_path = write_experiment_config(BASE_DIR, DATA_PATH, targets)
        rc = run_experiment_cli(cfg_path)
        if rc != 0:
            print(f"Warning: NormaML run finished with non-zero exit code {rc}", flush=True)
        else:
            print("NormaML experiment launched successfully.", flush=True)
    except Exception as e:
        print(f"Could not start NormaML experiment: {e}", flush=True)

    sys.exit(0)

if __name__ == "__main__":
    main()