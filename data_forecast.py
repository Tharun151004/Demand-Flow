#!/usr/bin/env python3
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(
        description="Wrapper to run TimeLLM retail forecasting for any N-day horizon")
    p.add_argument(
        "horizon",
        type=int,
        help="How many days ahead to forecast (e.g. 1, 7, 30, etc.)"
    )
    args = p.parse_args()

    N = args.horizon
    # Define your history window and overlap strategy:
    # – here I use at least 30 days of history, or 3× the horizon, whichever is larger.
    seq_len   = max(30, 3 * N)
    label_len = N
    pred_len  = N

    cmd = [
        sys.executable, "run_main.py",
        "--task_name",     "long_term_forecast",
        "--is_training",   "1",
        "--model_id",      f"retail_dyn_{N}d",
        "--model_comment", f"{N}day",
        "--model",         "TimeLLM",
        "--data",          "retail",
        "--root_path",     "./dataset",
        "--data_path",     "retail_inventory.csv",
        "--features",      "S",
        "--target",        "inventory_level",
        "--freq",          "d",
        "--seq_len",       str(seq_len),
        "--label_len",     str(label_len),
        "--pred_len",      str(pred_len),
        "--enc_in",        "1",
        "--dec_in",        "1",
        "--c_out",         "1",
        "--d_model",       "64",
        "--n_heads",       "4",
        "--e_layers",      "2",
        "--d_layers",      "1",
        "--d_ff",          "128",
        "--dropout",       "0.1",
        "--batch_size",    "16",
        "--learning_rate", "0.001",
        "--train_epochs",  "20",
        "--patience",      "5",
        "--num_workers",   "0",
        "--use_amp",
    ]

    print("Launching command:\n  " + " \\\n    ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()