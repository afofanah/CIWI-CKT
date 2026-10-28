#!/usr/bin/env python3
"""Run the whole CIWI-CKT analysis pipeline with one command.

    python run_all.py --root /path/to/WAVEFSL_Journal            # every step
    python run_all.py --root ... --steps figures theory           # selected steps
    python run_all.py --root ... --steps theory --plot-only       # redraw theory figures only

Steps, in order:
    chaos     offline chaos indicators from the raw data            (chaos_from_raw.py)
    exports   trained-model exports; skipped until the Adapter in
              run_all_exports.py is implemented                     (run_all_exports.py)
    figures   data, chaos, wave, statistics, transfer, stability,
              calibration figures and the LaTeX tables              (make_figures.py)
    theory    numerical validation of Lemmas 1-5, Theorems 1-3 and
              Propositions 1-2                                      (theory_checks.py)
    ablation  representation-level ablation of the chaos proxy      (ablation_embedding.py)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {
    "chaos": "chaos_from_raw.py",
    "exports": "run_all_exports.py",
    "figures": "make_figures.py",
    "theory": "theory_checks.py",
    "ablation": "ablation_embedding.py",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("CIWI_ROOT", "/Users/s5273738/WAVEFSL_Journal"),
                    help="project folder containing data/<dataset>/dataset.npy and matrix.npy")
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    ap.add_argument("--plot-only", action="store_true", help="theory step: redraw from saved CSVs")
    a = ap.parse_args()

    root = Path(a.root).expanduser().resolve()
    results = root / "results"
    env = dict(os.environ, CIWI_ROOT=str(root), CIWI_RESULTS=str(results),
               CIWI_ARTIFACTS=str(results / "artifacts"), CIWI_CONFIG=str(root / "config.yaml"))
    missing = [k for k in ("metr-la", "pems-bay", "chengdu_m", "shenzhen")
               if not (root / "data" / k / "dataset.npy").exists()]
    if missing:
        print(f"[warn] data not found for: {', '.join(missing)} (expected {root}/data/<key>/dataset.npy)")

    failed = []
    for step in [s for s in STEPS if s in a.steps]:
        cmd = [sys.executable, str(HERE / STEPS[step])]
        if step == "theory" and a.plot_only:
            cmd.append("--plot-only")
        print(f"\n=== {step}: {STEPS[step]} ===", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, env=env, cwd=str(HERE))
        print(f"=== {step}: {'done' if rc == 0 else f'FAILED (exit {rc})'} in {time.time() - t0:.0f}s ===")
        if rc:
            failed.append(step)

    print(f"\nOutputs: {results}  (theory: {results / 'theory_checks'})")
    if failed:
        print(f"Failed steps: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
