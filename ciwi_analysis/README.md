# CIWI-CKT analysis code

Figures, tables, theory validation and LaTeX sections for the CIWI-CKT revision.
Everything runs from one command:

```bash
pip install -r requirements.txt
python run_all.py --root /Users/s5273738/WAVEFSL_Journal
```

`--root` is the project folder. It must contain:

```
data/metr-la/dataset.npy     data/metr-la/matrix.npy
data/pems-bay/dataset.npy    data/pems-bay/matrix.npy
data/chengdu_m/dataset.npy   data/chengdu_m/matrix.npy
data/shenzhen/dataset.npy    data/shenzhen/matrix.npy
config.yaml                  (optional; read for his_num, pred_num, target_days)
```

Outputs go to `<root>/results/`, and the theory validation to `<root>/results/theory_checks/`.

## Files

| File | Purpose |
|---|---|
| `run_all.py` | Entry point; runs the steps below in order |
| `chaos_from_raw.py` | Offline chaos indicators (Lyapunov, permutation entropy, Higuchi fractal dimension) from raw windows |
| `run_all_exports.py` | Exports from your trained models; fill in its `Adapter` class first |
| `export_artifacts.py` | Helper functions used by the exports |
| `make_figures.py` | Data, chaos, wave, statistics, transfer, stability and calibration figures; dataset, efficiency and alternatives tables; `figure_numbers.tex` |
| `theory_checks.py` | Numerical validation of Lemmas 1-5, Theorems 1-3 and Propositions 1-2; `theory_numbers.tex` |
| `ablation_embedding.py` | Representation-level ablation of the chaos proxy (t-SNE plus within- vs unseen-city R²) |
| `latex/` | Manuscript sections and pre-built tables (see below) |

Run a subset of steps with `--steps`, e.g. `python run_all.py --root ... --steps theory --plot-only` to redraw the theory figures from saved CSVs (about 10 s instead of about 4 min).

## What runs now and what needs trained models

**From the raw data only (final):** `data_protocol`, `data_characteristics`, `chaos_feature_analysis` (2 of 3 panels), `wave_interference_visualization` (spectrum panel), `lemma_validation`, `theorem_validation`, `ablation_embedding`, `dataset_table.tex`, `efficiency_table.tex`.

**Needs trained-model exports:** chaos panel (c), wave panels (b)-(d), `statistical_significance_testing`, `cross_city_transfer_analysis`, `theoretical_validation` (trained checks), `input_stability`, `predictability_calibration`, and the values in `alternatives_table.tex`. Until then, the panels are skipped (never simulated), the alternatives table shows `--` and number macros show **??**.

To produce them, implement the `Adapter` methods in `run_all_exports.py` and rerun `run_all.py`. Unimplemented methods are skipped with a message.

## LaTeX

Add to the preamble:

```latex
\input{results/theory_checks/theory_numbers.tex}
\input{results/figure_numbers.tex}
```

| File | Use |
|---|---|
| `latex/section_theoretical_analysis.tex` | Corrected theory section (changes marked `% FIX:`) |
| `latex/section_theoretical_validation.tex` | Section "Theoretical Validation" |
| `latex/section_empirical_analysis.tex` | Section "Empirical Analysis" (RQ-tagged subsections) |
| `latex/experiments_block_replacement.tex` | Replaces the old efficiency, long-horizon, chaos, transfer, wave and statistics block |
| `latex/dataset_table_full.tex` | Stand-alone dataset table (alternative to the `\input` version) |
| `latex/tables/*.tex` | Complete tabulars written by `make_figures.py` |

Figures are referenced as `figs/results/<name>.pdf`; copy them from `<root>/results/` and `<root>/results/theory_checks/`.

## Settings to confirm against your training code

* `make_figures.py`: `SPLIT = (0.7, 0.1, 0.2)` (config.yaml has `split_ratio: 0.8`), `FEWSHOT_AT_END_OF_TRAIN = True`, `HORIZONS` (Chengdu/Shenzhen evaluated at 10/30/60 min).
* `run_all_exports.py`: `HORIZON_STEPS` must match the table columns.
* Normalisation statistics should come from the training split only.

## Font size

`make_figures.set_style(scale)` scales all text. `theory_checks.py` and `ablation_embedding.py` use `1.3`; the other figures use `1.0`.
