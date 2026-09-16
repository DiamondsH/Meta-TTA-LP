# Meta-TTA-LP — Core Implementation

Minimal reviewer-facing source for **FCN-LP pretraining → second-order MAML →
LLM-anchor-guided test-time label propagation**. Only the full method is run.
No datasets, pretrained weights, experimental results, figures, search scripts,
or remote-experiment infrastructure are distributed here.

## 1. Protocol and provenance

This release is derived from the current `run_metatta.py` used by the complete
three-seed training runs of September 14, 2026. Its SHA-256 is
`e70f80d3592748730a56b3afea0f1df78d094fcf61e62f72d2c91e2f6836054d`.
The FCN-LP backbone and propagation implementation are retained as necessary
components; this is not the older Reptile implementation.

**The pretraining protocol has deliberately changed:**

- **Pretraining:** run every configured epoch and keep the parameters immediately
  after the **last** optimizer update. There is no validation split, early
  stopping, intermediate checkpoint reload, or test-metric checkpoint selection.
  The method's existing internal training-event/node grouping for supervision
  and MMD is unchanged.
- **Meta-training:** freeze the backbone and learn LPN parameters using
  second-order MAML (`create_graph=True`). The initialization is selected using
  the recorded **training-task outer loss**, retaining the corresponding
  post-update parameters, as in the source implementation.
- **Adaptation:** reset LPN parameters to that initialization for each test event;
  run a fixed number of updates using entropy, signed consistency, LLM anchors,
  and sign stability. No true labels are supplied to adaptation.
- **Evaluation:** use fixed `argmax` predictions. Test ground truth is first read
  **after** the checkpoint and predictions have been written. It is never used
  to select an epoch, model candidate, threshold, or hyperparameter. No code path
  or configuration switch enables such selection.

Earlier historical experiments used a different checkpoint-selection protocol
and, in some cases, different candidate/decision rules or implementations. This
release is therefore **not an exact reproduction of historical best single-run
scores or of the existing manuscript tables**. The supplied numerical settings
are inherited fixed configurations; they are not claimed to be optimal or newly
validated under this last-epoch protocol. No manuscript or historical results
were changed as part of packaging. Only small synthetic training/regression
checks and real-data loading checks were performed for this release; full
benchmark training under the new protocol has not been run.

Model objectives and their gradient calculations are retained, including the
source's cross-entropy on probability inputs and MMD on cached, detached features
in the meta outer loss. That fixed-feature MMD affects the recorded outer loss,
not parameter gradients. Packaging also adds input checks and index-safe
handling of isolated nodes and confidence-filtered pseudo-label rows; it does
not change the anchor scoring formula or introduce ground-truth-based filtering.

## 2. Files and dependencies

| File | Role |
|---|---|
| `metatta_core.py` | Input loading, backbone, propagation, losses, pretraining, MAML, anchors and TTA |
| `train.py` | Standalone full-method entrypoint and final evaluation |
| `config.json` | Fixed dataset-specific numerical settings |
| `requirements.txt` | Direct runtime dependencies |
| `README.md` | Input contract, execution and protocol documentation |

Use Python **3.10** for the tested environment. The source environment uses Python 3.10.20,
PyTorch 2.7.1 (CUDA 12.6 build), PyG 2.6.1, NumPy 2.2.6 and pandas 2.3.0.
In a separate environment, install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Choose a PyTorch build compatible with your device/driver. The source imports
only `torch_geometric.data.Data` and `torch_geometric.nn.GCNConv` from PyG;
external baseline packages, CLIP, an LLM API, and plotting libraries are not
required. CPU execution is supported; the full training budgets are intended
for suitably provisioned hardware.

## 3. Input contract (data supplied separately)

Provide already-preprocessed files in the following layout:

```text
DATA_ROOT/
  <dataset>/
    dataforGCN_train.csv
    dataforGCN_test.csv
    TweetEmbeds.pt
    TweetGraph.pt
PSEUDO_ROOT/
  <dataset>/
    pseudo_labels_output_deepseek.csv
```

`PSEUDO_ROOT` defaults to `DATA_ROOT`. No features or graphs are regenerated.
Load only tensors from trusted preprocessing sources; tensor loading uses
`weights_only=True`.

- Training CSV requires the dataset ID column, `event`, and binary raw `label`.
  The ID columns are `post_id` (Twitter), `mid` (PHEME), and `image_id` (Weibo).
- Test CSV requires its ID column and `event`. Its optional `label` column is
  excluded from the training loader and read only for final evaluation.
  Without it, prediction still completes and evaluation is skipped.
- IDs must be nonempty and unique within each split, with no train/test overlap.
  Event names cannot be missing. Twitter and PHEME use the exact training event
  names listed in the code. Weibo retains the original seeded internal node
  grouping. Both classes must be present in each pretraining MMD group.
- `TweetEmbeds.pt` is a finite dense `[N, D]` tensor. Its rows must be **all train
  CSV rows followed by all test CSV rows**, in exactly that order.
- `TweetGraph.pt` is a finite sparse COO `[N, N]` tensor in the same node order.
  As in the original implementation, its nonzero coordinates define edges;
  backbone edge weights are learned from an initialization of ones, rather
  than initialized from the stored graph values. Propagation uses row 0 as
  destination and row 1 as source.
- Pseudo-label CSV requires its ID column, `pseudo_label`, and numeric
  `confidence`. Its IDs/order must exactly match the test CSV; mismatches fail
  rather than silently reordering nodes. Raw pseudo labels use the same 0/1
  convention. Missing values or `pseudo_label=-1` are unavailable targets;
  negative confidence rows are excluded. Existing CSV confidence values are
  on a 0–100 scale; the retained minimum confidence is 0. Extra columns,
  including any true labels, are not passed to training or adaptation.

**Transductive graph:** training, MAML, and TTA may use the full supplied graph
and features. Test-node true labels are zero placeholders during training; all
true-label tensors are zeroed before TTA. This does not claim inductive graph
isolation. IDs alone cannot verify that externally supplied features were
constructed in the stated row order; that remains the preprocessing contract.

### Label and metric definitions

Raw `label=1` maps to model **class 0**; raw `label=0` maps to **class 1**.
The output probability columns follow that order. `argmax` breaks exact ties in
favor of class 0. Binary Precision/Recall/F1 treat **class 1 / raw label 0** as
positive. In the project's PHEME encoding this is **non-rumour**, not rumour
F1. Macro-F1 and raw-label-1 F1 are reported separately to avoid ambiguity.
Metrics are fractions, not percentages; zero-denominator precision/recall/F1
are reported as zero.

## 4. Run the full method

Run from this directory, or invoke `train.py` using its absolute path:

```bash
python train.py --dataset pheme --dataset-root /path/to/preprocessed_data --output-dir /path/to/new_run/pheme_seed42 --seed 42 --device cuda:0
```

A Windows example (single command):

```powershell
python train.py --dataset pheme --dataset-root "D:\your_preprocessed_data" --output-dir "D:\MetaTTA_Runs\pheme_seed42" --seed 42 --device cuda:0
```

Change `--dataset` to `twitter` or `weibo` to select its preset. Use
`--pseudo-label-root` when pseudo labels are stored elsewhere, `--config` for a
separate configuration file, or `--device cpu` for a small CPU run.
The default seed is 42; CUDA is not silently replaced by CPU. Unknown config
keys and invalid settings are rejected. Every run starts from a fresh model.

Default training budgets:

| Dataset | Pretraining epochs | Hidden width | MAML episodes | Inner steps | TTA steps |
|---|---:|---:|---:|---:|---:|
| Twitter | 5000 | 32 | 30 | 3 | 10 |
| PHEME | 6000 | 64 | 60 | 1 | 50 |
| Weibo | 5000 | 32 | 60 | 1 | 10 |

All numerical settings are explicit in `config.json`. In the supplied presets,
`anchor_budget=null` uses per-event ratio/minimum anchor selection. An explicit
global budget first chooses at most B valid anchors; per-event `anchor_min`
padding can increase the final total beyond B. Set `anchor_min=0` when a strict
global cap is intended. `entropy_gamma` gates entropy only in `low`/`high` mode;
`all` uses every eligible node.

For a smoke run, copy the config **outside** this package, reduce `epochs`,
`meta_episodes` and `tta_steps` to small positive values, and pass that file via
`--config`. Such a run checks execution, not predictive performance.

## 5. Runtime outputs and reproducibility

`--output-dir` is required and must be new or empty and outside this package.
Existing run files are never overwritten. Runtime-only outputs are:

- `checkpoint.pt`: final pretraining model weights, separately stored
  training-outer-loss-selected MAML parameters, configuration, seed and stage
  metadata. Per-event adapted parameters are temporary, not a single global
  adapted checkpoint. No optimizer-resume or historical checkpoint loading is
  provided.
- `predictions.csv`: sample ID/event, two probabilities, fixed predicted class
  and corresponding raw label; no ground-truth column.
- `metrics.json`: one final evaluation, or `metrics: null` if labels are absent.
- `run_config.json` and `training.log`: resolved settings and training trace.

The script seeds Python, NumPy and PyTorch, requests deterministic algorithms,
disables TF32, and resets the TTA RNG to `(seed + 1000) mod 2**32` to match the
recent full-method training convention. Identical results across different
hardware, PyTorch builds or preprocessing versions are not guaranteed.

This source package has no dependency on the original project directory and
contains no data, runtime output, credentials or machine-specific paths.
