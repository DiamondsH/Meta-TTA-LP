# Meta-TTA-LP

**Meta Test-Time Adaptive Label Propagation for Cross-Event Multimodal Fake News Detection**

Meta-TTA-LP combines episodic meta-training with event-specific test-time label
propagation. Adaptation uses entropy minimization, signed neighbor consistency,
LLM pseudo-label anchoring, and sign stability. The backbone remains frozen
during meta-training and adaptation.

## Files

| File | Description |
|---|---|
| `metatta_core.py` | Data loading, model, losses, pretraining, MAML, and test-time adaptation |
| `train.py` | Training and evaluation entrypoint |
| `config.json` | Dataset-specific hyperparameters |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview and usage |

## Installation

Use Python 3.10 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Provide preprocessed multimodal features, graph structure, and LLM pseudo-labels
separately, then run:

```bash
python train.py --dataset pheme --dataset-root /path/to/data --output-dir /path/to/outputs/pheme_seed42 --seed 42 --device cuda:0
```

- Supported datasets: `twitter`, `pheme`, and `weibo`.
- Hyperparameters are defined in `config.json`; use `--config` to specify a configuration.
- Use `--pseudo-label-root` if pseudo-labels are stored separately.
- The output directory must be new or empty and outside this code directory.

## Training

MAML selects the initialization using
training-task outer loss, followed by event-specific adaptation. Test labels
are used only for final evaluation.
