#!/usr/bin/env python3
"""Train Full Meta-TTA-LP from scratch and evaluate once after prediction."""

import os
import sys

# Set these before importing torch. Do not create caches inside the code package.
sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import metatta as core


CONFIG_KEYS = {
    "epochs", "hidden", "num_classes", "dropout", "gcn_layers", "lr",
    "weight_decay", "log_every", "meta_episodes", "maml_inner_steps",
    "maml_inner_lr", "maml_outer_lr", "maml_max_nodes", "meta_mmd_lambda",
    "tta_steps", "tta_lr", "anchor_beta", "anchor_ratio", "anchor_min",
    "anchor_budget", "anchor_conf_thresh", "entropy_gamma", "entropy_gamma_mode",
    "consistency_lambda", "anchor_lambda", "consistency_margin", "sign_lambda",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(core.ID_COLUMNS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pseudo-label-root", type=Path,
                        help="Defaults to --dataset-root.")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().with_name("config.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0", help="cpu or cuda:N; no implicit CPU fallback.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty directory outside this code package.")
    return parser.parse_args(argv)


def load_config(path, dataset):
    configs = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(configs, dict) or dataset not in configs:
        raise ValueError(f"Config must contain an entry for {dataset}.")
    config = configs[dataset]
    if not isinstance(config, dict):
        raise ValueError("Dataset configuration must be an object.")
    missing, extra = CONFIG_KEYS - set(config), set(config) - CONFIG_KEYS
    if missing or extra:
        raise ValueError(f"Invalid configuration keys: missing={sorted(missing)}, extra={sorted(extra)}")
    positive_ints = (
        "epochs", "hidden", "num_classes", "gcn_layers", "log_every",
        "meta_episodes", "maml_inner_steps", "maml_max_nodes", "tta_steps",
    )
    for key in positive_ints:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["num_classes"] != 2 or config["gcn_layers"] < 2:
        raise ValueError("This implementation requires num_classes=2 and gcn_layers>=2.")
    if type(config["anchor_min"]) is not int or config["anchor_min"] < 0:
        raise ValueError("anchor_min must be a nonnegative integer.")
    if config["anchor_budget"] is not None:
        if type(config["anchor_budget"]) is not int or config["anchor_budget"] < 0:
            raise ValueError("anchor_budget must be null or a nonnegative integer.")
    if config["entropy_gamma_mode"] not in ("all", "low", "high"):
        raise ValueError("entropy_gamma_mode must be all, low, or high.")
    numeric = CONFIG_KEYS - set(positive_ints) - {
        "anchor_min", "anchor_budget", "entropy_gamma_mode",
    }
    for key in numeric:
        value = config[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number.")
    for key in ("lr", "maml_inner_lr", "maml_outer_lr", "tta_lr"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be strictly positive.")
    if config["dropout"] >= 1:
        raise ValueError("dropout must be in [0, 1).")
    for key in ("anchor_beta", "anchor_ratio", "anchor_conf_thresh"):
        if config[key] > 1:
            raise ValueError(f"{key} must be in [0, 1].")
    return config


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_logger(output_dir):
    logger = logging.getLogger("metatta")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    for handler in (
        logging.FileHandler(output_dir / "training.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def evaluate_fixed_predictions(test_csv, id_column, sample_ids, predictions):
    """Read test truth ONLY after predictions have been fixed and written.

    There is no threshold argument, checkpoint feedback, candidate comparison,
    or parameter update. If labels are unavailable, inference remains usable.
    """
    columns = pd.read_csv(test_csv, nrows=0).columns
    if "label" not in columns:
        return None
    truth = pd.read_csv(test_csv, usecols=[id_column, "label"], dtype={id_column: str})
    if truth[id_column].tolist() != list(sample_ids):
        raise ValueError("Test CSV IDs/order changed before final evaluation.")
    if not truth.label.isin([0, 1]).all():
        raise ValueError("Final evaluation requires binary raw labels 0 and 1.")
    labels = 1 - truth.label.to_numpy(dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if pred.shape != labels.shape or not np.isin(pred, [0, 1]).all():
        raise ValueError("Predicted classes do not match the test node list.")
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    def divide(a, b):
        return a / b if b else 0.0

    f1 = divide(2 * tp, 2 * tp + fp + fn)
    class0_f1 = divide(2 * tn, 2 * tn + fp + fn)
    return {
        "accuracy": divide(tp + tn, len(labels)),
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "f1": f1, "macro_f1": (f1 + class0_f1) / 2,
        "raw_label_1_f1": class0_f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n_test": len(labels),
        "positive_class": 1, "positive_raw_label": 0, "units": "fraction",
    }


def run(cli):
    config = load_config(cli.config, cli.dataset)
    if not 0 <= cli.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    device = torch.device(cli.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda:N.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a compatible torch build or specify --device cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device does not exist.")
    output_dir = cli.output_dir.resolve()
    package_dir = Path(__file__).resolve().parent
    if output_dir.is_relative_to(package_dir):
        raise ValueError("Choose an output directory outside the code package.")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Output directory must be new or empty; existing runs are never overwritten.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(output_dir)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    core.setup_seed(cli.seed)
    args = SimpleNamespace(**config)
    dataset_root = cli.dataset_root.resolve()
    pseudo_root = (cli.pseudo_label_root or cli.dataset_root).resolve()
    write_json(output_dir / "run_config.json", {
        "method": "Full Meta-TTA-LP", "dataset": cli.dataset, "seed": cli.seed,
        "device": str(device), "torch_version": str(torch.__version__),
        "parameters": config, "pretrain_rule": "last_epoch",
        "meta_rule": "training_outer_loss", "decision_rule": "argmax",
        "dataset_root": str(dataset_root), "pseudo_label_root": str(pseudo_root),
    })
    logger.info("Protocol: last-epoch pretraining; training-outer-loss MAML; fixed full TTA; final evaluation only.")
    data, train_data, test_data, pseudo = core.load_dataset(
        dataset_root, pseudo_root, cli.dataset, device,
    )
    logger.info(f"Loaded {len(train_data)} training nodes, {len(test_data)} test nodes, {data.num_edges} edges.")
    model, pretrain_info = core.train_fcnlp(
        args, cli.dataset, data, train_data, device, logger,
    )
    meta_params, meta_info = core.maml_meta_train(
        args, cli.dataset, model, data, train_data, device, logger,
    )
    # MAML uses a separate parameter mapping. The backbone/model below still
    # contains the FINAL pretraining parameters, not a selected earlier epoch.
    torch.save({
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "meta_params": {k: v.detach().cpu().clone() for k, v in meta_params.items()},
        "pretrain": pretrain_info, "meta": meta_info, "config": config,
        "dataset": cli.dataset, "seed": cli.seed,
    }, output_dir / "checkpoint.pt")
    data.y.zero_()  # No true labels (including training labels) are needed by TTA.
    model.eval()
    with torch.no_grad():
        base_out, _, x_cache = model(data)
    core.setup_seed((cli.seed + 1000) % (2**32))
    probs = core.test_time_adapt_probs(
        args, model, meta_params, data, test_data, pseudo, device, logger,
        base_out=base_out.detach(), x_cache=x_cache.detach(),
    )[data.test_mask].detach().cpu().numpy()
    if (probs.shape != (len(test_data), 2) or not np.isfinite(probs).all()
            or (probs < 0).any() or (probs > 1).any()
            or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)):
        raise FloatingPointError("Final predictions are not finite, normalized two-class probabilities.")
    pred = probs.argmax(axis=1)
    id_column = core.ID_COLUMNS[cli.dataset]
    sample_ids = test_data[id_column].tolist()
    pd.DataFrame({
        "sample_id": sample_ids, "event": test_data.event.tolist(),
        "probability_class_0": probs[:, 0], "probability_class_1": probs[:, 1],
        "predicted_class": pred, "predicted_raw_label": 1 - pred,
    }).to_csv(output_dir / "predictions.csv", index=False)
    # This is the first and only place test truth enters the program.
    metrics = evaluate_fixed_predictions(
        dataset_root / cli.dataset / "dataforGCN_test.csv", id_column, sample_ids, pred,
    )
    write_json(output_dir / "metrics.json", {
        "method": "Full Meta-TTA-LP", "dataset": cli.dataset, "seed": cli.seed,
        "decision_rule": "argmax", "metrics": metrics,
    })
    if metrics is None:
        logger.info("Predictions saved; test CSV has no label column, so evaluation was skipped.")
    else:
        logger.info("Final evaluation: " + json.dumps(metrics, allow_nan=False))
    logger.info(f"Complete: {output_dir}")
    return metrics


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
