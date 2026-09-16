"""Core Meta-TTA-LP: fixed-final-epoch pretraining, second-order MAML, and TTA.

Derived from run_metatta.py (source SHA-256:
e70f80d3592748730a56b3afea0f1df78d094fcf61e62f72d2c91e2f6836054d).
This release has no test-metric checkpoint/candidate/threshold selection.
Training labels are stored only for training nodes. TTA receives no test truth.
See README.md for the protocol change and the input/label conventions.
"""

import math
import random
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

ID_COLUMNS = {"twitter": "post_id", "pheme": "mid", "weibo": "image_id"}


SELECTED_EVENTS = {
    "twitter": [
        "boston",
        "columbianChemicals",
        "nepal",
        "pigFish",
        "bringback",
        "sochi",
        "malaysia",
        "sandy",
        "passport",
        "underwater",
        "livr",
    ],
    "pheme": ["Ottawa Shooting", "sydney siege", "Charlie Hebdo", "GermanwingsCrash"],
}
UNSELECTED_EVENTS = {
    "twitter": ["elephant", "garissa", "eclipse", "samurai"],
    "pheme": ["Ferguson"],
}

MAX_EDGES_META = 20000
MAX_EDGES_TTA = 30000
MIN_EVENT_SIZE = 4


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot_labels(label_lists):
    # FCN-LP convention: raw label 1 → class index 0, raw label 0 → class index 1
    all_labels = torch.tensor([int(l) for ll in label_lists for l in ll], dtype=torch.long)
    class_idx = torch.where(all_labels == 1, torch.zeros_like(all_labels), torch.ones_like(all_labels))
    return F.one_hot(class_idx, num_classes=2).float()


def get_data_splits(label_list_train, event_list_train, selected_events, unselected_events):
    event_map = {}
    for i, (label, event) in enumerate(zip(label_list_train, event_list_train)):
        if event not in event_map:
            event_map[event] = [[], []]
        event_map[event][0].append(i) if int(label) == 1 else event_map[event][1].append(i)
    seen_real, seen_fake, unseen_real, unseen_fake = [], [], [], []
    for event in selected_events:
        seen_real.extend(event_map[event][0])
        seen_fake.extend(event_map[event][1])
    for event in unselected_events:
        unseen_real.extend(event_map[event][0])
        unseen_fake.extend(event_map[event][1])
    return seen_real, seen_fake, unseen_real, unseen_fake


def make_seen_unseen(dataset, label_list_train, event_list_train):
    if dataset == "weibo":
        all_tweets = list(range(len(label_list_train)))
        unseen = set(random.sample(all_tweets, len(label_list_train) // 3))
        seen = list(set(all_tweets) - unseen)
        seen_real = [idx for idx in seen if int(label_list_train[idx]) == 1]
        seen_fake = [idx for idx in seen if int(label_list_train[idx]) == 0]
        unseen_real = [idx for idx in unseen if int(label_list_train[idx]) == 1]
        unseen_fake = [idx for idx in unseen if int(label_list_train[idx]) == 0]
    else:
        seen_real, seen_fake, unseen_real, unseen_fake = get_data_splits(
            label_list_train, event_list_train, SELECTED_EVENTS[dataset], UNSELECTED_EVENTS[dataset]
        )
        seen = seen_real + seen_fake
    return seen, seen_real, seen_fake, unseen_real, unseen_fake


def _compute_metrics(preds_cls, labels_cls):
    """Core metrics computation from hard class-index tensors."""
    tp = torch.sum(preds_cls * labels_cls)
    fp = torch.sum(preds_cls * (1 - labels_cls))
    fn = torch.sum((1 - preds_cls) * labels_cls)
    tn = torch.sum((1 - preds_cls) * (1 - labels_cls))
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return acc, precision, recall, f1


def accuracy(output, labels):
    preds_cls = output.max(1)[1]
    labels_cls = labels.max(1)[1]
    return _compute_metrics(preds_cls, labels_cls)


def raw_label_to_class_index(raw_labels):
    # FCN-LP encodes raw label=1 as class index 0, and raw label=0 as class index 1.
    return torch.where(raw_labels.long() == 1, torch.zeros_like(raw_labels), torch.ones_like(raw_labels)).long()


class MMDLoss(nn.Module):
    def forward(self, source, target):
        delta = source.float().mean(0) - target.float().mean(0)
        return delta.dot(delta.T)


class FCNLP(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout, num_edges, gcn_layers):
        super().__init__()
        self.edge_weight = nn.Parameter(torch.ones(num_edges))
        self.gc = nn.ModuleList([GCNConv(in_feature, hidden)])
        for _ in range(gcn_layers - 2):
            self.gc.append(GCNConv(hidden, hidden))
        self.gc.append(GCNConv(hidden, out_feature))
        self.lpn_lin = nn.Linear(hidden, hidden)
        self.lpn_alpha = nn.Linear(2 * hidden, out_feature)
        self.dropout_rate = dropout
        self.reset_lpn_parameters()

    def reset_lpn_parameters(self):
        nn.init.xavier_uniform_(self.lpn_lin.weight)
        nn.init.zeros_(self.lpn_lin.bias)
        nn.init.xavier_uniform_(self.lpn_alpha.weight)
        nn.init.zeros_(self.lpn_alpha.bias)

    def encode(self, x, edge_index):
        for conv in self.gc[:-1]:
            x = conv(x, edge_index, self.edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)
        logits = self.gc[-1](x, edge_index, self.edge_weight)
        return x, logits

    def lpn_params(self):
        return OrderedDict(
            [
                ("lin_weight", self.lpn_lin.weight),
                ("lin_bias", self.lpn_lin.bias),
                ("alpha_weight", self.lpn_alpha.weight),
                ("alpha_bias", self.lpn_alpha.bias),
            ]
        )

    def forward_with_lpn_params(self, data, params=None, x_cache=None, out_cache=None, edge_index=None):
        if x_cache is None or out_cache is None:
            x, logits = self.encode(data.x, data.edge_index)
            out = F.softmax(logits, dim=1)
        else:
            x, out = x_cache, out_cache
        if params is None:
            params = self.lpn_params()
        if edge_index is None:
            edge_index = data.edge_index
        y_hat = lpn_propagate(x, out.detach(), edge_index, params)
        return out, F.softmax(y_hat, dim=1), x

    def forward(self, data):
        return self.forward_with_lpn_params(data)


def lpn_propagate(x, label, edge_index, params):
    z = F.linear(x, params["lin_weight"], params["lin_bias"])
    # PyG MessagePassing receives SparseTensor adjacency in transposed form here.
    # This direction matches the original FCN-LP LPAconv output exactly.
    dst, src = edge_index[0], edge_index[1]
    alpha_in = torch.cat([z[dst], z[src]], dim=-1)
    alpha = torch.tanh(F.linear(alpha_in, params["alpha_weight"], params["alpha_bias"]))
    msg = label[src] * alpha
    out = torch.zeros_like(label)
    out.index_add_(0, dst, msg)
    return out


def load_dataset(dataset_root, pseudo_label_root, dataset, device):
    """Load fixed graph inputs, without reading the test CSV's label column.

    Node order is train CSV rows followed by test CSV rows. No sorting or
    label-based reordering is performed. IDs and pseudo-label rows must align.
    Test rows in data.y are zero placeholders, NOT one-hot ground truth.
    """
    id_column = ID_COLUMNS[dataset]
    root = Path(dataset_root) / dataset
    pseudo_root = Path(pseudo_label_root) / dataset
    train = pd.read_csv(
        root / "dataforGCN_train.csv", usecols=[id_column, "event", "label"],
        dtype={id_column: str, "event": str},
    )
    test = pd.read_csv(
        root / "dataforGCN_test.csv", usecols=[id_column, "event"],
        dtype={id_column: str, "event": str},
    )
    pseudo = pd.read_csv(
        pseudo_root / "pseudo_labels_output_deepseek.csv",
        usecols=[id_column, "pseudo_label", "confidence"], dtype={id_column: str},
    )
    if train.empty or test.empty:
        raise ValueError("Training and test node lists must both be nonempty.")
    for name, frame in (("train", train), ("test", test), ("pseudo", pseudo)):
        if frame[id_column].isna().any() or frame[id_column].str.strip().eq("").any():
            raise ValueError(f"Missing {id_column} in {name} CSV.")
        if frame[id_column].duplicated().any():
            raise ValueError(f"Duplicate {id_column} in {name} CSV.")
    if set(train[id_column]) & set(test[id_column]):
        raise ValueError("Training and test sample IDs overlap.")
    if test[id_column].tolist() != pseudo[id_column].tolist():
        raise ValueError("Pseudo-label IDs/order must exactly match the test CSV.")
    for name, frame in (("train", train), ("test", test)):
        if frame.event.isna().any() or frame.event.str.strip().eq("").any():
            raise ValueError(f"Missing event in {name} CSV.")
    if not train.label.isin([0, 1]).all() or train.label.nunique() != 2:
        raise ValueError("Training labels must contain both binary raw labels 0 and 1.")
    train["label"] = train.label.astype(int)
    if dataset in SELECTED_EVENTS:
        required_events = set(SELECTED_EVENTS[dataset] + UNSELECTED_EVENTS[dataset])
        missing = required_events - set(train.event)
        if missing:
            raise ValueError(f"Missing required training events: {sorted(missing)}")
    pseudo["pseudo_label"] = pd.to_numeric(pseudo.pseudo_label, errors="raise")
    pseudo["confidence"] = pd.to_numeric(pseudo.confidence, errors="raise")
    if not pseudo.pseudo_label.dropna().isin([-1, 0, 1]).all():
        raise ValueError("Pseudo labels must be 0/1; -1 or NaN denotes a missing label.")
    if not np.isfinite(pseudo.confidence.dropna().to_numpy()).all():
        raise ValueError("Pseudo confidence must be finite or missing.")
    # Existing CSVs use (-1, -1) for failed LLM responses. Keep the row/ID,
    # but never reinterpret its missing pseudo label as a binary target.
    pseudo.loc[pseudo.pseudo_label == -1, "pseudo_label"] = np.nan

    x = torch.load(root / "TweetEmbeds.pt", map_location="cpu", weights_only=True)
    graph = torch.load(root / "TweetGraph.pt", map_location="cpu", weights_only=True)
    n_train, n_test = len(train), len(test)
    n = n_train + n_test
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided:
        raise ValueError("TweetEmbeds.pt must contain a dense tensor.")
    x = x.float()
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] == 0:
        raise ValueError("Feature rows must equal train+test CSV rows, with nonzero feature width.")
    if not torch.isfinite(x).all():
        raise ValueError("Features contain NaN or infinity.")
    if not isinstance(graph, torch.Tensor) or graph.layout != torch.sparse_coo:
        raise ValueError("TweetGraph.pt must contain a sparse COO tensor.")
    graph = graph.float().coalesce()
    if tuple(graph.shape) != (n, n) or graph.sparse_dim() != 2 or graph.dense_dim() != 0:
        raise ValueError("Graph shape must be (train+test rows, train+test rows).")
    if graph._nnz() == 0 or not torch.isfinite(graph.values()).all():
        raise ValueError("Graph must have at least one edge and finite values.")
    edges = graph.indices()
    if int(edges.min()) < 0 or int(edges.max()) >= n:
        raise ValueError("Graph endpoint is outside the feature row range.")
    y = torch.zeros((n, 2), dtype=torch.float32)
    y[:n_train] = one_hot_labels([train.label.tolist()])
    data = Data(
        x=x, edge_index=edges, y=y,
        train_mask=torch.arange(n) < n_train,
        test_mask=torch.arange(n) >= n_train,
    ).to(device)
    return data, train, test, pseudo


def train_fcnlp(args, dataset, data, train_data, device, logger):
    """Run every configured pretraining epoch and return the FINAL parameters.

    No test metric is computed here. No checkpoint is compared or reloaded.
    The original supervised/MMD event grouping and optimization are retained.
    """
    seen, seen_real, seen_fake, unseen_real, unseen_fake = make_seen_unseen(
        dataset, train_data.label.tolist(), train_data.event.tolist()
    )
    if any(len(group) == 0 for group in (seen_real, seen_fake, unseen_real, unseen_fake)):
        raise ValueError("Pretraining MMD requires both classes in each internal event/node group.")
    model = FCNLP(
        data.x.size(1), args.hidden, args.num_classes, args.dropout,
        data.num_edges, args.gcn_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()
    mmd = MMDLoss()
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, yhat, x = model(data)
        loss = ce(out[seen], data.y[seen]) + ce(yhat[seen], data.y[seen])
        loss = loss + mmd(x[unseen_real], x[seen_real]) + mmd(x[unseen_fake], x[seen_fake])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite pretraining loss at epoch {epoch}.")
        acc_train, _, _, _ = accuracy(yhat[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % args.log_every == 0 or epoch == args.epochs:
            logger.info(
                f"[pretrain] epoch={epoch}/{args.epochs} loss={loss.item():.6f} "
                f"train_acc={acc_train.item():.6f} time={time.time() - start:.2f}s"
            )
    model.eval()
    info = {"selection": "last_epoch", "epoch": args.epochs, "train_loss": loss.item()}
    logger.info(f"[pretrain] Using final epoch {args.epochs}; no checkpoint selection.")
    return model, info


def subset_edges(edge_index, node_idx, max_edges=None):
    device = edge_index.device
    node_idx = node_idx.to(device)
    # Include isolated high-index nodes, not just the largest existing endpoint.
    extent = max(int(node_idx.max().item()), int(edge_index.max().item()) if edge_index.numel() else -1) + 1
    mask_nodes = torch.zeros(extent, dtype=torch.bool, device=device)
    mask_nodes[node_idx] = True
    edge_mask = mask_nodes[edge_index[0]] & mask_nodes[edge_index[1]]
    sub_edges = edge_index[:, edge_mask]
    if sub_edges.numel() == 0:
        # Self-loops keep the differentiable path valid for sparse events.
        sub_edges = torch.stack([node_idx, node_idx], dim=0)
    if max_edges is not None and sub_edges.size(1) > max_edges:
        perm = torch.randperm(sub_edges.size(1), device=device)[:max_edges]
        sub_edges = sub_edges[:, perm]
    return sub_edges


def event_indices(df, offset=0):
    out = {}
    for event, group in df.groupby("event"):
        out[str(event)] = torch.tensor(group.index.to_numpy() + offset, dtype=torch.long)
    return out


def sample_nodes(idx, max_nodes, device):
    idx = idx.to(device)
    if idx.numel() <= max_nodes:
        return idx
    perm = torch.randperm(idx.numel(), device=device)[:max_nodes]
    return idx[perm]


def entropy_loss(yhat, gamma, node_idx=None, mode="all"):
    """Entropy minimization loss.

    Args:
        yhat: soft predictions, shape [N, C]
        gamma: threshold (used in 'low' and 'high' modes)
        node_idx: optional subset of nodes
        mode:
            'all'  — standard entropy minimization on ALL nodes (default, recommended)
            'low'  — only minimize entropy of nodes with ent < gamma (original behavior)
            'high' — only minimize entropy of nodes with ent > gamma (uncertain nodes)
    """
    if node_idx is not None:
        yhat = yhat[node_idx]
    ent = -(yhat.clamp_min(1e-8) * yhat.clamp_min(1e-8).log()).sum(dim=1)
    if mode == "all":
        return ent.mean()
    elif mode == "low":
        mask = ent < gamma
        if mask.any():
            return ent[mask].mean()
        return ent.mean() * 0.0
    elif mode == "high":
        mask = ent > gamma
        if mask.any():
            return ent[mask].mean()
        return ent.mean() * 0.0
    else:
        raise ValueError(f"Unknown entropy_gamma_mode: {mode}")


def edge_sign_from_alpha(alpha):
    """Signed score: clamp(alpha[:, 1] - alpha[:, 0], -1, 1)."""
    return (alpha[:, 1] - alpha[:, 0]).clamp(-1.0, 1.0)


def compute_edge_alpha(x, edge_index, params):
    """Compute per-edge alpha from LPN parameters (used for sign-based losses)."""
    z = F.linear(x, params["lin_weight"], params["lin_bias"])
    dst, src = edge_index[0], edge_index[1]
    alpha_in = torch.cat([z[dst], z[src]], dim=-1)
    alpha = torch.tanh(F.linear(alpha_in, params["alpha_weight"], params["alpha_bias"]))
    return alpha


def signed_consistency_loss(yhat, edge_index, edge_sign, margin=1.0, max_edges=MAX_EDGES_META):
    """Signed neighbor consistency: pull same-sign edges together, push opposite-sign apart.

    edge_sign: tensor of shape [E] in [-1, 1]
      - positive (sign > 0): positive correlation → predictions should be close (pull)
      - negative (sign < 0): negative correlation → predictions should be far (push, hinge margin)
    The distance is the squared L2 distance. Positive edges use distance * |sign|;
    negative edges use max(0, margin - distance) * |sign|. Each sign group is averaged.
    """
    if edge_index.size(1) > max_edges:
        perm = torch.randperm(edge_index.size(1), device=edge_index.device)[:max_edges]
        edge_index = edge_index[:, perm]
        edge_sign = edge_sign[perm]
    src, dst = edge_index[0], edge_index[1]
    diff = ((yhat[src] - yhat[dst]) ** 2).sum(dim=1)  # [E]
    abs_sign = edge_sign.abs()
    pos_mask = edge_sign > 0
    neg_mask = edge_sign < 0

    loss_pos = torch.tensor(0.0, device=yhat.device)
    loss_neg = torch.tensor(0.0, device=yhat.device)

    if pos_mask.any():
        # Pull positive edges together (weighted by sign magnitude)
        loss_pos = (diff[pos_mask] * abs_sign[pos_mask]).mean()
    if neg_mask.any():
        # Push negative edges apart (hinge/margin loss, weighted by sign magnitude)
        hinge = torch.clamp(margin - diff[neg_mask], min=0.0)
        loss_neg = (hinge * abs_sign[neg_mask]).mean()

    return loss_pos + loss_neg


def sign_stability_loss(alpha_prev, alpha_curr):
    """Sign consistency regularization: penalize large changes in edge sign pattern.

    Uses cosine similarity between flattened prev and current alpha vectors.
    Returns (1 - cos_sim) so it is a loss to minimize.
    """
    a1 = alpha_prev.flatten()
    a2 = alpha_curr.flatten()
    cos_sim = F.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0)).squeeze()
    return 1.0 - cos_sim


def conflict_edge_degree(yhat, edge_index):
    """For each node, count how many incident edges connect nodes of opposite predicted class.

    Returns per-node conflict edge count tensor (float).
    """
    pred = yhat.argmax(dim=1)
    src, dst = edge_index[0], edge_index[1]
    conflict = (pred[src] != pred[dst]).float()
    degree = torch.zeros(pred.size(0), device=yhat.device)
    degree.index_add_(0, src, conflict)
    degree.index_add_(0, dst, conflict)
    return degree


def anchor_loss(yhat, anchors, pseudo_targets):
    if anchors is None or anchors.numel() == 0:
        return yhat.sum() * 0.0
    return F.mse_loss(yhat[anchors], pseudo_targets)


def tta_loss(yhat, edge_index, edge_sign, loss_args, anchors=None, pseudo_targets=None, node_idx=None):
    """TTA terms: entropy + signed consistency + anchor (stability is added by the caller).

    edge_sign: per-edge sign score in [-1, 1] (from current LPN alpha).
    loss_args: namespace with entropy_gamma, consistency_lambda, anchor_lambda, consistency_margin.
    """
    loss = entropy_loss(yhat, loss_args.entropy_gamma, node_idx=node_idx,
                        mode=getattr(loss_args, "entropy_gamma_mode", "all"))
    margin = getattr(loss_args, "consistency_margin", 1.0)
    loss = loss + loss_args.consistency_lambda * signed_consistency_loss(
        yhat, edge_index, edge_sign, margin=margin,
    )
    if anchors is not None and anchors.numel() > 0:
        loss = loss + loss_args.anchor_lambda * anchor_loss(yhat, anchors, pseudo_targets)
    return loss


def maml_meta_train(args, dataset, model, data, train_data, device, logger):
    """Second-order MAML; select the initialization using TRAINING outer loss.

    The backbone is frozen. Both query supervision and source event sampling
    use training rows only. No test metric or test label is available.
    """
    for p in model.gc.parameters():
        p.requires_grad_(False)
    model.edge_weight.requires_grad_(False)
    model.eval()
    with torch.no_grad():
        base_out, _, x_cache = model(data)
    train_events = event_indices(train_data, offset=0)
    eligible = [e for e, idx in train_events.items() if idx.numel() >= MIN_EVENT_SIZE]
    if not eligible:
        raise ValueError("No training event has enough nodes for MAML.")

    meta_params = OrderedDict(
        (k, v.requires_grad_(True)) for k, v in clone_lpn_params(model.lpn_params(), device).items()
    )
    outer_opt = torch.optim.Adam(list(meta_params.values()), lr=args.maml_outer_lr)
    mmd_loss = MMDLoss()
    best_loss = float("inf")
    best_ep = 0
    best_meta_state = None
    start = time.time()
    inner_loss_args = SimpleNamespace(
        entropy_gamma=args.entropy_gamma,
        entropy_gamma_mode=getattr(args, "entropy_gamma_mode", "all"),
        consistency_lambda=args.consistency_lambda,
        anchor_lambda=0.0,
        consistency_margin=getattr(args, "consistency_margin", 1.0),
    )
    meta_mmd_lambda = getattr(args, "meta_mmd_lambda", 0.1)
    for ep in range(1, args.meta_episodes + 1):
        # Leave-one-event-out: pick one event as target (query), rest as source (support)
        target_event = random.choice(eligible)
        source_events = [e for e in eligible if e != target_event]
        # Sample target event nodes for inner/outer loop
        target_idx = sample_nodes(train_events[target_event], args.maml_max_nodes, device)
        target_edges = subset_edges(data.edge_index, target_idx, max_edges=MAX_EDGES_META)
        # Sample a subset of source event nodes (combined) for MMD / outer supervision
        src_node_list = []
        for se in source_events:
            src_node_list.append(train_events[se])
        source_idx = torch.cat(src_node_list).to(device) if src_node_list else torch.empty(0, dtype=torch.long, device=device)
        if source_idx.numel() > args.maml_max_nodes:
            perm = torch.randperm(source_idx.numel(), device=device)[: args.maml_max_nodes]
            source_idx = source_idx[perm]

        # Inner loop: unsupervised TTA adaptation on target event → fast parameters
        fast = OrderedDict((k, v) for k, v in meta_params.items())
        for _ in range(args.maml_inner_steps):
            y_inner = F.softmax(lpn_propagate(x_cache, base_out.detach(), target_edges, fast), dim=1)
            alpha_inner = compute_edge_alpha(x_cache, target_edges, fast)
            sign_inner = edge_sign_from_alpha(alpha_inner)
            inner_loss = tta_loss(y_inner, target_edges, sign_inner, inner_loss_args, node_idx=target_idx)
            grads = torch.autograd.grad(inner_loss, tuple(fast.values()), create_graph=True)
            fast = OrderedDict(
                (k, v - args.maml_inner_lr * g) for (k, v), g in zip(fast.items(), grads)
            )
        # Outer loop: supervised CE on target (query) + MMD alignment between source & target features
        y_query = F.softmax(lpn_propagate(x_cache, base_out.detach(), target_edges, fast), dim=1)
        outer_loss = F.cross_entropy(y_query[target_idx], data.y[target_idx].argmax(dim=1))
        # MMD on LPN input features (x_cache) between source and target events
        if source_idx.numel() > 0 and target_idx.numel() > 0:
            outer_loss = outer_loss + meta_mmd_lambda * (
                mmd_loss(x_cache[source_idx], x_cache[target_idx])
            )
        if not torch.isfinite(outer_loss):
            raise FloatingPointError(f"Non-finite MAML outer loss at episode {ep}.")
        outer_opt.zero_grad()
        outer_loss.backward()
        outer_opt.step()
        if outer_loss.item() < best_loss:
            best_loss = outer_loss.item()
            best_ep = ep
            best_meta_state = clone_lpn_params(meta_params, device)
        if ep % args.log_every == 0 or ep == args.meta_episodes:
            logger.info(
                f"[maml] episode={ep:03d}/{args.meta_episodes} target={target_event} "
                f"src_events={len(source_events)} "
                f"loss={outer_loss.item():.4f} best_loss={best_loss:.4f}@{best_ep} "
                f"time={time.time() - start:.2f}s"
            )

    # Keep the original training-outer-loss checkpoint rule.
    maml_best = {
        "episodes": args.meta_episodes,
        "best_loss": best_loss,
        "best_episode": best_ep,
    }

    # Restore meta_params to best values
    meta_params = OrderedDict(
        (k, v.to(device).detach().clone().requires_grad_(True))
        for k, v in best_meta_state.items()
    )

    logger.info(
        f"\n=== {dataset} MAML Best Result === "
        f"Episode {best_ep}, best_outer_loss={best_loss:.6f}"
    )
    return meta_params, {"selection": "training_outer_loss", **maml_best}


def normalize_score(x):
    if x.numel() == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi == lo:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def select_anchors(test_idx, yhat, edge_index, pseudo_df, args, device, min_confidence=0.0):
    local_n = test_idx.numel()
    budget = max(args.anchor_min, int(math.ceil(args.anchor_ratio * local_n)))
    budget = min(budget, local_n)
    valid = pseudo_df["pseudo_label"].notna() & pseudo_df["confidence"].notna()
    valid = valid & (pseudo_df["confidence"].astype(float) >= min_confidence)
    valid_mask = torch.tensor(valid.to_numpy(), dtype=torch.bool, device=device)
    candidate_local = valid_mask.nonzero(as_tuple=False).view(-1)
    if candidate_local.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty((0, 2), device=device)
    candidate_global = test_idx[candidate_local]
    # Additional model confidence filter (keep pseudo targets aligned): only keep nodes where model is confident
    conf_thresh = getattr(args, "anchor_conf_thresh", 0.0)
    if conf_thresh > 0.0:
        max_probs = yhat[candidate_global].max(dim=1).values
        conf_mask = max_probs >= conf_thresh
        candidate_local = candidate_local[conf_mask]
        candidate_global = candidate_global[conf_mask]
        if candidate_global.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty((0, 2), device=device)
    ent = -(yhat[candidate_global].clamp_min(1e-8) * yhat[candidate_global].clamp_min(1e-8).log()).sum(dim=1)
    degree = torch.bincount(edge_index.flatten(), minlength=yhat.size(0)).float().to(device)
    deg = degree[candidate_global]
    # Conflict edge participation: how many incident edges connect nodes of opposite predicted class
    # (core indicator of "information bottleneck" nodes on the graph)
    conflict_deg = conflict_edge_degree(yhat, edge_index)
    cdeg = conflict_deg[candidate_global]
    # Three-dimensional anchor scoring per method doc: entropy + centrality + conflict degree
    score = normalize_score(ent) + normalize_score(deg) + normalize_score(cdeg)
    pseudo_raw_labels = torch.tensor(
        pseudo_df.iloc[candidate_local.detach().cpu().numpy()]["pseudo_label"].astype(int).to_numpy(), dtype=torch.long, device=device
    )
    pseudo_class_labels = raw_label_to_class_index(pseudo_raw_labels)
    top = torch.topk(score, min(budget, score.numel())).indices
    anchors = candidate_global[top]
    targets = F.one_hot(pseudo_class_labels[top], num_classes=2).float()
    return anchors, targets


def select_global_anchors(test_idx, yhat, edge_index, pseudo_df, args, device, min_confidence=0.0):
    """
    Select a global set of anchors across all test nodes (not per-event).
    Uses the same three-factor scoring (entropy + degree + conflict degree).
    Selects at most budget valid global anchors; event minima may add more later.
    Falls back to per-event ratio-based selection if anchor_budget is None.

    Returns:
        anchors: global node indices of selected anchors
        targets: one-hot pseudo-label vectors for the selected anchors
    """
    budget = getattr(args, "anchor_budget", None)
    if budget is None:
        # Fallback: use ratio-based per-event selection handled by select_anchors()
        return None, None
    total_n = test_idx.numel()
    budget = min(budget, total_n)
    valid = pseudo_df["pseudo_label"].notna() & pseudo_df["confidence"].notna()
    valid = valid & (pseudo_df["confidence"].astype(float) >= min_confidence)
    valid_mask = torch.tensor(valid.to_numpy(), dtype=torch.bool, device=device)
    candidate_local = valid_mask.nonzero(as_tuple=False).view(-1)
    if candidate_local.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty((0, 2), device=device)
    candidate_global = test_idx[candidate_local]
    # Additional model confidence filter (keep pseudo targets aligned): only keep nodes where model is confident
    conf_thresh = getattr(args, "anchor_conf_thresh", 0.0)
    if conf_thresh > 0.0:
        max_probs = yhat[candidate_global].max(dim=1).values
        conf_mask = max_probs >= conf_thresh
        candidate_local = candidate_local[conf_mask]
        candidate_global = candidate_global[conf_mask]
        if candidate_global.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty((0, 2), device=device)
    ent = -(yhat[candidate_global].clamp_min(1e-8) * yhat[candidate_global].clamp_min(1e-8).log()).sum(dim=1)
    degree = torch.bincount(edge_index.flatten(), minlength=yhat.size(0)).float().to(device)
    deg = degree[candidate_global]
    conflict_deg = conflict_edge_degree(yhat, edge_index)
    cdeg = conflict_deg[candidate_global]
    score = normalize_score(ent) + normalize_score(deg) + normalize_score(cdeg)
    pseudo_raw_labels = torch.tensor(
        pseudo_df.iloc[candidate_local.detach().cpu().numpy()]["pseudo_label"].astype(int).to_numpy(), dtype=torch.long, device=device
    )
    pseudo_class_labels = raw_label_to_class_index(pseudo_raw_labels)
    top = torch.topk(score, min(budget, score.numel())).indices
    anchors = candidate_global[top]
    targets = F.one_hot(pseudo_class_labels[top], num_classes=2).float()
    return anchors, targets


def test_time_adapt_probs(
    args, model, meta_params, data, test_data, pseudo_df, device, logger,
    base_out=None, x_cache=None,
):
    """Run only Full Meta-TTA-LP, with fixed settings and no test truth.

    Event-local parameters reset to the MAML initialization for each event.
    Only test-node predictions are consumed by the training entrypoint.
    """
    model.eval()
    if base_out is None or x_cache is None:
        with torch.no_grad():
            base_out, base_yhat, x_cache = model(data)
    else:
        base_yhat = F.softmax(base_out, dim=1)
    test_offset = int(data.train_mask.sum().item())
    events = event_indices(test_data, offset=test_offset)
    adapted = base_yhat.detach().clone()
    sign_lambda = getattr(args, "sign_lambda", 0.01)
    loss_args = SimpleNamespace(
        entropy_gamma=args.entropy_gamma,
        entropy_gamma_mode=getattr(args, "entropy_gamma_mode", "all"),
        consistency_lambda=args.consistency_lambda,
        anchor_lambda=args.anchor_lambda,
        consistency_margin=getattr(args, "consistency_margin", 1.0),
    )
    # Global anchor selection (only when anchor_budget is set)
    global_anchors = None
    global_targets = None
    if getattr(args, "anchor_budget", None) is not None:
        test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
        test_edges = subset_edges(data.edge_index, test_idx, max_edges=MAX_EDGES_TTA)
        # Compute y0 on full test graph for global scoring
        global_params = clone_lpn_params(meta_params, device)
        with torch.no_grad():
            y_global = F.softmax(lpn_propagate(x_cache, base_out.detach(), test_edges, global_params), dim=1)
        global_anchors, global_targets = select_global_anchors(
            test_idx,
            y_global,
            test_edges,
            pseudo_df,
            args,
            device,
            min_confidence=0.0,
        )
        logger.info(f"[tta] Full Meta-TTA-LP: global_anchors={global_anchors.numel()}")
    for event, idx_cpu in events.items():
        idx = idx_cpu.to(device)
        sub_edges = subset_edges(data.edge_index, idx, max_edges=MAX_EDGES_TTA)
        event_rows = test_data.index[test_data["event"].astype(str) == event].to_numpy()
        pseudo_event = pseudo_df.iloc[event_rows].reset_index(drop=True)
        params = OrderedDict(
            (k, v.requires_grad_(True)) for k, v in clone_lpn_params(meta_params, device).items()
        )
        # Optimize only the LPN parameters; backbone features remain frozen.
        with torch.no_grad():
            y0 = F.softmax(lpn_propagate(x_cache, base_out.detach(), sub_edges, params), dim=1)
            # Initial edge sign (before adaptation) for sign stability loss
            alpha_init = compute_edge_alpha(x_cache, sub_edges, params)
            sign_init = edge_sign_from_alpha(alpha_init)
        if global_anchors is not None:
            # Use anchors from global budget: keep only those within this event
            idx_set = set(idx.tolist())
            mask = torch.tensor([a.item() in idx_set for a in global_anchors], dtype=torch.bool, device=device)
            anchors = global_anchors[mask]
            targets = global_targets[mask]
            # Enforce per-event minimum (pad with event-local anchors if needed)
            if anchors.numel() < args.anchor_min and args.anchor_min > 0:
                local_anchors, local_targets = select_anchors(
                    idx, y0, sub_edges, pseudo_event, args, device,
                    min_confidence=0.0,
                )
                # Add local anchors not already in global set
                global_set = set(anchors.tolist())
                extra_idx = [i for i, a in enumerate(local_anchors.tolist()) if a not in global_set]
                if extra_idx:
                    needed = args.anchor_min - anchors.numel()
                    extra_idx = extra_idx[:needed]
                    extra = torch.tensor(extra_idx, dtype=torch.long, device=device)
                    anchors = torch.cat([anchors, local_anchors[extra]])
                    targets = torch.cat([targets, local_targets[extra]])
        else:
            anchors, targets = select_anchors(
                idx,
                y0,
                sub_edges,
                pseudo_event,
                args,
                device,
                min_confidence=0.0,
            )
        tta_steps = args.tta_steps
        tta_lr = args.tta_lr
        opt = torch.optim.Adam(list(params.values()), lr=tta_lr)
        loss = y0.sum() * 0.0
        step_losses = []
        for step in range(1, tta_steps + 1):
            y = F.softmax(lpn_propagate(x_cache, base_out.detach(), sub_edges, params), dim=1)
            alpha_curr = compute_edge_alpha(x_cache, sub_edges, params)
            sign_curr = edge_sign_from_alpha(alpha_curr)
            mixed_targets = targets
            if targets.numel() > 0:
                mixed_targets = args.anchor_beta * targets + (1.0 - args.anchor_beta) * y[anchors].detach()
            loss = tta_loss(y, sub_edges, sign_curr, loss_args, anchors, mixed_targets, node_idx=idx)
            # Sign stability: penalize drift in edge sign pattern from initial
            loss = loss + sign_lambda * sign_stability_loss(sign_init, sign_curr)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite TTA loss for event {event}, step {step}.")
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_losses.append(loss.item())
        with torch.no_grad():
            y_final = F.softmax(lpn_propagate(x_cache, base_out.detach(), sub_edges, params), dim=1)
            adapted[idx] = y_final[idx]
            logger.info(
                f"[tta] Full Meta-TTA-LP event={event} nodes={int(idx.numel())} "
                f"anchors={int(anchors.numel())} steps={tta_steps} "
                f"loss 1st={step_losses[0]:.6f} -> last={step_losses[-1]:.6f}"
            )
    return adapted


def clone_lpn_params(params, device):
    return OrderedDict((k, v.detach().clone().to(device)) for k, v in params.items())
