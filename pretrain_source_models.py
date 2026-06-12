#!/usr/bin/env python3
"""
pretrain_source_models.py — HP search + source pretraining for SF-UniDA.

Phase 1: Grid-search best (learning_rate, batch_size) on a small set of
         probe source domains to find stable training HP for each
         backbone × dataset combination.
         Results are cached in configs/source_hparams.json and reused on
         subsequent runs — no repeated grid search.
Phase 2: Train and save source model checkpoints for every source domain ID
         listed in SOURCE_DOMAINS using the HP found in Phase 1.

Existing valid checkpoints are skipped; corrupted ones (NaN weights) are
automatically retrained.

Usage:
    python pretrain_source_models.py --backbone MambaSSM --dataset HAR
    python pretrain_source_models.py --backbone PatchTST  --dataset EEG --skip_hp --lr 1e-3 --bs 64
    python pretrain_source_models.py --backbone CNN       --dataset HHAR --force  # overwrite existing
    python pretrain_source_models.py --backbone CNN       --dataset HAR  --redo_hp
"""

import argparse
import itertools
import json
import os
import sys

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.data_model_configs import get_dataset_class
from dataloader.dataloader import data_generator, get_label_encoder
from models.models import get_backbone_class, classifier
from utils import fix_randomness

# ── Source domain IDs — edit here to add / remove domains ──────────────────
#
# Keys are dataset names; values are the unique source domain IDs (strings)
# for which source-model checkpoints are required at SF-UniDA training time.
# These are derived from the 'src' side of each dataset's scenarios list.

# SOURCE_DOMAINS = {
#     "HAR":  ["1", "6", "9", "12", "13", "15", "17", "22", "24", "30"],
#     "HHAR": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
#     "EEG":  ["0", "3", "5", "6", "7", "9", "12", "13", "16", "18"],
# }


SOURCE_DOMAINS = {
    "HAR":  ["1", "6", "9", "12", "13", "15", "17", "22", "24", "30", "2", "7", "18", "20", "28"],
    "HHAR": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
    "EEG":  ["0", "3", "5", "6", "7", "9", "12", "13", "16", "18", "4", "8", "10"],
}

# SOURCE_DOMAINS = {
#     "HAR":  ["2", "7", "18", "20", "28"],
#     "HHAR": ["3", "6", "7"],
#     "EEG":  ["4", "8", "10"],
# }

# Probe source IDs used ONLY for HP grid search (from scenarios_hp).
# For HAR and EEG these are disjoint from SOURCE_DOMAINS.
HP_PROBE_DOMAINS = {
    "HAR":  ["2", "7", "18", "20", "28"],
    "HHAR": ["3", "6", "7"],
    "EEG":  ["4", "8", "10"],
}

# ── HP search grid ──────────────────────────────────────────────────────────
LR_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
#BS_GRID = [32, 64, 128]
BS_GRID = [64]


HP_EPOCHS  = 20   # short training for HP selection
SRC_EPOCHS = 100  # full source pretraining

WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 5.0
PATIENCE     = 5    # early stopping: epochs without test-acc improvement
TEMPERATURE  = 2.0  # logit temperature during training (> 1 softens predictions)

# Path where best HP are cached after each grid search.
HP_CACHE_PATH = os.path.join("configs", "source_hparams.json")


# ── HP cache helpers ─────────────────────────────────────────────────────────

def _load_hp_cache():
    if os.path.exists(HP_CACHE_PATH):
        with open(HP_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_hp_cache(cache):
    os.makedirs(os.path.dirname(HP_CACHE_PATH), exist_ok=True)
    with open(HP_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_hparams_obj(backbone, dataset):
    """Return an instantiated hparams object for the given backbone × dataset."""
    if backbone == "TSLANet":
        from configs.hparams_TSLANet import get_hparams_class as _f
    elif backbone == "S3Layer":
        from configs.hparams_S3Layer import get_hparams_class as _f
    elif backbone == "FNO":
        from configs.hparams_FNO import get_hparams_class as _f
    elif backbone == "CNN":
        from configs.hparams_CNN import get_hparams_class as _f
    elif backbone == "Mantis":
        from configs.hparams_Mantis import get_hparams_class as _f
    elif backbone == "Moment":
        from configs.hparams_Moment import get_hparams_class as _f
    elif backbone == "TimesNet":
        from configs.hparams_TimesNet import get_hparams_class as _f
    elif backbone == "Chronos":
        from configs.hparams_Chronos import get_hparams_class as _f
    elif backbone == "PatchTST":
        from configs.hparams_PatchTST import get_hparams_class as _f
    elif backbone in ("Mamba", "MambaFast", "MambaSSM"):
        from configs.hparams_Mamba import get_hparams_class as _f
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return _f(dataset)()


def _build_network(backbone, configs, device):
    """Build a source model: Sequential(feature_extractor, classifier).

    Mirrors the network structure in Algorithm.__init__ so that the saved
    state_dict() keys match what load_source_model() expects.
    """
    backbone_class = get_backbone_class(backbone)
    fe  = backbone_class(configs)
    clf = classifier(configs)
    net = nn.Sequential(fe, clf)
    return net.to(device)


def _is_corrupt(ckpt_path):
    """Return True if the checkpoint cannot be loaded or contains NaN weights."""
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return any(
            v.is_floating_point() and torch.isnan(v).any()
            for v in sd.values()
        )
    except Exception:
        return True


def _get_all_source_ids(dataset_name):
    """Derive all unique source IDs from dataset_configs.scenarios (dynamic)."""
    cfg = get_dataset_class(dataset_name)()
    return sorted({src for src, _ in cfg.scenarios}, key=lambda x: int(x))

# src_ids = _get_all_source_ids(dataset)  # dynamic alternative — kept for future use


# ── Training helpers ─────────────────────────────────────────────────────────

def _prepare_source(data_path, src_id, configs):
    """Mirror abstract_trainer.load_data() for source-only setup.

    Removes private source classes and builds a label encoder so that
    num_classes and label indices match what SF-UniDA algorithms expect
    when they call load_source_model().

    Returns a deepcopy of configs (num_classes updated), the encoder array,
    and the list of private source class indices.
    """
    cfg = copy.deepcopy(configs)
    pri_cl_src = (cfg.private_classes[0]["src"]
                  if hasattr(cfg, "private_classes") and cfg.private_classes
                  else [])
    if pri_cl_src:
        encoder = get_label_encoder(data_path, src_id, cfg, pri_cl_src, "train")
    else:
        encoder = None
    return cfg, encoder, pri_cl_src


def _make_loader(data_path, src_id, cfg, encoder, pri_cl, batch_size, dtype="train"):
    hparams = {"batch_size": batch_size, "weight_decay": WEIGHT_DECAY}
    return data_generator(data_path, src_id, cfg, hparams, encoder, pri_cl, dtype, src=(dtype == "train"))


def _train_epoch(net, loader, optimizer, device, temperature=1.0):
    net.train()
    total_loss, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.float().to(device), y.long().to(device)
        loss = F.cross_entropy(net(x) / temperature, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def _evaluate(net, loader, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.float().to(device), y.long().to(device)
            preds = net(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


# ── Phase 1: HP grid search ──────────────────────────────────────────────────

def hp_search(backbone, dataset, device, data_path, configs):
    """Grid-search (lr, bs) on probe source domains. Returns (best_lr, best_bs)."""
    probe_ids = HP_PROBE_DOMAINS.get(dataset, [])
    # probe_ids = _get_all_source_ids(dataset)  # dynamic alternative — kept for future use

    if not probe_ids:
        print(f"  [HP] No probe domains defined for {dataset}. Using defaults lr=1e-3, bs=64.")
        return 1e-3, 64

    best_lr, best_bs, best_acc = None, None, -1.0

    for lr, bs in itertools.product(LR_GRID, BS_GRID):
        run_accs = []
        for src_id in probe_ids:
            fix_randomness(0)
            try:
                cfg, encoder, pri_cl = _prepare_source(data_path, src_id, configs)
                train_loader = _make_loader(data_path, src_id, cfg, encoder, pri_cl, bs, dtype="train")
                test_loader  = _make_loader(data_path, src_id, cfg, encoder, pri_cl, bs, dtype="test")
            except Exception as e:
                print(f"  [HP] Skipping probe {dataset}/{src_id}: {e}")
                continue

            net = _build_network(backbone, cfg, device)
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
            for _ in range(HP_EPOCHS):
                _train_epoch(net, train_loader, opt, device, TEMPERATURE)
            acc = _evaluate(net, test_loader, device)
            run_accs.append(acc)
            del net

        if not run_accs:
            continue

        avg_acc = sum(run_accs) / len(run_accs)
        print(f"  [HP] lr={lr:.0e}  bs={bs:3d}  avg_test_acc={avg_acc:.4f}")
        if avg_acc > best_acc:
            best_acc, best_lr, best_bs = avg_acc, lr, bs

    print(f"  [HP] Best: lr={best_lr:.0e}, bs={best_bs}  (test_acc={best_acc:.4f})")
    return best_lr, best_bs


# ── Phase 2: full source pretraining ────────────────────────────────────────

def pretrain_one(backbone, dataset, src_id, lr, bs, device, data_path, configs, ckpt_path):
    """Train with early stopping (PATIENCE) + temperature; saves best test-acc weights."""
    fix_randomness(42)
    cfg, encoder, pri_cl = _prepare_source(data_path, src_id, configs)
    train_loader = _make_loader(data_path, src_id, cfg, encoder, pri_cl, bs, dtype="train")
    test_loader  = _make_loader(data_path, src_id, cfg, encoder, pri_cl, bs, dtype="test")
    net = _build_network(backbone, cfg, device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_acc, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, SRC_EPOCHS + 1):
        loss = _train_epoch(net, train_loader, opt, device, TEMPERATURE)
        acc  = _evaluate(net, test_loader, device)

        if acc > best_acc:
            best_acc   = acc
            best_state = copy.deepcopy(net.state_dict())
            no_improve = 0
            print(f"    [{epoch:3d}/{SRC_EPOCHS}] loss={loss:.4f}  test_acc={acc:.4f}  [best]")
        else:
            no_improve += 1
            if epoch % 10 == 0:
                print(f"    [{epoch:3d}/{SRC_EPOCHS}] loss={loss:.4f}  test_acc={acc:.4f}  (no improvement: {no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    net.load_state_dict(best_state)
    print(f"  Best test accuracy: {best_acc:.4f}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(net.state_dict(), ckpt_path)
    print(f"  Saved → {ckpt_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pretrain source models for SF-UniDA methods")
    parser.add_argument("--backbone",  required=True,
                        help="Backbone name (CNN, MambaSSM, PatchTST, …)")
    parser.add_argument("--dataset",   required=True,
                        help="Dataset name (HAR, HHAR, EEG, …)")
    parser.add_argument("--data_path", default="data",
                        help="Root data directory (dataset name is appended)")
    parser.add_argument("--device",    default="cuda",
                        help="torch device string (cuda / cpu)")
    parser.add_argument("--skip_hp",   action="store_true",
                        help="Skip HP search and use --lr / --bs directly")
    parser.add_argument("--redo_hp",   action="store_true",
                        help="Ignore cached HP and run grid search again")
    parser.add_argument("--lr",        type=float, default=None,
                        help="Learning rate (used when --skip_hp is set)")
    parser.add_argument("--bs",        type=int,   default=None,
                        help="Batch size (used when --skip_hp is set)")
    parser.add_argument("--force",     action="store_true",
                        help="Retrain all domains even if a valid checkpoint exists")
    args = parser.parse_args()

    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(args.data_path, args.dataset)

    configs = get_dataset_class(args.dataset)()

    # Apply backbone-specific config adjustments (mirrors abstract_trainer.py)
    if args.backbone == "FNO":
        configs.isFNO = True
    if args.backbone == "TimesNet":
        configs.isTimesNet = True

    print(f"\n=== Pretrain Source Models: {args.backbone} / {args.dataset} ===")
    print(f"    device={device}  data_path={data_path}")

    # ── Phase 1: HP selection ────────────────────────────────────────────────
    hp_cache = _load_hp_cache()
    cache_key = f"{args.backbone}/{args.dataset}"

    if args.skip_hp:
        lr = args.lr if args.lr is not None else 1e-3
        bs = args.bs if args.bs is not None else 64
        print(f"Skipping HP search — using lr={lr:.0e}, bs={bs}")
    elif not args.redo_hp and cache_key in hp_cache:
        lr = hp_cache[cache_key]["lr"]
        bs = hp_cache[cache_key]["bs"]
        print(f"\nPhase 1: Using cached HP for {cache_key}: lr={lr:.0e}, bs={bs}")
        print(f"         (run with --redo_hp to force a new search)")
    else:
        print("\nPhase 1: HP grid search on probe domains …")
        lr, bs = hp_search(args.backbone, args.dataset, device, data_path, configs)
        hp_cache[cache_key] = {"lr": lr, "bs": bs}
        _save_hp_cache(hp_cache)
        print(f"  [HP] Saved to {HP_CACHE_PATH}")

    # ── Phase 2: full training ───────────────────────────────────────────────
    src_ids = SOURCE_DOMAINS.get(args.dataset)
    # src_ids = _get_all_source_ids(args.dataset)  # dynamic alternative — kept for future use

    if not src_ids:
        print(f"\nNo SOURCE_DOMAINS entry for '{args.dataset}'. "
              f"Add it to the SOURCE_DOMAINS dict at the top of this file.")
        sys.exit(1)

    print(f"\nPhase 2: Training {len(src_ids)} source models "
          f"(lr={lr:.0e}, bs={bs}, epochs={SRC_EPOCHS}) …")

    for src_id in src_ids:
        ckpt_path = os.path.join(
            "source_models", "Vanilla", args.backbone, args.dataset, f"src_{src_id}.pt"
        )
        if os.path.exists(ckpt_path) and not args.force:
            if _is_corrupt(ckpt_path):
                print(f"  [CORRUPT] {ckpt_path} — retraining")
            else:
                print(f"  [SKIP]    {ckpt_path}")
                continue

        print(f"  Training src_{src_id} …")
        pretrain_one(args.backbone, args.dataset, src_id,
                     lr, bs, device, data_path, configs, ckpt_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
