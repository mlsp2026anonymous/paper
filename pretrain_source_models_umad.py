#!/usr/bin/env python3
"""
pretrain_source_models_umad.py — Source pretraining for UMAD (two-head co-training).

Mirrors pretrain_source_models.py but uses UMAD's specific training objective:
  L_src = 0.5*(CE_h1 + CE_h2) + lam_orth * ||W1^T W2||_F
with label smoothing on both CE terms.

Checkpoints are saved as:
  source_models/UMAD/{backbone}/{dataset}/src_{id}.pt

The checkpoint format matches what UMAD.load_source_model() expects:
  - keys "0.*" / "1.*"  →  feature extractor / classifier head 1 (via nn.Sequential)
  - keys "classifier2.*" →  classifier head 2

Phase 1: HP grid search on probe domains (test-set accuracy, same as generic script).
Phase 2: Full co-training with early stopping (patience PATIENCE epochs on test acc).

Usage:
    python pretrain_source_models_umad.py --backbone CNN --dataset HAR
    python pretrain_source_models_umad.py --backbone MambaSSM --dataset EEG --skip_hp --lr 1e-3
    python pretrain_source_models_umad.py --backbone CNN --dataset HHAR --force
    python pretrain_source_models_umad.py --backbone CNN --dataset HAR  --redo_hp
"""

import argparse
import copy
import itertools
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.data_model_configs import get_dataset_class
from dataloader.dataloader import data_generator, get_label_encoder
from models.models import get_backbone_class, classifier
from utils import fix_randomness

# ── Domain lists (same as pretrain_source_models.py) ───────────────────────────
SOURCE_DOMAINS = {
    "HAR":  ["1", "6", "9", "12", "13", "15", "17", "22", "24", "30", "2", "7", "18", "20", "28"],
    "HHAR": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
    "EEG":  ["0", "3", "5", "6", "7", "9", "12", "13", "16", "18", "4", "8", "10"],
}

HP_PROBE_DOMAINS = {
    "HAR":  ["2", "7", "18", "20", "28"],
    "HHAR": ["3", "6", "7"],
    "EEG":  ["4", "8", "10"],
}

# ── HP search grid ──────────────────────────────────────────────────────────────
LR_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
BS_GRID = [64]

HP_EPOCHS  = 20
SRC_EPOCHS = 100

# ── Training constants ──────────────────────────────────────────────────────────
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 5.0
PATIENCE     = 5      # early stopping: epochs without test-acc improvement
TEMPERATURE  = 2.0    # logit temperature (> 1 softens predictions)
ALPHA_LS     = 0.1    # label-smoothing coefficient (matches UMAD default)
LAM_ORTH     = 0.1    # orthogonal constraint weight (matches UMAD default)

HP_CACHE_PATH  = os.path.join("configs", "source_hparams_umad.json")
CKPT_SUBDIR    = "UMAD"


# ── HP cache helpers ────────────────────────────────────────────────────────────

def _load_hp_cache():
    if os.path.exists(HP_CACHE_PATH):
        with open(HP_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_hp_cache(cache):
    os.makedirs(os.path.dirname(HP_CACHE_PATH), exist_ok=True)
    with open(HP_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


# ── Network helpers ─────────────────────────────────────────────────────────────

def _build_umad_network(backbone, configs, device):
    """Build fe + clf1 + clf2 as separate modules (not wrapped in Sequential)."""
    backbone_class = get_backbone_class(backbone)
    fe   = backbone_class(configs).to(device)
    clf1 = classifier(configs).to(device)
    clf2 = classifier(configs).to(device)
    return fe, clf1, clf2


def _is_corrupt(ckpt_path):
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return any(
            v.is_floating_point() and torch.isnan(v).any()
            for v in sd.values()
        )
    except Exception:
        return True


# ── Data helpers ────────────────────────────────────────────────────────────────

def _prepare_source(data_path, src_id, configs):
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
    return data_generator(data_path, src_id, cfg, hparams, encoder, pri_cl, dtype,
                          src=(dtype == "train"))


# ── UMAD-specific training ──────────────────────────────────────────────────────

def _umad_loss(fe, clf1, clf2, x, y, num_classes):
    """UMAD co-training loss: 0.5*(CE_h1 + CE_h2) + lam_orth * ||W1^T W2||_F."""
    feat = fe(x)

    q = torch.zeros(y.size(0), num_classes, device=x.device)
    q.scatter_(1, y.view(-1, 1), 1.0)
    q = (1.0 - ALPHA_LS) * q + ALPHA_LS / num_classes

    p1 = F.softmax(clf1(feat) / TEMPERATURE, dim=-1)
    p2 = F.softmax(clf2(feat) / TEMPERATURE, dim=-1)

    ce1 = -(q * torch.log(p1 + 1e-5)).sum(dim=-1).mean()
    ce2 = -(q * torch.log(p2 + 1e-5)).sum(dim=-1).mean()

    W1   = clf1.logits.weight
    W2   = clf2.logits.weight
    orth = torch.norm(W1.t() @ W2, p="fro")

    return 0.5 * (ce1 + ce2) + LAM_ORTH * orth


def _train_epoch(fe, clf1, clf2, loader, optimizer, device, num_classes):
    fe.train(); clf1.train(); clf2.train()
    total_loss, n = 0.0, 0
    all_params = list(fe.parameters()) + list(clf1.parameters()) + list(clf2.parameters())
    for x, y, _ in loader:
        x, y = x.float().to(device), y.long().to(device)
        loss = _umad_loss(fe, clf1, clf2, x, y, num_classes)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def _evaluate(fe, clf1, clf2, loader, device):
    """Test accuracy using the ensemble average of both heads."""
    fe.eval(); clf1.eval(); clf2.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.float().to(device), y.long().to(device)
            feat = fe(x)
            p = 0.5 * (F.softmax(clf1(feat), dim=-1) + F.softmax(clf2(feat), dim=-1))
            correct += (p.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _save_ckpt(fe, clf1, clf2, ckpt_path):
    """Save in the format UMAD.load_source_model() expects."""
    network = nn.Sequential(fe, clf1)
    save_dict = dict(network.state_dict())
    save_dict.update({f"classifier2.{k}": v for k, v in clf2.state_dict().items()})
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(save_dict, ckpt_path)


# ── Phase 1: HP grid search ─────────────────────────────────────────────────────

def hp_search(backbone, dataset, device, data_path, configs):
    probe_ids = HP_PROBE_DOMAINS.get(dataset, [])
    if not probe_ids:
        print(f"  [HP] No probe domains for {dataset}. Using defaults lr=1e-3, bs=64.")
        return 1e-3, 64

    best_lr, best_bs, best_acc = None, None, -1.0

    for lr, bs in itertools.product(LR_GRID, BS_GRID):
        run_accs = []
        for src_id in probe_ids:
            fix_randomness(0)
            try:
                cfg, encoder, pri_cl = _prepare_source(data_path, src_id, configs)
                train_loader = _make_loader(data_path, src_id, cfg, encoder, pri_cl, bs, "train")
                test_loader  = _make_loader(data_path, src_id, cfg, encoder, pri_cl, bs, "test")
            except Exception as e:
                print(f"  [HP] Skipping probe {dataset}/{src_id}: {e}")
                continue

            fe, clf1, clf2 = _build_umad_network(backbone, cfg, device)
            opt = torch.optim.Adam(
                list(fe.parameters()) + list(clf1.parameters()) + list(clf2.parameters()),
                lr=lr, weight_decay=WEIGHT_DECAY,
            )
            for _ in range(HP_EPOCHS):
                _train_epoch(fe, clf1, clf2, train_loader, opt, device, cfg.num_classes)
            acc = _evaluate(fe, clf1, clf2, test_loader, device)
            run_accs.append(acc)
            del fe, clf1, clf2

        if not run_accs:
            continue

        avg_acc = sum(run_accs) / len(run_accs)
        print(f"  [HP] lr={lr:.0e}  bs={bs:3d}  avg_test_acc={avg_acc:.4f}")
        if avg_acc > best_acc:
            best_acc, best_lr, best_bs = avg_acc, lr, bs

    print(f"  [HP] Best: lr={best_lr:.0e}, bs={best_bs}  (test_acc={best_acc:.4f})")
    return best_lr, best_bs


# ── Phase 2: full co-training with early stopping ───────────────────────────────

def pretrain_one(backbone, src_id, lr, bs, device, data_path, configs, ckpt_path):
    """Co-train fe + clf1 + clf2 with early stopping; saves best test-acc checkpoint."""
    fix_randomness(42)
    cfg, encoder, pri_cl = _prepare_source(data_path, src_id, configs)
    train_loader = _make_loader(data_path, src_id, cfg, encoder, pri_cl, bs, "train")
    test_loader  = _make_loader(data_path, src_id, cfg, encoder, pri_cl, bs, "test")

    fe, clf1, clf2 = _build_umad_network(backbone, cfg, device)
    opt = torch.optim.Adam(
        list(fe.parameters()) + list(clf1.parameters()) + list(clf2.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )

    best_acc, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, SRC_EPOCHS + 1):
        loss = _train_epoch(fe, clf1, clf2, train_loader, opt, device, cfg.num_classes)
        acc  = _evaluate(fe, clf1, clf2, test_loader, device)

        if acc > best_acc:
            best_acc   = acc
            best_state = (copy.deepcopy(fe.state_dict()),
                          copy.deepcopy(clf1.state_dict()),
                          copy.deepcopy(clf2.state_dict()))
            no_improve = 0
            print(f"    [{epoch:3d}/{SRC_EPOCHS}] loss={loss:.4f}  test_acc={acc:.4f}  [best]")
        else:
            no_improve += 1
            if epoch % 10 == 0:
                print(f"    [{epoch:3d}/{SRC_EPOCHS}] loss={loss:.4f}  test_acc={acc:.4f}"
                      f"  (no improvement: {no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    fe.load_state_dict(best_state[0])
    clf1.load_state_dict(best_state[1])
    clf2.load_state_dict(best_state[2])

    print(f"  Best test accuracy: {best_acc:.4f}")
    _save_ckpt(fe, clf1, clf2, ckpt_path)
    print(f"  Saved → {ckpt_path}")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pretrain UMAD two-head source models")
    parser.add_argument("--backbone",  required=True)
    parser.add_argument("--dataset",   required=True)
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--skip_hp",   action="store_true")
    parser.add_argument("--redo_hp",   action="store_true")
    parser.add_argument("--lr",        type=float, default=None)
    parser.add_argument("--bs",        type=int,   default=None)
    parser.add_argument("--force",     action="store_true",
                        help="Retrain all domains even if a valid checkpoint exists")
    args = parser.parse_args()

    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(args.data_path, args.dataset)
    configs   = get_dataset_class(args.dataset)()

    if args.backbone == "FNO":
        configs.isFNO = True
    if args.backbone == "TimesNet":
        configs.isTimesNet = True

    print(f"\n=== UMAD Source Pretraining: {args.backbone} / {args.dataset} ===")
    print(f"    device={device}  data_path={data_path}")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    hp_cache  = _load_hp_cache()
    cache_key = f"{args.backbone}/{args.dataset}"

    if args.skip_hp:
        lr = args.lr if args.lr is not None else 1e-3
        bs = args.bs if args.bs is not None else 64
        print(f"Skipping HP search — using lr={lr:.0e}, bs={bs}")
    elif not args.redo_hp and cache_key in hp_cache:
        lr = hp_cache[cache_key]["lr"]
        bs = hp_cache[cache_key]["bs"]
        print(f"\nPhase 1: Cached HP for {cache_key}: lr={lr:.0e}, bs={bs}"
              f"  (--redo_hp to re-run)")
    else:
        print("\nPhase 1: HP grid search on probe domains …")
        lr, bs = hp_search(args.backbone, args.dataset, device, data_path, configs)
        hp_cache[cache_key] = {"lr": lr, "bs": bs}
        _save_hp_cache(hp_cache)
        print(f"  [HP] Saved to {HP_CACHE_PATH}")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    src_ids = SOURCE_DOMAINS.get(args.dataset)
    if not src_ids:
        print(f"\nNo SOURCE_DOMAINS entry for '{args.dataset}'.")
        sys.exit(1)

    print(f"\nPhase 2: Co-training {len(src_ids)} UMAD source models "
          f"(lr={lr:.0e}, bs={bs}, max_epochs={SRC_EPOCHS}, patience={PATIENCE}) …")

    for src_id in src_ids:
        ckpt_path = os.path.join(
            "source_models", CKPT_SUBDIR, args.backbone, args.dataset, f"src_{src_id}.pt"
        )
        if os.path.exists(ckpt_path) and not args.force:
            if _is_corrupt(ckpt_path):
                print(f"  [CORRUPT] {ckpt_path} — retraining")
            else:
                print(f"  [SKIP]    {ckpt_path}")
                continue

        print(f"  Co-training src_{src_id} …")
        pretrain_one(args.backbone, src_id, lr, bs, device, data_path, configs, ckpt_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
