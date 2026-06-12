"""
Source-Free Universal Domain Adaptation algorithms.

Currently implements
--------------------
  GLC   — Global-Local Clustering (CVPR 2023)
          arxiv.org/abs/2303.07110
  GLCpp — GLC++ with contrastive affinity learning (TPAMI 2025)
          ieeexplore.ieee.org/document/11123595
  LEAD  — Learning feature decomposition for SF-UniDA (arxiv 2403.03421)
          github.com/ispc-lab/LEAD
  UMAD —  Universal Model ADaptation under Domain and Category Shift. 
          arxiv.org/abs/2112.08553

How to add a new SF-UniDA method
---------------------------------
1. Subclass SFUniDAAlgorithm (or GLC for GLC-family variants).
2. Add the class name to SF_METHODS below.
3. Add its default hparams to configs/hparams_sfunida.py.

The trainer automatically detects SF methods via SF_METHODS and:
  - routes to get_sf_algorithm_class() instead of get_algorithm_class()
  - trains / loads a source checkpoint before calling update()
  - calls update(None, trg_loader, ...) — src_loader is always None here
"""

import math

import numpy as np
import skimage.filters as sfil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from algorithms.algorithms import Algorithm, classifier   # reuse base class + head

# ── Registry ──────────────────────────────────────────────────────────────────
SF_METHODS = {"GLC", "GLCpp", "LEAD", "UMAD"}


def get_sf_algorithm_class(name):
    if name not in globals():
        raise NotImplementedError(f"SF-UniDA algorithm not found: {name}")
    return globals()[name]


# ── Base class ────────────────────────────────────────────────────────────────

class SFUniDAAlgorithm(Algorithm):
    """
    Base for all SF-UniDA algorithms.

    Contract with the trainer
    -------------------------
    - load_source_model(state_dict) is called with a source-pretrained
      network.state_dict() before update() is invoked.
    - update(None, trg_loader, avg_meter, logger) — src_loader is always None.
    - decision_function() thresholds max-softmax confidence to flag unknowns.
    """

    IS_SF_UNIDA = True          # sentinel read by the trainer

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)
        self.hparams = hparams
        self.device  = device
        self._feat_dim = None
        self.auto_threshold  = bool(hparams.get("auto_threshold", False))
        print(f"-----------Use Auto-Thresholding : {self.auto_threshold}-----------")
        if "threshold_method" in hparams:
            print(f"-----------Method Auto-Thresholding : {hparams['threshold_method']}-----------")
        self.final_threshold = None   # set by update() when auto_threshold=True

    # ------------------------------------------------------------------
    def load_source_model(self, state_dict):
        """Load weights from a source-pretrained checkpoint."""
        self.network.load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    @property
    def feat_dim(self):
        """Input dimension of the linear classifier (= feature extractor output dim)."""
        if self._feat_dim is None:
            self._feat_dim = self.classifier.logits.in_features
        return self._feat_dim

    # ------------------------------------------------------------------
    def decision_function(self, preds):
        """
        Threshold max-softmax confidence to detect unknown target samples.
        Returns class index, or -1 where confidence < threshold.
        Uses final_threshold (Yen auto-selected) when available, else w_0.
        """
        confidence, pred = F.softmax(preds, dim=1).max(dim=1)
        pred = pred.clone()
        threshold = (self.final_threshold if self.final_threshold is not None
                     else self.hparams.get("w_0", 0.5))
        pred[confidence < threshold] = -1
        return pred

    def _apply_auto_threshold(self, trg_loader, logger):
        """Compute and store an auto-selected threshold on the full target confidence distribution."""
        _THR_METHODS = {
            "yen":      sfil.threshold_yen,
            "otsu":     sfil.threshold_otsu,
            "li":       sfil.threshold_li,
            "triangle": sfil.threshold_triangle,
        }
        method_name = str(self.hparams.get("threshold_method", "yen")).lower()
        thr_fn = _THR_METHODS.get(method_name, sfil.threshold_yen)
        _, pred_bank = self._build_banks(trg_loader.dataset, trg_loader.batch_size)
        conf = pred_bank.max(dim=-1).values.numpy()
        self.final_threshold = float(thr_fn(conf))
        logger.debug(f"[{self.__class__.__name__}] auto_threshold ({method_name}): {self.final_threshold:.4f}")

    # ------------------------------------------------------------------
    def _build_banks(self, indexed_dataset, batch_size):
        """
        Run inference over the full dataset and return ordered banks.

        Returns
        -------
        feat_bank : [N, D] L2-normalised features (CPU)
        pred_bank : [N, C] softmax predictions   (CPU)

        Banks are indexed by dataset index, not by batch order.
        """
        N = len(indexed_dataset)
        D = self.feat_dim
        C = self.configs.num_classes

        feat_bank = torch.zeros(N, D)
        pred_bank = torch.zeros(N, C)

        loader = DataLoader(indexed_dataset, batch_size=batch_size * 2,
                            shuffle=False, drop_last=False, num_workers=0)

        self.feature_extractor.eval()
        self.classifier.eval()

        with torch.no_grad():
            for x, _, idx in loader:
                x = x.float().to(self.device)
                f = self.feature_extractor(x)
                p = F.softmax(self.classifier(f), dim=-1)
                feat_bank[idx] = F.normalize(f, p=2, dim=-1).cpu()
                pred_bank[idx] = p.cpu()

        return feat_bank, pred_bank

    # ------------------------------------------------------------------
    def compute_src_loss(self, feat, src_y):
        """
        Loss for source-model training. Default: standard cross-entropy on h1.
        Override for algorithms with custom source training (e.g. UMAD).
        """
        return F.cross_entropy(self.classifier(feat), src_y)

    def get_src_optimizer(self, lr, weight_decay):
        """
        Optimizer for source-model training. Default: Adam over fe + classifier.
        Override to include extra parameters (e.g. UMAD's classifier2).
        """
        return torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.classifier.parameters()),
            lr=lr, weight_decay=weight_decay,
        )

    def extra_src_state_dict(self):
        """
        Extra state to merge into the source checkpoint beyond self.network.
        Returns a flat dict of {prefixed_key: tensor}. Default: empty.
        Override to persist extra parameters (e.g. UMAD's classifier2).
        """
        return {}

    # ------------------------------------------------------------------
    def update(self, src_loader, trg_loader, avg_meter, logger):
        raise NotImplementedError


# ── GLC ───────────────────────────────────────────────────────────────────────

class GLC(SFUniDAAlgorithm):
    """
    Global-Local Clustering for SF-UniDA  (CVPR 2023).
    arxiv.org/abs/2303.07110

    Per-epoch pipeline
    ------------------
    1. Global step  — one-vs-all pseudo-labeling using per-class positive
       prototypes vs. K-means-clustered negative centroids in feature space.
       The number of clusters K = class_num × coeff is selected once (epoch 0)
       via Silhouette score on a t-SNE 2-D projection.
    2. Local step   — KNN-consistency loss: each sample's prediction is pulled
       towards the mean prediction of its K nearest neighbours in the bank.

    GLC loss = λ_psd × CE(global_psd) + λ_knn × CE(KNN_pred)
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(backbone, configs, hparams, device)

        # Classifier stays frozen; only the feature extractor is updated (matches official impl).
        for p in self.classifier.parameters():
            p.requires_grad = False
        trainable = [p for p in self.feature_extractor.parameters() if p.requires_grad]

        self.optimizer = torch.optim.SGD(
            trainable,
            lr=hparams["learning_rate"],
            momentum=0.9,
            weight_decay=hparams["weight_decay"],
            nesterov=True,
        )

        self.local_K = hparams.get("local_K",  4)
        self.rho     = hparams.get("rho",      0.3)
        self.lam_psd = hparams.get("lam_psd",  1.0)
        self.lam_knn = hparams.get("lam_knn",  1.0)

    # ── Global pseudo-label generation ────────────────────────────────

    @torch.no_grad()
    def _obtain_global_pseudo_labels(self, feat_bank, pred_bank, epoch, best_state):
        """
        One-vs-all global clustering on CPU.

        Parameters
        ----------
        feat_bank  : [N, D]  L2-normalised, CPU
        pred_bank  : [N, C]  softmax,        CPU
        epoch      : 0-based index  (K-selection runs only at epoch 0)
        best_state : dict with key 'coeff' (mutated at epoch 0)

        Returns
        -------
        hard_psd_bank : [N, C] soft pseudo-labels (CPU)
                        Rows sum to 1. Unknown samples get a uniform
                        distribution over all classes.
        """
        N, C = pred_bank.shape
        D    = feat_bank.shape[1]

        # ── K selection via Silhouette on t-SNE (once per scenario) ──
        if epoch == 0:
            feat_np = feat_bank.numpy()
            if N > 10_000:                        # subsample for speed
                rng = np.random.default_rng(0)
                feat_np = feat_np[rng.choice(N, N // 3, replace=False)]

            from sklearn.manifold import TSNE
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            feat_2d = TSNE(n_components=2, init="pca", random_state=0).fit_transform(feat_np)

            best_score, best_coeff = -1.0, 1.0
            for coeff in [0.25, 0.5, 1.0, 2.0, 3.0]:
                KK     = max(int(C * coeff), 2)
                labels = KMeans(n_clusters=KK, random_state=0, n_init=10).fit_predict(feat_2d)
                score  = silhouette_score(feat_2d, labels)
                if score > best_score:
                    best_score, best_coeff = score, coeff

            best_state["coeff"] = best_coeff

        coeff    = best_state["coeff"]
        KK       = max(int(C * coeff), 2)
        pos_topk = max(int(N / C / coeff), 1)

        # ── Positive / negative split per class ──────────────────────
        sorted_pred, sorted_idx = torch.sort(pred_bank, dim=0, descending=True)
        # sorted_idx: [N, C]  — sample indices sorted by class confidence
        pos_idx = sorted_idx[:pos_topk, :].t()    # [C, pos_topk]
        neg_idx = sorted_idx[pos_topk:,  :].t()   # [C, neg_topk]

        # per-class prior weight: how certain we are about positive assignment
        pos_prior = (sorted_pred[:pos_topk, :].mean(dim=0, keepdim=True).t()
                     * (1.0 - self.rho) + self.rho)               # [C, 1]

        # positive prototype per class  [C, 1, D]
        pos_feats = feat_bank[pos_idx]             # [C, pos_topk, D]
        pos_proto = F.normalize(pos_feats.mean(dim=1, keepdim=True), p=2, dim=-1)

        # negative features per class   [C, neg_topk, D]
        neg_feats = feat_bank[neg_idx]

        # ── One-vs-all: cluster negatives, compute similarity ─────────
        feat_proto_max_idxs = torch.zeros(N, C)

        _cluster_fn = self._get_cluster_fn(D, KK)

        for c in range(C):
            neg_np       = neg_feats[c].numpy().astype(np.float32)
            KK_c         = min(KK, neg_np.shape[0])   # guard against tiny splits #TODO: Keep this safe guard or remove to be closer to official implementation
            centroids    = torch.from_numpy(_cluster_fn(neg_np, KK_c)).float()
            neg_centroids = F.normalize(centroids, p=2, dim=-1)   # [KK_c, D]

            cls_pos  = pos_proto[c]                                # [1, D]
            pos_simi = (feat_bank @ cls_pos.t()) * pos_prior[c]   # [N, 1]
            neg_simi = feat_bank @ neg_centroids.t()               # [N, KK_c]

            all_simi = torch.cat([pos_simi, neg_simi], dim=1)     # [N, 1+KK_c]
            feat_proto_max_idxs[:, c] = all_simi.argmax(dim=-1).float()

        # ── Known / unknown decision ──────────────────────────────────
        # A sample is "known as class c" if pos prototype wins AND it is the
        # closest prototype to the sample (prior-based tiebreak).
        psd_prior_simi = feat_bank @ pos_proto.squeeze(1).t()       # [N, C]
        psd_prior_idx  = psd_prior_simi.argmax(dim=-1, keepdim=True)
        psd_prior      = torch.zeros_like(psd_prior_simi).scatter(1, psd_prior_idx, 1.0)

        hard_psd = (feat_proto_max_idxs == 0).float() * psd_prior   # [N, C]
        is_unk   = hard_psd.sum(dim=-1) == 0
        hard_psd[is_unk] += 1.0                                      # uniform for unknowns
        hard_psd = hard_psd / (hard_psd.sum(dim=-1, keepdim=True) + 1e-4)

        return hard_psd   # [N, C]

    @staticmethod
    def _get_cluster_fn(D, KK):
        """Return a clustering function (D, KK) → centroids array."""
        try:
            import faiss

            def _faiss(data, k):
                km = faiss.Kmeans(D, k, niter=100, verbose=False,
                                  min_points_per_centroid=1, gpu=False)
                km.train(data)
                return km.centroids

            return _faiss
        except ImportError:
            from sklearn.cluster import KMeans

            def _sklearn(data, k):
                return KMeans(n_clusters=k, n_init=10, random_state=0).fit(data).cluster_centers_

            return _sklearn

    # ── Mini-batch training ───────────────────────────────────────────

    def _training_epoch(self, indexed_loader, hard_psd_bank, pred_bank,
                        feat_bank, avg_meter, best_state):
        self.feature_extractor.train()
        self.classifier.train()

        for x, _, idx in indexed_loader:
            x   = x.float().to(self.device)
            idx = idx.long()                      # CPU indices into the banks

            hard_psd = hard_psd_bank[idx].to(self.device)   # [B, C]

            # ── Forward ──────────────────────────────────────────────
            f    = self.feature_extractor(x)
            pred = F.softmax(self.classifier(f), dim=-1)     # [B, C]
            f_n  = F.normalize(f, p=2, dim=-1)               # [B, D]
            f_n_cpu = f_n.detach().cpu()

            # ── KNN lookup in the (CPU) feature bank ─────────────────
            with torch.no_grad():
                simi = f_n_cpu @ feat_bank.t()               # [B, N]
                # exclude self-similarity
                simi[torch.arange(len(idx)), idx] = -2.0
                nn_idx  = simi.topk(self.local_K, dim=-1, largest=True).indices  # [B, K]
                nn_pred = pred_bank[nn_idx].mean(dim=1).to(self.device)          # [B, C]

                # live-update banks so KNN stays current within the epoch
                pred_bank[idx] = pred.detach().cpu()
                feat_bank[idx] = f_n_cpu

            # ── Losses ───────────────────────────────────────────────
            psd_loss = -(hard_psd * torch.log(pred + 1e-5)).sum(dim=-1).mean()
            knn_loss = -(nn_pred  * torch.log(pred + 1e-5)).sum(dim=-1).mean()
            extra    = self._extra_loss(f_n, nn_idx, feat_bank, best_state)

            loss = self.lam_psd * psd_loss + self.lam_knn * knn_loss + extra

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_meter["Total_loss"].update(loss.item(), x.size(0))
            avg_meter["Psd_loss"].update(psd_loss.item(),  x.size(0))
            avg_meter["KNN_loss"].update(knn_loss.item(),  x.size(0))

    def _extra_loss(self, f_n, nn_idx, feat_bank, best_state):
        """Hook for subclasses. GLC returns 0; GLCpp overrides this."""
        return 0.0

    # ── Main update loop ──────────────────────────────────────────────

    def update(self, src_loader, trg_loader, avg_meter, logger):
        train_loader = DataLoader(
            trg_loader.dataset,
            batch_size=trg_loader.batch_size,
            shuffle=True, drop_last=True, num_workers=0,
        )

        best_state = {"coeff": 1.0}   # mutated by _obtain_global_pseudo_labels at epoch 0

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # Rebuild banks from current model state
            feat_bank, pred_bank = self._build_banks(trg_loader.dataset, trg_loader.batch_size)

            # Global pseudo-labels (slow at epoch 0 due to t-SNE; fast afterwards)
            hard_psd_bank = self._obtain_global_pseudo_labels(
                feat_bank, pred_bank, epoch - 1, best_state)

            # One epoch of target adaptation
            self._training_epoch(
                train_loader, hard_psd_bank, pred_bank, feat_bank, avg_meter, best_state)

            logger.debug(f"[Epoch {epoch}/{self.hparams['num_epochs']}]")
            for key, val in avg_meter.items():
                logger.debug(f"  {key}: {val.avg:.4f}")
            logger.debug("-------------------------------------")

        if self.auto_threshold:
            self._apply_auto_threshold(trg_loader, logger)

        return self.network.state_dict(), None


# ── GLC++ ─────────────────────────────────────────────────────────────────────

class GLCpp(GLC):
    """
    GLC++ — GLC with contrastive affinity learning  (TPAMI 2025).
    ieeexplore.ieee.org/document/11123595

    Adds a regularisation loss that encourages each sample's feature to be
    closer to its K nearest neighbours (positive pairs) than to hard negatives
    sampled from the current mini-batch.

    GLC++ loss = λ_psd × CE(global_psd)
               + λ_knn × CE(KNN_pred)
               + λ_reg × (Σ neg_sim[start:end] − Σ pos_sim)
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(backbone, configs, hparams, device)
        self.lam_reg = hparams.get("lam_reg", 1.0)

    def _extra_loss(self, f_n, nn_idx, feat_bank, best_state):
        """
        Contrastive affinity regularisation.

        f_n       : [B, D]  L2-normed features on device
        nn_idx    : [B, K]  CPU indices of K nearest neighbours
        feat_bank : [N, D]  CPU feature bank
        best_state: dict with 'coeff'
        """
        B     = f_n.shape[0]
        C     = self.configs.num_classes
        coeff = best_state.get("coeff", 1.0)

        # Positive pairs: similarity to K nearest neighbours
        nn_feats = feat_bank[nn_idx].to(self.device)              # [B, K, D]
        pos_simi = torch.einsum("bd,bkd->bk", f_n, nn_feats)     # [B, K]

        # Hard negatives: all-vs-all in the mini-batch, sorted descending
        neg_simi   = f_n @ f_n.detach().t()                       # [B, B]
        neg_sorted = neg_simi.sort(dim=-1, descending=True).values[:, 1:]  # skip self

        # Skip the very top entries (likely same-class) and use the next K
        start = max(int(np.ceil(B / C / coeff)), 1)
        end   = min(start + self.local_K, neg_sorted.shape[1])

        reg_loss = (neg_sorted[:, start:end].sum(dim=-1)
                    - pos_simi.sum(dim=-1)).mean()
        return self.lam_reg * reg_loss


# ── LEAD ──────────────────────────────────────────────────────────────────────

class LEAD(SFUniDAAlgorithm):
    """
    LEAD — Learning feature decomposition for SF-UniDA (arxiv 2403.03421).
    github.com/ispc-lab/LEAD

    Core idea
    ---------
    SVD of the source classifier weight matrix W decomposes the feature space
    into a "known" subspace (first C left singular vectors) and an "unknown"
    subspace (the D-C dimensional complement).  The unknown-projection norm
    serves as an open-set score.

    Pseudo-label pipeline (per epoch; Ct re-estimated every 10 epochs via t-SNE)
    ----------------------
    1. Fit GMM(k=2) to unknown_norm → μ₂ = high-norm component mean.
    2. S_common(n,c) = sqrt((1-exp(-tar_simi)) * exp(src_simi - 1))
       with both similarities clamped to [0, +∞).
    3. threshold(n,c) = per_cls_prior[c] + S_common * clamp(μ₂ - prior, 0)
    4. psd_label[n] = argmax_c(S_common[n]), then mark unknown if
       unknown_norm[n] >= threshold[n, best_cls].
    5. Confidence weight: 1 - (1 + (unknown_norm - threshold)²/α)^(-(α+1)/2)
       overridden to 1.0 for definite unknowns (≥ μ₂) and definite knowns (< prior).

    Losses
    ------
    L_ce  : Weighted soft-label CE — known→one-hot, unknown→uniform 1/C
    L_reg : Weighted binary CE on softmax([known_norm, unknown_norm])
    L_con : KNN consistency CE
    total = λ_psd × L_ce + L_reg + L_con

    Note: only the feature extractor is optimised; classifier stays frozen.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(backbone, configs, hparams, device)

        # Only the feature extractor is trained; classifier remains frozen.
        trainable = [p for p in self.feature_extractor.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            trainable,
            lr=hparams["learning_rate"],
            momentum=0.9,
            weight_decay=hparams["weight_decay"],
            nesterov=True,
        )

        self.local_K = hparams.get("local_K", 4)
        self.lam_psd = hparams.get("lam_psd", 0.3)
        self.alpha   = hparams.get("alpha",   1e-4)

        self.known_basis   = None   # [C, D]   — set once in update()
        self.unknown_basis = None   # [D-C, D] — set once in update()

    # ── SVD decomposition ─────────────────────────────────────────────

    @torch.no_grad()
    def _compute_svd_bases(self):
        """Decompose feature space using SVD of the fixed classifier weights."""
        W = self.classifier.logits.weight.data   # [C, D], on device
        C = W.shape[0]
        U, _, _ = torch.linalg.svd(W.t(), full_matrices=True)   # U: [D, D]
        self.known_basis   = F.normalize(U[:, :C].t(), p=2, dim=-1)   # [C, D]
        self.unknown_basis = F.normalize(U[:, C:].t(), p=2, dim=-1)   # [D-C, D]

    # ── Pseudo-label generation ───────────────────────────────────────

    @torch.no_grad()
    def _obtain_LEAD_pseudo_labels(self, feat_bank, pred_bank, epoch, best_state):
        """
        Returns (all CPU tensors)
        -------
        psd_onehot  : [N, C] float — soft one-hot (known) or uniform 1/C (unknown)
        psd_unk_flg : [N]    bool  — True for unknown samples
        psd_weight  : [N]    float — confidence weight per sample
        """
        from sklearn.mixture import GaussianMixture

        N, C = pred_bank.shape
        feat_gpu = feat_bank.to(self.device)   # [N, D]

        # ── Ct re-estimation via t-SNE + Silhouette (every 10 epochs) ────
        if epoch % 10 == 0:
            from sklearn.manifold import TSNE
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            feat_np = feat_bank.numpy()
            if N > 10_000:
                rng = np.random.default_rng(0)
                feat_np = feat_np[rng.choice(N, N // 3, replace=False)]

            feat_2d = TSNE(n_components=2, init="pca", random_state=0).fit_transform(feat_np)

            best_sil, best_coeff = -1.0, best_state.get("coeff", 1.0)
            for coeff in [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]:
                KK     = max(int(C * coeff), 2)
                labels = KMeans(n_clusters=KK, random_state=0, n_init=10).fit_predict(feat_2d)
                score  = silhouette_score(feat_2d, labels)
                if score > best_sil:
                    best_sil, best_coeff = score, coeff
            best_state["coeff"] = best_coeff

        coeff        = best_state.get("coeff", 1.0)
        Ct           = max(int(C * coeff), 2)
        pos_topk_num = max(int(N / Ct), 1)

        # ── Project onto known/unknown bases ──────────────────────────
        unk_proj     = feat_gpu @ self.unknown_basis.t()   # [N, D-C]
        unknown_norm = unk_proj.norm(dim=-1)               # [N]

        # ── GMM on unknown_norm ────────────────────────────────────────
        unk_np = unknown_norm.cpu().numpy().reshape(-1, 1)
        gmm    = GaussianMixture(n_components=2, random_state=0, max_iter=200)
        gmm.fit(unk_np)
        mu_vals      = torch.tensor(gmm.means_).squeeze().to(self.device)
        gaussian_mu2 = mu_vals.max()

        # ── Prototypes ────────────────────────────────────────────────
        _, top_idx = pred_bank.topk(pos_topk_num, dim=0)   # [pos_topk_num, C] — CPU
        tar_proto  = F.normalize(
            torch.stack([feat_bank[top_idx[:, c]].mean(0) for c in range(C)]),
            p=2, dim=-1,
        ).to(self.device)   # [C, D]
        src_proto  = F.normalize(self.classifier.logits.weight.data, p=2, dim=-1)  # [C, D]

        # Both similarities clamped to [0, +∞) as in the original
        tar_simi = (feat_gpu @ tar_proto.t()).clamp(min=0.0)   # [N, C]
        src_simi = (feat_gpu @ src_proto.t()).clamp(min=0.0)   # [N, C]

        # ── S_common ──────────────────────────────────────────────────
        s_common = torch.sqrt(
            (1.0 - torch.exp(-tar_simi)) * torch.exp(src_simi - 1.0)
        )   # [N, C]

        # ── Per-class unknown_norm prior ───────────────────────────────
        per_cls_prior = torch.stack([
            unknown_norm[top_idx[:, c].to(self.device)].mean()
            for c in range(C)
        ])   # [C]

        # ── Per-sample per-class threshold ─────────────────────────────
        delta     = (gaussian_mu2 - per_cls_prior).clamp(min=0)   # [C]
        threshold = per_cls_prior + s_common * delta               # [N, C]

        # ── Pseudo-label: argmax(S_common), then threshold on best class ─
        psd_label    = s_common.argmax(dim=-1)   # [N] — initial class by S_common
        psd_label_oh = psd_label.clone()         # preserved for one-hot construction
        psd_weight   = torch.ones(N, device=self.device)

        alpha = self.alpha
        for i in range(C):
            label_idxs = torch.where(psd_label == i)[0]
            if len(label_idxs) == 0:
                continue

            unk_norm_i = unknown_norm[label_idxs]
            thresh_i   = threshold[label_idxs, i]
            prior_i    = per_cls_prior[i]

            # Samples above their class threshold → unknown (encoded as class C)
            psd_label[label_idxs] = torch.where(
                unk_norm_i >= thresh_i,
                torch.full_like(psd_label[label_idxs], C),
                psd_label[label_idxs],
            )

            # w = 1 - (1 + (unknown_norm - threshold)² / α)^(-(α+1)/2)
            # → low weight near the boundary, high weight far from it
            d_sq = (unk_norm_i - thresh_i) ** 2
            psd_weight[label_idxs] = (
                1.0 - (1.0 + d_sq / alpha) ** (-(alpha + 1.0) / 2.0)
            )

            # Definite unknowns (≥ μ₂) and definite knowns (< prior) → weight = 1
            psd_weight[label_idxs[unk_norm_i >= gaussian_mu2]] = 1.0
            psd_weight[label_idxs[unk_norm_i <  prior_i]]      = 1.0

        psd_unk_flg = (psd_label == C)   # [N]

        # Soft one-hot: known → one-hot, unknown → uniform 1/C
        psd_onehot = torch.zeros(N, C, device=self.device).scatter_(
            1, psd_label_oh.unsqueeze(1), 1.0
        )
        psd_onehot[psd_unk_flg] = 1.0
        psd_onehot = psd_onehot / (psd_onehot.sum(dim=-1, keepdim=True) + 1e-5)

        return psd_onehot.cpu(), psd_unk_flg.cpu(), psd_weight.cpu()

    # ── Mini-batch training ───────────────────────────────────────────

    def _training_epoch(self, indexed_loader, psd_onehot_bank, psd_unk_bank,
                        psd_weight_bank, feat_bank, pred_bank, avg_meter, iter_state):
        self.feature_extractor.train()
        self.classifier.eval()   # classifier stays frozen throughout

        T_total = iter_state["T_total"]

        for x, _, idx in indexed_loader:
            x   = x.float().to(self.device)
            idx = idx.long()

            # Per-iteration LR schedule: (1 + 10·t/T)^(-0.75)
            t        = iter_state["t"]
            lr_scale = (1.0 + 10.0 * t / max(T_total, 1)) ** (-0.75)
            for pg in self.optimizer.param_groups:
                pg["lr"] = iter_state["lr0"][id(pg)] * lr_scale

            # Forward pass
            f    = self.feature_extractor(x)
            f_n  = F.normalize(f, p=2, dim=-1)               # [B, D]
            pred = F.softmax(self.classifier(f), dim=-1)     # [B, C]
            f_n_cpu = f_n.detach().cpu()

            psd_onehot = psd_onehot_bank[idx].to(self.device)              # [B, C]
            psd_unk    = psd_unk_bank[idx].to(self.device)                 # [B] bool
            weight     = psd_weight_bank[idx].unsqueeze(1).to(self.device) # [B, 1]

            # L_ce: weighted soft-label CE on all samples
            # known → one-hot target, unknown → uniform 1/C target
            L_ce = (-(psd_onehot * weight * torch.log(pred + 1e-5)).sum(dim=-1)).mean()

            # L_reg: weighted binary CE on softmax([known_norm, unknown_norm])
            kn_norm  = (f_n @ self.known_basis.t()  ).norm(dim=-1, keepdim=True)  # [B, 1]
            unk_norm = (f_n @ self.unknown_basis.t()).norm(dim=-1, keepdim=True)   # [B, 1]
            reg_prob    = F.softmax(torch.cat([kn_norm, unk_norm], dim=-1), dim=-1)  # [B, 2]
            # One-hot target: [1,0] for known (class 0), [0,1] for unknown (class 1)
            reg_target  = torch.zeros_like(reg_prob).scatter_(
                1, psd_unk.long().unsqueeze(1), 1.0
            )
            L_reg = (-(reg_target * weight * torch.log(reg_prob + 1e-5)).sum(dim=-1)).mean()

            # L_con: KNN consistency
            with torch.no_grad():
                simi = f_n_cpu @ feat_bank.t()               # [B, N]
                simi[torch.arange(len(idx)), idx] = -2.0
                nn_idx  = simi.topk(self.local_K, dim=-1, largest=True).indices
                nn_pred = pred_bank[nn_idx].mean(dim=1).to(self.device)

                pred_bank[idx] = pred.detach().cpu()
                feat_bank[idx] = f_n_cpu

            L_con = -(nn_pred * torch.log(pred + 1e-5)).sum(dim=-1).mean()

            loss = self.lam_psd * L_ce + L_reg + L_con

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_meter["Total_loss"].update(loss.item(), x.size(0))
            avg_meter["CE_loss"].update(L_ce.item(),   x.size(0))
            avg_meter["Reg_loss"].update(L_reg.item(), x.size(0))
            avg_meter["Con_loss"].update(L_con.item(), x.size(0))

            iter_state["t"] += 1

    # ── Main update loop ──────────────────────────────────────────────

    def update(self, src_loader, trg_loader, avg_meter, logger):
        self._compute_svd_bases()

        train_loader = DataLoader(
            trg_loader.dataset,
            batch_size=trg_loader.batch_size,
            shuffle=True, drop_last=True, num_workers=0,
        )

        n_epochs  = self.hparams["num_epochs"]
        n_batches = max(len(trg_loader.dataset) // trg_loader.batch_size, 1)

        iter_state = {
            "t":       0,
            "T_total": n_epochs * n_batches,
            "lr0":     {id(pg): pg["lr"] for pg in self.optimizer.param_groups},
        }
        best_state = {"coeff": 1.0}

        for epoch in range(1, n_epochs + 1):
            feat_bank, pred_bank = self._build_banks(trg_loader.dataset, trg_loader.batch_size)

            psd_onehot, psd_unk, psd_weight = self._obtain_LEAD_pseudo_labels(
                feat_bank, pred_bank, epoch - 1, best_state)

            self._training_epoch(
                train_loader, psd_onehot, psd_unk, psd_weight,
                feat_bank, pred_bank, avg_meter, iter_state)

            logger.debug(f"[Epoch {epoch}/{n_epochs}]")
            for key, val in avg_meter.items():
                logger.debug(f"  {key}: {val.avg:.4f}")
            logger.debug("-------------------------------------")

        if self.auto_threshold:
            self._apply_auto_threshold(trg_loader, logger)

        return self.network.state_dict(), None




# ── UMAD ──────────────────────────────────────────────────────────────────────

class UMAD(SFUniDAAlgorithm):
    """
    UMAD — Universal Model ADaptation under Domain and Category Shift.
    arxiv.org/abs/2112.08553

    Source checkpoints are expected at:
        source_models/UMAD/{backbone}/{dataset}/src_{id}.pt
    generated by pretrain_source_models_umad.py.

    Source model (two-head classifier)
    -----------------------------------
    Two linear heads h1=self.classifier and h2=self.classifier2 share one
    feature extractor and are trained jointly with:
        Lsrc = 0.5*(CE_h1 + CE_h2) + lam_orth * ||W1^T W2||_F
    Label smoothing alpha=0.1 is applied to the CE terms.
    The orthogonal constraint encourages the two heads to learn different
    features, which is key for a discriminative consistency score.

    Target adaptation
    -----------------
    Both classifier heads are frozen; only the feature extractor is updated.

    Informative consistency score:
        wt = <p1(xt), p2(xt)>          (inner product of softmax outputs)

    Threshold w0 estimated from Mixup of target samples:
        w0 = E[iscore(0.5*xi + 0.5*xj)]
        rho = rho_frac * w0   (default rho_frac=0.1)

    Bilateral objective:
        Ltgt = Lunk − Llmi      (minimised)

    For unknown samples Xt- = {xt : wt < w0 − rho}:
        Lunk = −E_v [ E_{Xt-}[ Σ_k (1/K) log p_{v,k}(xt) ] ]
        (uniform CE pushes predictions toward a flat distribution)

    For known samples Xt+ = {xt : wt > w0 + rho}:
        Llmi = E_v [ E_{Xt+}[p_v log p_v] − DKL(Q_v || Flatten(Q_v, T)) ]
        where Q_v = E_{Xt+}[p_v],  Flatten(p, T)_i = p_i^T / Σ_j p_j^T
        (simultaneous entropy minimisation + batch-level diversity)
    """

    SRC_SUBDIR = "UMAD"   # trainer inserts this between "source_models" and backbone

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(backbone, configs, hparams, device)

        from models.models import classifier as _Classifier
        self.classifier2 = _Classifier(configs).to(device)

        self.lam_orth = hparams.get("lam_orth", 0.1)
        self.T_flat   = hparams.get("T_flat",   0.1)
        self.rho_frac = hparams.get("rho_frac", 0.1)
        self.alpha_ls = hparams.get("alpha_ls", 0.1)

        # Set to True to fall back to the base-class max-softmax decision_function.
        # When False (default), inference uses the paper's iscore with the w0 computed
        # during update() — which is the correct behaviour per the UMAD paper.
        self._use_maxsoftmax_inference = False

        self._w0_inference = None   # set by update(); used by decision_function

        trainable = [p for p in self.feature_extractor.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            trainable,
            lr=hparams["learning_rate"],
            momentum=0.9,
            weight_decay=hparams["weight_decay"],
            nesterov=True,
        )

    # ── Source training hooks ─────────────────────────────────────────────────

    def compute_src_loss(self, feat, src_y):
        """Co-training loss with label smoothing and orthogonal constraint."""
        K     = self.configs.num_classes
        alpha = self.alpha_ls

        q = torch.zeros(src_y.size(0), K, device=feat.device)
        q.scatter_(1, src_y.view(-1, 1), 1.0)
        q = (1.0 - alpha) * q + alpha / K

        p1 = F.softmax(self.classifier(feat),  dim=-1)
        p2 = F.softmax(self.classifier2(feat), dim=-1)

        ce1 = -(q * torch.log(p1 + 1e-5)).sum(dim=-1).mean()
        ce2 = -(q * torch.log(p2 + 1e-5)).sum(dim=-1).mean()

        W1   = self.classifier.logits.weight
        W2   = self.classifier2.logits.weight
        orth = torch.norm(W1.t() @ W2, p='fro')

        return 0.5 * (ce1 + ce2) + self.lam_orth * orth

    def get_src_optimizer(self, lr, weight_decay):
        """Adam over feature extractor and both classifier heads."""
        params = (
            list(self.feature_extractor.parameters()) +
            list(self.classifier.parameters()) +
            list(self.classifier2.parameters())
        )
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def extra_src_state_dict(self):
        """Persist classifier2 alongside self.network in the source checkpoint."""
        return {f"classifier2.{k}": v
                for k, v in self.classifier2.state_dict().items()}

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def load_source_model(self, state_dict):
        """
        Load source checkpoint. Handles both single-head (legacy) and two-head
        checkpoints. If classifier2 is absent, h2 is initialised from h1 with a
        small random perturbation to break weight symmetry.
        """
        net_sd  = {k: v for k, v in state_dict.items()
                   if k in self.network.state_dict()}
        clf2_sd = {k[len("classifier2."):]: v
                   for k, v in state_dict.items()
                   if k.startswith("classifier2.")}

        self.network.load_state_dict(net_sd, strict=False)

        if clf2_sd:
            self.classifier2.load_state_dict(clf2_sd)
        else:
            sd2 = {k: v.clone() + 0.01 * torch.randn_like(v)
                   for k, v in self.classifier.state_dict().items()}
            self.classifier2.load_state_dict(sd2)

    # ── Threshold via Mixup ───────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_w0(self, trg_loader):
        """
        Estimate w0 = E[iscore(0.5*xi + 0.5*xj)] over random target pairs.
        At most 200 mixed samples are used for efficiency.
        """
        self.feature_extractor.eval()
        self.classifier.eval()
        self.classifier2.eval()

        data_list = [x.float() for x, _, _ in trg_loader]
        all_x = torch.cat(data_list, dim=0)
        N     = all_x.size(0)
        n     = min(200, max(N // 2, 1))

        idx1 = torch.randperm(N)[:n]
        idx2 = torch.randperm(N)[:n]

        scores = []
        for s in range(0, n, 64):
            e  = min(s + 64, n)
            xm = 0.5 * all_x[idx1[s:e]].to(self.device) \
               + 0.5 * all_x[idx2[s:e]].to(self.device)
            f  = self.feature_extractor(xm)
            p1 = F.softmax(self.classifier(f),  dim=-1)
            p2 = F.softmax(self.classifier2(f), dim=-1)
            scores.append((p1 * p2).sum(dim=-1).cpu())

        return float(torch.cat(scores).mean())

    # ── Inference ─────────────────────────────────────────────────────────────

    def _build_banks(self, indexed_dataset, batch_size):
        """Extend base _build_banks to also populate self._iscore_bank."""
        feat_bank, pred_bank = super()._build_banks(indexed_dataset, batch_size)

        if not self._use_maxsoftmax_inference and not self.auto_threshold:
            N = len(indexed_dataset)
            iscore_bank = torch.zeros(N)
            loader = DataLoader(indexed_dataset, batch_size=batch_size * 2,
                                shuffle=False, drop_last=False, num_workers=0)
            self.feature_extractor.eval()
            self.classifier2.eval()
            with torch.no_grad():
                for x, _, idx in loader:
                    x = x.float().to(self.device)
                    f  = self.feature_extractor(x)
                    p1 = F.softmax(self.classifier(f),  dim=-1)
                    p2 = F.softmax(self.classifier2(f), dim=-1)
                    iscore_bank[idx] = (p1 * p2).sum(dim=-1).cpu()
            self._iscore_bank = iscore_bank

        return feat_bank, pred_bank

    def decision_function(self, preds):
        """
        Paper-faithful inference: classify as unknown where iscore < w0.
        iscore = <p1(x), p2(x)> is computed by _build_banks and stored in
        self._iscore_bank.  Falls back to the base-class max-softmax path when
        _use_maxsoftmax_inference=True or _w0_inference is not yet set.
        """
        # Priority 1: auto-threshold (Yen/Otsu/…) on max-softmax — for comparison
        # Priority 2: fixed max-softmax with manually set w_0 hparam
        # Priority 3 (default): paper-faithful iscore with learned w0
        if self.auto_threshold or self._use_maxsoftmax_inference \
                or self._w0_inference is None or not hasattr(self, "_iscore_bank"):
            return super().decision_function(preds)

        _, pred = preds.max(dim=-1)
        pred = pred.clone()
        pred[self._iscore_bank < self._w0_inference] = -1
        return pred

    # ── Flatten helper ────────────────────────────────────────────────────────

    @staticmethod
    def _flatten(q, T):
        """Flatten(p, T)_i = p_i^T / Σ_j p_j^T  — makes p more uniform."""
        q_T = q.pow(T)
        return q_T / (q_T.sum() + 1e-5)

    # ── Bilateral loss terms ──────────────────────────────────────────────────

    def _llmi(self, p1, p2):
        """
        Localized MI for known samples, averaged over both heads.
        Llmi = E_v [ E_{Xt+}[p_v log p_v]  −  DKL(Q_v || Flatten(Q_v, T)) ]
        Positive Llmi → maximised (subtracted in Ltgt).
        """
        T   = self.T_flat
        val = p1.new_zeros(1).squeeze()
        for p in (p1, p2):
            ent = (p * torch.log(p + 1e-5)).sum(dim=-1).mean()
            Q   = p.mean(dim=0)
            Q_h = self._flatten(Q.detach(), T)
            kl  = (Q * (torch.log(Q + 1e-5) - torch.log(Q_h + 1e-5))).sum()
            val = val + ent - kl
        return 0.5 * val

    def _lunk(self, p1, p2):
        """
        Entropic open-set loss for unknown samples, averaged over both heads.
        Lunk = −E_v [ E_{Xt-}[ Σ_k (1/K) log p_{v,k} ] ]
        """
        K   = self.configs.num_classes
        val = p1.new_zeros(1).squeeze()
        for p in (p1, p2):
            val = val + (-(1.0 / K) * torch.log(p + 1e-5).sum(dim=-1)).mean()
        return 0.5 * val

    # ── Per-epoch training ────────────────────────────────────────────────────

    def _training_epoch(self, trg_loader, w0, rho, avg_meter, iter_state):
        self.feature_extractor.train()
        self.classifier.eval()
        self.classifier2.eval()

        T_total = iter_state["T_total"]

        for x, _, _ in trg_loader:
            x = x.float().to(self.device)

            # Polynomial LR decay: (1 + 10·t/T)^(−0.75)  (following SHOT)
            t        = iter_state["t"]
            lr_scale = (1.0 + 10.0 * t / max(T_total, 1)) ** (-0.75)
            for pg in self.optimizer.param_groups:
                pg["lr"] = iter_state["lr0"][id(pg)] * lr_scale

            f  = self.feature_extractor(x)
            p1 = F.softmax(self.classifier(f),  dim=-1)   # [B, K]
            p2 = F.softmax(self.classifier2(f), dim=-1)   # [B, K]
            wt = (p1 * p2).sum(dim=-1)                    # [B]

            known_m = wt > (w0 + rho)
            unk_m   = wt < (w0 - rho)

            Lunk = self._lunk(p1[unk_m],   p2[unk_m])   if unk_m.any()   \
                   else x.new_zeros(1).squeeze()
            Llmi = self._llmi(p1[known_m], p2[known_m]) if known_m.any() \
                   else x.new_zeros(1).squeeze()

            loss = Lunk - Llmi

            if loss.grad_fn is None:
                # Both masks empty — every sample is in the ambiguous band.
                # No gradient signal this batch; skip the update step.
                avg_meter["Total_loss"].update(0.0, x.size(0))
                avg_meter["Unk_loss"].update(0.0,  x.size(0))
                avg_meter["LMI_loss"].update(0.0,  x.size(0))
                iter_state["t"] += 1
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_meter["Total_loss"].update(loss.item(), x.size(0))
            avg_meter["Unk_loss"].update(Lunk.item(),  x.size(0))
            avg_meter["LMI_loss"].update(Llmi.item(),  x.size(0))

            iter_state["t"] += 1

    # ── Main update loop ──────────────────────────────────────────────────────

    def update(self, src_loader, trg_loader, avg_meter, logger):
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        for p in self.classifier2.parameters():
            p.requires_grad_(False)

        w0  = self._compute_w0(trg_loader)
        rho = self.rho_frac * w0
        self._w0_inference = w0
        logger.debug(f"[UMAD] w0={w0:.4f}, ρ={rho:.4f}")

        n_epochs  = self.hparams["num_epochs"]
        n_batches = max(len(trg_loader.dataset) // trg_loader.batch_size, 1)
        iter_state = {
            "t":       0,
            "T_total": n_epochs * n_batches,
            "lr0":     {id(pg): pg["lr"] for pg in self.optimizer.param_groups},
        }

        for epoch in range(1, n_epochs + 1):
            self._training_epoch(trg_loader, w0, rho, avg_meter, iter_state)

            logger.debug(f"[Epoch {epoch}/{n_epochs}]")
            for key, val in avg_meter.items():
                logger.debug(f"  {key}: {val.avg:.4f}")
            logger.debug("-------------------------------------")

        if self.auto_threshold:
            self._apply_auto_threshold(trg_loader, logger)

        return self.network.state_dict(), None
