"""Controlled experiment: BPR (pairwise ranking loss) vs BCE (pointwise) on the same DLRM.

Isolates the training objective as the ONLY variable:
  - identical architecture (DLRMModel, 8 features, [943,1682] embeddings, MLP [128,64,32])
  - identical seed, optimizer family, epochs
  - identical evaluation (compute_ranking_metrics on the test split)

Evaluation uses proper per-(user,item) continuous features for every candidate
(matching the fixed methodology in retrain_and_compare.py), so item-specific
columns (item_mean/count/popularity, user×item interaction) are correct for
every candidate rather than frozen to one user's cached row.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.preprocessing import NUM_FEATURES, load_movielens_data, prepare_splits  # noqa: E402
from models.classical import FeatureBuilder  # noqa: E402
from models.dlrm import DLRMModel  # noqa: E402
from scripts.retrain_and_compare import compute_ranking_metrics  # noqa: E402

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ml-100k", "u.data")
CARDINALITIES = [943, 1682]
EMBEDDING_DIM = 64
MLP_LAYERS = [128, 64, 32]
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2
K = 10
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def build_model():
    return DLRMModel(
        num_features=NUM_FEATURES,
        cardinalities=CARDINALITIES,
        embedding_dim=EMBEDDING_DIM,
        mlp_layers=MLP_LAYERS,
        dropout=DROPOUT,
    )


def forward_logit(model, cont, cat):
    """Replicates DLRMModel.forward but returns the pre-sigmoid logit (for BPR)."""
    x = model.continuous_layer(cont)
    cat_embeds = [emb(cat[:, i]) for i, emb in enumerate(model.embeddings[: cat.shape[1]])]
    if len(cat_embeds) >= 2:
        interaction = cat_embeds[0] * cat_embeds[1]
        x = torch.cat([x] + cat_embeds + [interaction], dim=1)
    else:
        x = torch.cat([x] + cat_embeds, dim=1)
    x = model.mlp(x)
    return model.output_layer(x).squeeze(-1)


def feats_for(fb, uidx_arr, iidx_arr):
    """Vectorised per-(user,item) 8-feature builder, matching FeatureBuilder.__call__."""
    f = np.zeros((len(uidx_arr), 8), dtype=np.float32)
    f[:, 0] = fb.user_mean[uidx_arr]
    f[:, 1] = fb.user_count[uidx_arr]
    f[:, 2] = fb.user_var[uidx_arr]
    f[:, 3] = fb.user_days_active[uidx_arr]
    f[:, 4] = fb.item_mean[iidx_arr]
    f[:, 5] = fb.item_count[iidx_arr]
    f[:, 6] = fb.item_pop_rank[iidx_arr]
    f[:, 7] = fb.user_mean[uidx_arr] - fb.item_mean[iidx_arr]
    return f


# --------------------------------------------------------------------------- #
# Training -- BCE (pointwise)
# --------------------------------------------------------------------------- #
class PointDS(Dataset):
    def __init__(self, cont, cat, targets):
        self.cont = torch.tensor(cont, dtype=torch.float32)
        self.cat = torch.tensor(cat, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.cont[i], self.cat[i], self.targets[i]


def train_bce(train_cont, train_cat, train_targets):
    torch.manual_seed(SEED)
    model = build_model()
    loader = DataLoader(PointDS(train_cont, train_cat, train_targets), batch_size=BATCH_SIZE, shuffle=True)
    crit = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best_loss, best_state = float("inf"), None
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot, n = 0.0, 0
        for cont_b, cat_b, tgt_b in loader:
            opt.zero_grad()
            loss = crit(model(cont_b, cat_b).squeeze(), tgt_b)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        avg = tot / n
        sched.step(avg)
        if avg < best_loss:
            best_loss, best_state = avg, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    print(f"  [BCE] trained in {time.time() - t0:.1f}s (best loss {best_loss:.4f})")
    return model


# --------------------------------------------------------------------------- #
# Training -- BPR (pairwise)
# --------------------------------------------------------------------------- #
def train_bpr(train_raw, user2idx, item2idx, fb):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    num_items = len(item2idx)

    # Positive interactions (rating >= 4) and per-user positive sets for neg sampling.
    pos_u, pos_i = [], []
    user_pos = {}
    for uid, iid, rating, _ in train_raw:
        if iid not in item2idx:
            continue
        if rating >= 4:
            u, i = user2idx[uid], item2idx[iid]
            pos_u.append(u)
            pos_i.append(i)
            user_pos.setdefault(u, set()).add(i)
    pos_u = np.array(pos_u)
    pos_i = np.array(pos_i)
    print(f"  [BPR] {len(pos_u)} positive interactions, {len(user_pos)} users")

    best_loss, best_state = float("inf"), None
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        # Sample one negative per positive (uniform, reject items in the user's positive set).
        neg_i = rng.integers(0, num_items, size=len(pos_i))
        for j in range(len(neg_i)):
            ps = user_pos[pos_u[j]]
            while neg_i[j] in ps:
                neg_i[j] = rng.integers(0, num_items)

        order = rng.permutation(len(pos_u))
        model.train()
        tot, n = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            b = order[start : start + BATCH_SIZE]
            u_b, ip_b, ineg_b = pos_u[b], pos_i[b], neg_i[b]

            cont_pos = torch.tensor(feats_for(fb, u_b, ip_b))
            cont_neg = torch.tensor(feats_for(fb, u_b, ineg_b))
            cat_pos = torch.tensor(np.stack([u_b, ip_b], axis=1), dtype=torch.long)
            cat_neg = torch.tensor(np.stack([u_b, ineg_b], axis=1), dtype=torch.long)

            opt.zero_grad()
            s_pos = forward_logit(model, cont_pos, cat_pos)
            s_neg = forward_logit(model, cont_neg, cat_neg)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        avg = tot / n
        sched.step(avg)
        if avg < best_loss:
            best_loss, best_state = avg, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    print(f"  [BPR] trained in {time.time() - t0:.1f}s (best loss {best_loss:.4f})")
    return model


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def make_peritem_rank(model, fb):
    def rank(uidx, item_indices):
        u_arr = np.full(len(item_indices), uidx)
        i_arr = np.array(item_indices)
        cont = torch.tensor(feats_for(fb, u_arr, i_arr))
        cat = torch.tensor(np.stack([u_arr, i_arr], axis=1), dtype=torch.long)
        with torch.no_grad():
            s = model(cont, cat).squeeze().numpy()
        return s if s.ndim else np.array([s.item()])

    return rank


def main():
    print("=" * 72)
    print("EXPERIMENT: BPR (pairwise) vs BCE (pointwise) -- same DLRM, same eval")
    print("=" * 72)

    raw = load_movielens_data(DATA_PATH)
    (train_cont, train_cat, train_targets, test_cont, test_cat, test_targets,
     user2idx, item2idx, test_raw) = prepare_splits(raw)

    sorted_idx = np.argsort(raw[:, 3])
    raw_sorted = raw[sorted_idx]
    split = int(len(raw_sorted) * 0.8)
    train_raw = raw_sorted[:split]
    fb = FeatureBuilder(train_raw, user2idx, item2idx)

    # Build the eval structures used by retrain_and_compare.py. Per-item
    # continuous features are rebuilt from the FeatureBuilder at rank time.
    user_test = {}
    for i in range(len(test_raw)):
        uidx = user2idx[test_raw[i, 0]]
        user_test.setdefault(uidx, []).append((int(test_cat[i, 1]), test_targets[i]))

    print("\n[1] Training BCE (pointwise) baseline...")
    bce_model = train_bce(train_cont, train_cat, train_targets)

    print("\n[2] Training BPR (pairwise)...")
    bpr_model = train_bpr(train_raw, user2idx, item2idx, fb)

    print("\n[3] Evaluating both with per-item-feature eval...\n")
    rows = []
    for name, model in [("BCE", bce_model), ("BPR", bpr_model)]:
        peritem = compute_ranking_metrics(user_test, make_peritem_rank(model, fb), k=K)
        rows.append((f"{name} (per-item-feat eval)", peritem))

    print(f"{'Model':<28} {'NDCG@10':>9} {'Prec@10':>9} {'Recall@10':>10} {'HR@10':>7} {'AUC':>7}")
    print("-" * 74)
    for name, m in rows:
        print(f"{name:<28} {m['NDCG@10']:>9.4f} {m['Precision@10']:>9.4f} "
              f"{m['Recall@10']:>10.4f} {m['HitRate@10']:>7.4f} {m['AUC']:>7.4f}")
    print("\nReference (from reports/model_comparison.json):")
    print(f"{'LogReg':<28} {0.8023:>9.4f} {0.6880:>9.4f} {0.4299:>10.4f} {1.0000:>7.4f} {0.6818:>7.4f}")
    print(f"{'XGBoost':<28} {0.7836:>9.4f} {0.6707:>9.4f} {0.4206:>10.4f} {1.0000:>7.4f} {0.6668:>7.4f}")


if __name__ == "__main__":
    main()
