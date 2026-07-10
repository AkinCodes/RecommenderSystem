"""Train SASRec (self-attentive next-item recommender) on MovieLens 100K.

Recipe (mirrors scripts/experiment_bpr.py's negative-sampling idiom):
  * Build per-user chronological liked-item sequences from the 0.80 train slice.
  * BPR next-item loss: for each position t (predicting item t+1) sample ONE
    negative item not in the user's liked set; maximise the score margin of the
    true next item over the negative.
  * Leave-last-one-out (LOO) validation for early stopping: the final item of
    each user's sequence is held out; validation HR@10 (true next item vs. 100
    sampled negatives) drives best-checkpoint selection.

Evaluation reports BOTH standard SASRec protocols:
  (a) LOO next-item retrieval over the FULL catalogue: HR@10 / NDCG@10 / MRR
      (true next item ranked against every item in the catalogue).
  (b) Re-ranking the SAME held-out test candidates the DLRM is scored on, via
      dot(encode(history), item_emb(candidate)), through the EXISTING
      compute_ranking_metrics — so the numbers line up with the DLRM table.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.preprocessing import (  # noqa: E402
    build_id_mappings,
    build_user_sequences,
    load_movielens_data,
    prepare_splits,
)
from models.sasrec import PAD_IDX, SASRec  # noqa: E402
from scripts.retrain_and_compare import compute_ranking_metrics  # noqa: E402

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ml-100k", "u.data")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "trained_sasrec_movielens.pth")

EMBEDDING_DIM = 64
MAX_LEN = 50
NUM_BLOCKS = 2
NUM_HEADS = 1
DROPOUT = 0.2
EPOCHS = 30
BATCH_SIZE = 128
LR = 0.001
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 5
K = 10
NUM_VAL_NEGS = 100  # negatives per user for LOO validation ranking
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------------------- #
# Sequence tensors
# --------------------------------------------------------------------------- #
def pad_left(seq_ids, max_len):
    """Left-pad (or truncate to the last max_len) a list of SASRec item ids."""
    seq = seq_ids[-max_len:]
    return [PAD_IDX] * (max_len - len(seq)) + seq


def build_training_examples(train_sequences):
    """From each user's train sequence, form (history, positive-next) pairs.

    ``train_sequences`` values are SASRec ids (item2idx + 1). For a sequence
    [i0, i1, i2] we produce histories ending at each prefix predicting the next:
    (…i0 -> i1), (…i0,i1 -> i2). The user's full liked set is kept for negative
    rejection.
    """
    examples = []          # list of (padded_history_list, positive_id)
    user_pos_sets = []     # aligned liked-set per example (for neg rejection)
    for ids in train_sequences.values():
        if len(ids) < 2:
            continue
        liked = set(ids)
        for t in range(1, len(ids)):
            history = pad_left(ids[:t], MAX_LEN)
            examples.append((history, ids[t]))
            user_pos_sets.append(liked)
    return examples, user_pos_sets


def sample_negatives(user_pos_sets, num_items, rng):
    """One uniform negative per example, rejecting items in that user's liked set."""
    negs = rng.integers(1, num_items + 1, size=len(user_pos_sets))  # 1..num_items (skip PAD)
    for j, liked in enumerate(user_pos_sets):
        while int(negs[j]) in liked:
            negs[j] = rng.integers(1, num_items + 1)
    return negs


def bpr_score(model, hist_tensor, item_tensor):
    """Dot product of the encoded history with an item embedding => a score."""
    user_vec = model.encode(hist_tensor)              # (B, D)
    item_vec = model.item_vectors(item_tensor)        # (B, D)
    return (user_vec * item_vec).sum(dim=1)           # (B,)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(model, examples, user_pos_sets, num_items, val_pairs, val_sequences):
    rng = np.random.default_rng(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    histories = np.array([h for h, _ in examples], dtype=np.int64)   # (E, MAX_LEN)
    positives = np.array([p for _, p in examples], dtype=np.int64)   # (E,)
    n = len(examples)
    print(f"  {n} (history, next-item) training pairs from {len(val_sequences)} users")

    best_hr, best_state, no_improve = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        negatives = sample_negatives(user_pos_sets, num_items, rng)
        order = rng.permutation(n)
        model.train()
        tot, nb = 0.0, 0
        for start in range(0, n, BATCH_SIZE):
            b = order[start : start + BATCH_SIZE]
            hist = torch.tensor(histories[b])
            pos = torch.tensor(positives[b])
            neg = torch.tensor(negatives[b])

            opt.zero_grad()
            s_pos = bpr_score(model, hist, pos)
            s_neg = bpr_score(model, hist, neg)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1

        val_hr = loo_validation_hr(model, val_pairs, num_items, rng)
        if val_hr > best_hr:
            best_hr = val_hr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(f"  Epoch {epoch:2d}/{EPOCHS} -- loss {tot / nb:.4f}  "
              f"val HR@10 {val_hr:.4f}  no_improve {no_improve}")
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Trained in {time.time() - t0:.1f}s (best val HR@10 {best_hr:.4f})")
    return best_hr


def loo_validation_hr(model, val_pairs, num_items, rng):
    """HR@10: rank each user's held-out true next item against NUM_VAL_NEGS negs."""
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for history, target, liked in val_pairs:
            negs = []
            while len(negs) < NUM_VAL_NEGS:
                cand = int(rng.integers(1, num_items + 1))
                if cand not in liked and cand != target:
                    negs.append(cand)
            candidates = torch.tensor([target] + negs)
            hist = torch.tensor([history])
            user_vec = model.encode(hist)                        # (1, D)
            scores = (user_vec * model.item_vectors(candidates)).sum(dim=1)  # (1+neg,)
            rank = int((scores > scores[0]).sum().item())        # 0-based rank of target
            if rank < K:
                hits += 1
            total += 1
    return hits / total if total else 0.0


# --------------------------------------------------------------------------- #
# LOO full-catalogue retrieval evaluation
# --------------------------------------------------------------------------- #
def loo_full_catalogue(model, loo_pairs, num_items):
    """Rank each held-out true next item against the ENTIRE catalogue.

    Returns HR@10, NDCG@10, MRR — the standard SASRec retrieval protocol.
    """
    model.eval()
    hrs, ndcgs, mrrs = [], [], []
    all_item_ids = torch.arange(1, num_items + 1)        # skip PAD row 0
    with torch.no_grad():
        item_mat = model.item_vectors(all_item_ids)      # (num_items, D)
        for history, target, liked in loo_pairs:
            hist = torch.tensor([history])
            user_vec = model.encode(hist)[0]             # (D,)
            scores = item_mat @ user_vec                 # (num_items,)
            # Mask items the user already liked (except the target) so we don't
            # penalise the model for ranking known-liked items above the target.
            for it in liked:
                if it != target:
                    scores[it - 1] = -1e9                # -1 to map id->row index
            target_score = scores[target - 1]
            rank = int((scores > target_score).sum().item())  # 0-based
            hrs.append(1.0 if rank < K else 0.0)
            ndcgs.append(1.0 / np.log2(rank + 2) if rank < K else 0.0)
            mrrs.append(1.0 / (rank + 1))
    return {
        "HR@10": float(np.mean(hrs)),
        "NDCG@10": float(np.mean(ndcgs)),
        "MRR": float(np.mean(mrrs)),
        "num_users": len(hrs),
    }


# --------------------------------------------------------------------------- #
# Data prep helpers
# --------------------------------------------------------------------------- #
def build_loo_pairs(sequences):
    """Leave-last-one-out: (history-without-last, last-item, liked-set) per user."""
    pairs = []
    for ids in sequences.values():
        if len(ids) < 2:
            continue
        history = pad_left(ids[:-1], MAX_LEN)
        pairs.append((history, ids[-1], set(ids)))
    return pairs


def make_sasrec_rank_fn(model, sequences_by_user, user2idx):
    """rank_fn(uidx, item_indices) -> scores, for compute_ranking_metrics reuse.

    Scores each candidate by dot(encode(user's train history), item_emb(candidate)).
    Candidate item_indices are item2idx values; shift +1 for the SASRec table.
    Users with no train history fall back to zero scores (uniform).
    """
    idx2user = {v: k for k, v in user2idx.items()}
    cache = {}

    def rank_fn(uidx, item_indices):
        if uidx not in cache:
            uid = idx2user.get(uidx)
            ids = sequences_by_user.get(uid, [])
            seq_ids = [i + 1 for i in ids]               # item2idx -> SASRec id
            if seq_ids:
                hist = torch.tensor([pad_left(seq_ids, MAX_LEN)])
                with torch.no_grad():
                    cache[uidx] = model.encode(hist)[0]  # (D,)
            else:
                cache[uidx] = None
        user_vec = cache[uidx]
        if user_vec is None:
            return np.zeros(len(item_indices), dtype=np.float32)
        cand = torch.tensor([i + 1 for i in item_indices])   # item2idx -> SASRec id
        with torch.no_grad():
            scores = (model.item_vectors(cand) @ user_vec).numpy()
        return scores

    return rank_fn


def main():
    print("=" * 72)
    print("TRAIN SASRec -- self-attentive next-item recommender (MovieLens 100K)")
    print("=" * 72)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    raw = load_movielens_data(DATA_PATH)
    user2idx, item2idx = build_id_mappings(raw)
    num_items = len(item2idx)

    # Full liked-item sequences from the 0.80 train slice (item2idx values).
    seqs_item2idx = build_user_sequences(raw, item2idx, like_threshold=4, train_ratio=0.8)
    # Shift to SASRec ids (PAD=0) for the model.
    seqs_sasrec = {u: [i + 1 for i in ids] for u, ids in seqs_item2idx.items()}

    # Leave-last-one-out: last item is the validation/LOO target; train on the rest.
    train_sequences = {u: ids[:-1] for u, ids in seqs_sasrec.items() if len(ids) >= 2}
    loo_pairs = build_loo_pairs(seqs_sasrec)

    print(f"  Users with >=2 liked train items: {len(train_sequences)}")
    print(f"  Catalogue size (num_items): {num_items}")

    examples, user_pos_sets = build_training_examples(train_sequences)

    model = SASRec(
        num_items=num_items, embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN,
        num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS, dropout=DROPOUT,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  SASRec parameters: {total_params:,}")

    print("\n[1] Training (BPR next-item, LOO early stopping)...")
    train(model, examples, user_pos_sets, num_items, loo_pairs, train_sequences)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  Model saved to {MODEL_SAVE_PATH}")

    print("\n[2] LOO full-catalogue next-item retrieval (standard SASRec protocol)...")
    loo = loo_full_catalogue(model, loo_pairs, num_items)
    print(f"  Evaluated {loo['num_users']} users (true next item vs. full catalogue)")
    print(f"  HR@10:   {loo['HR@10']:.4f}")
    print(f"  NDCG@10: {loo['NDCG@10']:.4f}")
    print(f"  MRR:     {loo['MRR']:.4f}")

    print("\n[3] Re-ranking the DLRM's held-out test candidates (comparable table)...")
    (_tc, _tcat, _tt, test_cont, test_cat, test_targets,
     u2i, i2i, test_raw) = prepare_splits(raw)
    user_test = {}
    for i in range(len(test_raw)):
        uidx = u2i[test_raw[i, 0]]
        user_test.setdefault(uidx, []).append((int(test_cat[i, 1]), test_targets[i]))

    rank_fn = make_sasrec_rank_fn(model, seqs_item2idx, u2i)
    rerank = compute_ranking_metrics(user_test, rank_fn, k=K)
    print(f"  Evaluated {rerank['num_eval_users']} users (same protocol as DLRM table)")
    print(f"  NDCG@10:   {rerank['NDCG@10']:.4f}")
    print(f"  Recall@10: {rerank['Recall@10']:.4f}")
    print(f"  HR@10:     {rerank['HitRate@10']:.4f}")
    print(f"  MRR:       {rerank['MRR']:.4f}")
    print(f"  AUC:       {rerank['AUC']:.4f}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  LOO full-catalogue retrieval:  HR@10 {loo['HR@10']:.4f}  "
          f"NDCG@10 {loo['NDCG@10']:.4f}  MRR {loo['MRR']:.4f}")
    print(f"  Re-ranking DLRM candidates:    NDCG@10 {rerank['NDCG@10']:.4f}  "
          f"HR@10 {rerank['HitRate@10']:.4f}  MRR {rerank['MRR']:.4f}  AUC {rerank['AUC']:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
