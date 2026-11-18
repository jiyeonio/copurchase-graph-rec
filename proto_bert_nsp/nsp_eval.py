"""Evaluation of the paper, from scores of the 50 candidates of a user (1:49) or of the whole catalogue.

1:49             the target against the 49 fixed negatives of the user; NDCG@k = 1/log2(rank+2) if rank < k (rank from 0)
full catalogue   the target against all items. "masked": the items of the user's history are removed (the target never is),
                 "unmasked": nothing is removed. Ties get the mid-rank (model/core.py eval_full, baselines/base_full_eval.py).
groups           repeat: the target is in the user's history; novel: it is not

The 1:49 evaluators of the paper take the position of the target in an argsort; that is this rank whenever no negative has exactly
the score of the target. Such ties get the mid-rank here and their number is reported (tied_users). Scores that are not finite stop
the evaluation. The layout of the results is that of baselines/base_full_eval.py.
"""
import numpy as np

KS = (5, 10)


def _ndcg(rank, k):
    return float(1.0 / np.log2(rank + 2.0)) if rank < k else 0.0


def _rank(scores, t):
    """0-based rank of the score t among scores, mid-rank among equal scores (t itself is one of them)"""
    return float((scores > t).sum() + 0.5 * ((scores == t).sum() - 1))


def _finite(scores, uid):
    scores = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"user {uid}: non-finite scores")
    return scores


def evaluate_candidates(scores, split):
    """scores[uid]: the scores of split.cand[uid] (target first). -> ({"1to49": {uid: {group, ndcg@5, ndcg@10}}}, tied users)"""
    out, tied = {}, 0
    for u in split.users:
        s = _finite(scores[u], u)
        if s.shape != (len(split.cand[u]),):
            raise ValueError(f"user {u}: {s.shape} scores for {len(split.cand[u])} candidates")
        tied += int((s == s[0]).sum() > 1)
        rank = _rank(s, s[0])
        out[u] = {"group": "repeat" if split.target[u] in split.seen[u] else "novel", **{f"ndcg@{k}": _ndcg(rank, k) for k in KS}}
    return {"1to49": out}, tied


def evaluate_full(scores, split, num_items):
    """scores[uid]: the scores of all items. -> ({"1to49": ..., "full": {uid: {unmasked@k, masked@k, group}}}, tied users of 1:49)
    The 1:49 part ranks the same scores on the candidates of the user."""
    out = {"1to49": {}, "full": {}}
    tied = 0
    for u in split.users:
        s = _finite(scores[u], u)
        if s.shape != (num_items,):
            raise ValueError(f"user {u}: scores of shape {s.shape}, expected {(num_items,)}")
        target, group = split.target[u], "repeat" if split.target[u] in split.seen[u] else "novel"
        c = s[np.asarray(split.cand[u])]
        tied += int((c == c[0]).sum() > 1)
        r49 = _rank(c, c[0])
        out["1to49"][u] = {"group": group, **{f"ndcg@{k}": _ndcg(r49, k) for k in KS}}
        masked = s.copy()
        seen = [i for i in split.seen[u] if i != target]
        if seen:
            masked[seen] = -np.inf
        ranks = {"unmasked": _rank(s, s[target]), "masked": _rank(masked, s[target])}
        out["full"][u] = {**{f"{tag}@{k}": _ndcg(ranks[tag], k) for tag in ("unmasked", "masked") for k in KS}, "group": group}
    return out, tied


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def summarize_1to49(p49):
    S = {}
    for g in ("overall", "novel", "repeat"):
        rows = [r for r in p49.values() if g == "overall" or r["group"] == g]
        S[g] = {"n": len(rows), **{f"ndcg@{k}": _mean([r[f"ndcg@{k}"] for r in rows]) for k in KS}}
    return S


def summarize_full(pfull):
    """the keys of a full_eval.json row of baselines/base_full_eval.py"""
    S = {}
    for tag in ("masked", "unmasked"):
        for k in KS:
            for g in ("overall", "novel", "repeat"):
                S[f"full_{tag}_ndcg@{k}_{g}"] = _mean([r[f"{tag}@{k}"] for r in pfull.values() if g == "overall" or r["group"] == g])
    S["n"] = len(pfull)
    S["n_novel"] = sum(r["group"] == "novel" for r in pfull.values())
    return S
