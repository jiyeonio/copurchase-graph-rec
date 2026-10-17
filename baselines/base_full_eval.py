"""Full-catalogue evaluation of the baselines (SASRec, BERT4Rec, GRU4Rec, LightGCN) without changing their scripts.

The baseline script (baselines/original/{kc,ic}_<model>.py) is read from disk and everything before its final "run seeds" block is executed
unchanged (data loading, model class, training function, its own 1:49 evaluator). Per seed this wrapper then
  * loads an existing checkpoint (--ckpt-dir) if there is one, otherwise calls the script's own train_one_seed,
  * runs the script's own 1:49 evaluator,
  * ranks the target against all items with the same scoring function.
Full-catalogue ranking as in model/core.py: "masked" removes the items of the user's history from the ranking (the target is never
removed); ties get the mid-rank; NDCG@k = 1/log2(rank+2) if rank < k.

usage: base_full_eval.py SCRIPT KIND --data-dir D [--emb-dir E] --out-dir O [--ckpt-dir C] [--seeds ...]
"""
import sys, os, re, json, argparse, time

ap = argparse.ArgumentParser()
ap.add_argument("script"); ap.add_argument("kind", choices=["sasrec", "gru4rec", "bert4rec", "lightgcn"])
ap.add_argument("--data-dir", required=True); ap.add_argument("--emb-dir", default=None)
ap.add_argument("--out-dir", required=True); ap.add_argument("--ckpt-dir", default=None)
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
A = ap.parse_args()

src = open(A.script, encoding="utf-8").read()
cut = re.search(r"^all_seed_results = \[\]$", src, flags=re.M).start()     # the final "run seeds" block of the script starts here
assert "= train_one_seed(seed)" not in src[:cut] and "= evaluate_seed(seed)" not in src[:cut]
sys.argv = [A.script, "--data-dir", A.data_dir, "--out-dir", A.out_dir, "--seeds", *map(str, A.seeds)] + (["--emb-dir", A.emb_dir] if A.emb_dir else [])
NS = {"__name__": "__baseline__", "__file__": A.script}
exec(compile(src[:cut], A.script, "exec"), NS)

import numpy as np, torch
device = NS["device"]; NUM_ITEMS = NS["NUM_ITEMS"]
ALL = torch.arange(NUM_ITEMS, dtype=torch.long, device=device)
is_kc = os.path.basename(A.script).startswith("kc_")
CK = {("sasrec", True): "best_katcher_sasrec_original_bce1neg_seed{}.pt", ("bert4rec", True): "best_katcher_bert4rec_seed{}.pt",
      ("gru4rec", True): "best_katcher_gru4rec_bprmax_adapted_seed{}.pt", ("sasrec", False): "best_instacart_sasrec_original_bce1neg_seed{}.pt",
      ("bert4rec", False): "best_instacart_bert4rec_seed{}.pt", ("gru4rec", False): "best_instacart_gru4rec_bprmax_adapted_seed{}.pt"}


def build(kind):
    if kind == "sasrec":
        return NS["SASRecBaseline"](num_items=NUM_ITEMS, hidden_dim=NS["HIDDEN_DIM"], max_seq_len=NS["MAX_ITEM_SEQ_LEN"], num_blocks=NS["NUM_BLOCKS"],
                                    num_heads=NS["NUM_HEADS"], ff_dim=NS["FF_DIM"], dropout=NS["DROPOUT"]).to(device)
    if kind == "bert4rec":
        return NS["BERT4RecBaseline"](num_items=NUM_ITEMS, hidden_dim=NS["HIDDEN_DIM"], max_seq_len=NS["MAX_SEQ_LEN"], num_blocks=NS["NUM_BLOCKS"],
                                      num_heads=NS["NUM_HEADS"], ff_dim=NS["FF_DIM"], dropout=NS["DROPOUT"]).to(device)
    return NS["GRU4RecBaseline"](NUM_ITEMS, embed_dim=NS["EMBED_DIM"], hidden_dim=NS["HIDDEN_DIM"], num_layers=NS["NUM_GRU_LAYERS"], dropout=NS["DROPOUT"]).to(device)


def rank_metrics(scores, target, seen):
    """scores: (NUM_ITEMS,) float tensor. Returns dict of ndcg for masked / unmasked."""
    scores = scores.float(); t = scores[target]
    out = {}
    for tag in ("unmasked", "masked"):
        s = scores
        if tag == "masked":
            s = scores.clone(); idx = [i for i in seen if i != target and 0 <= i < NUM_ITEMS]
            if idx: s[torch.tensor(idx, dtype=torch.long, device=s.device)] = float("-inf")
        rank = float((s > t).sum() + 0.5 * ((s == t).sum() - 1))
        for k in (5, 10): out[f"{tag}@{k}"] = float(1.0 / np.log2(rank + 2)) if rank < k else 0.0
    return out


@torch.no_grad()
def full_eval(model, gemb=None):
    if model is not None: model.eval()
    pu = {}
    if A.kind == "lightgcn":
        users = list(NS["TEST_CANDIDATES"].keys()); tg = {u: int(NS["TEST_CANDIDATES"][u][0]) for u in users}; seen = NS["TEST_SEEN_FULL"]
    else:
        users = list(NS["test_targets"].keys()); tg = {u: int(NS["test_targets"][u]) for u in users}; seen = NS["TEST_SEEN_ITEMS"]
    for u in users:
        if A.kind == "lightgcn":
            h = torch.tensor(NS["TEST_HISTORY_ITEMS"][u], dtype=torch.long, device=device)
            sc = gemb @ gemb[h].mean(0)
        else:
            seq = NS["test_sequences"].get(u, [])
            if A.kind == "bert4rec":
                prep = NS["build_eval_input_with_mask"](seq)
                if prep is None: continue
                sc = model.score_candidates_at_mask(input_tokens=prep[0], mask_position=prep[1], candidate_ids=ALL)
            else:
                prep = NS["single_sequence_to_tensor"](seq)
                if prep is None: continue
                sc = model.score_candidates(prep[0], prep[1], ALL)
        r = rank_metrics(sc, tg[u], seen[u]); r["group"] = "repeat" if tg[u] in seen[u] else "novel"; pu[u] = r
    return pu


def summarize(pu):
    S = {}
    for tag in ("masked", "unmasked"):
        for k in (5, 10):
            for g in ("overall", "novel", "repeat"):
                v = [r[f"{tag}@{k}"] for r in pu.values() if g == "overall" or r["group"] == g]
                S[f"full_{tag}_ndcg@{k}_{g}"] = float(np.mean(v)) if v else None
    S["n"] = len(pu); S["n_novel"] = sum(r["group"] == "novel" for r in pu.values())
    return S


os.makedirs(A.out_dir, exist_ok=True)
res = []
for seed in A.seeds:
    t0 = time.time(); src_ck = "none"; best_val = None
    if A.kind == "lightgcn":
        gp = NS["GRAPH_EMB_TEMPLATE"].format(seed=seed) if is_kc else NS["graph_embedding_path"](seed)
        gemb = torch.tensor(np.load(gp), dtype=torch.float32, device=device)
        g149, pu149 = NS["evaluate_seed"](seed); pu = full_eval(None, gemb); src_ck = gp
    else:
        ck = os.path.join(A.ckpt_dir, CK[(A.kind, is_kc)].format(seed)) if A.ckpt_dir else None
        if ck and os.path.exists(ck):
            NS["set_seed"](seed); model = build(A.kind); model.load_state_dict(torch.load(ck, map_location=device)); model.eval(); src_ck = ck
        else:
            model, best_val = NS["train_one_seed"](seed); src_ck = "retrained:" + str(NS["BEST_MODEL_PATH"])
        g149, pu149 = NS["evaluate_grouped_test"](model); pu = full_eval(model)
    S = summarize(pu)
    row = {"seed": seed, "checkpoint": src_ck, "best_val_ndcg@10": best_val, "test_1to49": g149, **S, "minutes": (time.time() - t0) / 60}
    res.append(row)
    json.dump({"1to49": pu149, "full": pu}, open(os.path.join(A.out_dir, f"per_user_seed{seed}.json"), "w"))
    json.dump(res, open(os.path.join(A.out_dir, "full_eval.json"), "w"), indent=1)
    print(f"[{A.kind} seed {seed}] 1:49 NDCG@10 {g149['overall']['ndcg@10']:.4f} (n={g149['overall']['n']}) | full masked {S['full_masked_ndcg@10_overall']:.4f} "
          f"novel {S['full_masked_ndcg@10_novel']:.4f} repeat {S['full_masked_ndcg@10_repeat']:.4f} | unmasked {S['full_unmasked_ndcg@10_overall']:.4f} | n={S['n']} n_novel={S['n_novel']} | {src_ck}", flush=True)

def ms(key, f=lambda r: r):
    v = np.array([f(r)[key] if not callable(key) else key(r) for r in res], dtype=float); return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
agg = {"1to49_ndcg@10": ms(lambda r: r["test_1to49"]["overall"]["ndcg@10"]), "1to49_ndcg@5": ms(lambda r: r["test_1to49"]["overall"]["ndcg@5"])}
for k in res[0]:
    if k.startswith("full_"): agg[k] = ms(k)
json.dump({"script": A.script, "kind": A.kind, "seeds": A.seeds, "per_seed": res, "aggregate_mean_std": agg}, open(os.path.join(A.out_dir, "full_eval.json"), "w"), indent=1)
print("AGG", json.dumps({k: [round(x, 4) for x in v] for k, v in agg.items()}), flush=True)
