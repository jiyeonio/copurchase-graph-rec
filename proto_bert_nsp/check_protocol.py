"""Checks that this prototype reads the data and evaluates as the baselines of the paper do, and that its sentences are built from the
same users, orders and targets. Reads only; trains nothing; needs the tokenizer but not the model.

  1. the data-loading part of baselines/original/kc_sasrec.py is executed unchanged on the same folder (as baselines/base_full_eval.py
     does) and compared with nsp_data.py: users, item sequences, targets, validation negatives, fixed 1:49 test candidates, seen items
  2. the split files (the rows with the text) against the data folder: numbering of items and users, users of each split, orders, targets
  3. per-user NDCG of one fixed random scorer: nsp_eval.py against the baseline's evaluate_grouped_test (1:49) and against rank_metrics
     of baselines/base_full_eval.py (full catalogue, masked and unmasked)
  4. the sentences: one candidate text per item; every 1:49 pair of the test users assembled from token ids equals the tokenizer's own
     encoding of the pair and has at most 512 tokens

usage: python proto_bert_nsp/check_protocol.py [--data-dir D] [--split-dir S] [--emb-dir E]
       --emb-dir: the folder of item_embeddings_text.npy (the baseline takes the number of items from it)
"""
import argparse
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, ROOT)
import nsp_data
import nsp_eval
import nsp_text
import paths

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data-dir", default=paths.KC)
p.add_argument("--split-dir", default=paths.KC_SPLIT)
p.add_argument("--emb-dir", default=None)
p.add_argument("--num-items", type=int, default=None)
p.add_argument("--model", default=nsp_text.MODEL, help="the tokenizer of this model is used")
A = p.parse_args()
data_dir = os.path.realpath(A.data_dir)
ds = nsp_data.load(data_dir, A.num_items or (None if os.path.exists(os.path.join(data_dir, "item2idx.json")) else paths.N_ITEMS["kc"]))
rows = nsp_data.load_rows(A.split_dir)
N = ds.num_items
print(nsp_data.describe(ds, rows))

failures = 0


def same(what, mine, theirs):
    global failures
    ok = mine == theirs
    failures += not ok
    print(("OK      " if ok else "DIFFERS ") + what)


def flat(orders):
    return [i for o in orders for i in o]


# ---- 1. the baseline's own loading code
tmp = tempfile.mkdtemp()
script = os.path.join(ROOT, "baselines", "original", "kc_sasrec.py")
emb = A.emb_dir or os.path.dirname(paths.KC_TEXT)
if not os.path.exists(os.path.join(emb, "item_embeddings_text.npy")):
    emb = tmp
    np.save(os.path.join(tmp, "item_embeddings_text.npy"), np.zeros((N, 1), dtype=np.float32))
    print(f"NOTE no item_embeddings_text.npy found: the baseline is given the number of items of item2idx.json ({N})")
src = open(script, encoding="utf-8").read()
cut = re.search(r"^all_seed_results = \[\]$", src, flags=re.M).start()
B = {"__name__": "__baseline__", "__file__": script}
old_argv, sys.argv = sys.argv, [script, "--data-dir", data_dir, "--out-dir", tmp, "--seeds", "42", "--emb-dir", emb]
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src[:cut], script, "exec"), B)
sys.argv = old_argv

same("number of items", N, B["NUM_ITEMS"])
same("training users and item sequences", {u: flat(ds.train.orders[u]) for u in ds.train.users},
     {u: B["train_sequences"][u] for u in B["train_users"] if B["train_sequences"][u]})
same("training targets", ds.train.target, {u: B["train_targets"][u] for u in ds.train.users})
for name, split in (("validation", ds.val), ("test", ds.test)):
    key = name[:3] if name == "validation" else name
    seqs = {u: B[f"{key}_sequences"][u] for u in B[f"{key}_users"] if B[f"{key}_sequences"][u]}
    if name == "validation":
        cand = {u: B["deterministic_val_negatives"](u, s, B["val_targets"][u], 49) for u, s in seqs.items()}
        cand = {u: [B["val_targets"][u], *negs] for u, negs in cand.items() if negs is not None}
    else:
        cand = {u: [B["test_targets"][u], *B["negative_test_dict"][u][:49]] for u in seqs}
    same(f"{name} users and item sequences", {u: flat(split.orders[u]) for u in split.users}, {u: seqs[u] for u in cand})
    same(f"{name} targets and 1:49 candidates", split.cand, cand)
same("seen items of the test users", ds.test.seen, {u: B["TEST_SEEN_ITEMS"][u] for u in ds.test.users})

# ---- 2. the rows with the text against the data folder
diff = nsp_data.match(ds, rows)
same("split files and data folder: numbering, users, orders and targets" + "".join(f"\n          {d}" for d in diff), diff, [])


# ---- 3. one fixed random scorer through both evaluators
class Scorer:
    def __init__(self, device):
        self.table = torch.randn(N + 1, 16, generator=torch.Generator().manual_seed(0)).to(device)

    def eval(self):
        return self

    def all_items(self, tokens):                          # tokens: item index + 1
        return self.table[1:] @ self.table[tokens].mean(0)

    def score_candidates(self, input_tokens, lengths, candidate_ids):       # interface of the baseline models
        return self.all_items(input_tokens[0])[candidate_ids]


device = B["device"]
model = Scorer(device)
scores = {u: model.all_items(torch.tensor(flat(ds.test.orders[u]), device=device) + 1) for u in ds.test.users}
full = {u: s.float().cpu().numpy() for u, s in scores.items()}
mine49, tied = nsp_eval.evaluate_candidates({u: s[ds.test.cand[u]] for u, s in full.items()}, ds.test)
mine, _ = nsp_eval.evaluate_full(full, ds.test, N)
_, theirs = B["evaluate_grouped_test"](model)
same(f"1:49 per-user NDCG@5/@10 and groups of a random scorer ({len(mine49['1to49'])} test users, {tied} with ties)", mine49["1to49"], theirs)
same("1:49 from the scores of all items", mine["1to49"], theirs)

full_src = open(os.path.join(ROOT, "baselines", "base_full_eval.py"), encoding="utf-8").read()
m = re.search(r"^def rank_metrics\(.*?(?=^\S)", full_src, flags=re.M | re.S)
if m is None:
    failures += 1
    print("DIFFERS rank_metrics not found in baselines/base_full_eval.py")
else:
    F = {"torch": torch, "np": np, "NUM_ITEMS": N}
    exec(m.group(0), F)
    theirs = {}
    for u in ds.test.users:
        r = F["rank_metrics"](scores[u], ds.test.target[u], B["TEST_SEEN_ITEMS"][u])
        r["group"] = "repeat" if ds.test.target[u] in B["TEST_SEEN_ITEMS"][u] else "novel"
        theirs[u] = r
    same("full-catalogue per-user NDCG (masked and unmasked) of a random scorer", mine["full"], theirs)

# ---- 4. the sentences
texts = nsp_text.Texts(nsp_text.load_tokenizer(A.model))
cand_text = nsp_text.candidate_texts(rows)
same(f"one candidate text per item ({len(set(cand_text))} distinct texts for {len(cand_text)} items)", len(cand_text), N)
if not diff:
    cut_users = bare = 0
    pairs = []
    for u in ds.test.users:
        user = rows.users["test"][u]
        prompt, n = nsp_text.history_prompt(user.orders, texts, texts.longest([cand_text[i] for i in ds.test.cand[u]]))
        cut_users, bare = cut_users + (n < len(user.orders)), bare + (n == 0)
        pairs += [(prompt, cand_text[i]) for i in ds.test.cand[u]]
    try:
        n = nsp_text.check_pairs(pairs, texts)
        print(f"OK      {n} test pairs (1:49): assembled token ids equal the tokenizer's encoding, none above {nsp_text.MAX_LEN} tokens")
    except ValueError as e:
        failures += 1
        print(f"DIFFERS test pairs: {e}")
    twins = sum(any(cand_text[i] == cand_text[ds.test.target[u]] for i in ds.test.cand[u][1:]) for u in ds.test.users)
    print(f"NOTE    test prompts that do not hold every order of the user: {cut_users} of {len(ds.test.users)}; without any order: {bare}")
    print(f"NOTE    test users with a negative whose text equals the text of the target (they tie under 1:49): {twins}")

shutil.rmtree(tmp, ignore_errors=True)
print("ALL OK" if not failures else f"{failures} CHECK(S) DIFFER")
sys.exit(1 if failures else 0)
