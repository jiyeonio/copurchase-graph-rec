"""BERT-NSP (Park and Oh, 2024) on the data, the split and the evaluation of the paper (prototype, see README.md).

usage
  python proto_bert_nsp/run.py train      stage 1: training and the 1:49 evaluation, one seed after another
  python proto_bert_nsp/run.py full       stage 2: ranking of the whole catalogue with the checkpoint of stage 1 (can be resumed)
  python proto_bert_nsp/run.py collect    full_eval.json from the seeds that are finished (the two stages write it as well)

Reads the data folder and the split folder (never writes to them) and writes to <RESULTS>/proto_bert_nsp/kc/<name>/ :
  full_eval.json            per-seed rows and mean / std over the seeds, with the keys of baselines/base_full_eval.py
  per_user_seed{S}.json     per-user NDCG under 1:49 and full-catalogue ranking, in the layout of baselines/base_full_eval.py
  result_seed{S}.json       stage 1 of one seed with its settings (a finished seed is not trained again)
  full_seed{S}.json         stage 2 of one seed; scores_full_seed{S}.npy holds the scores of all items for every test user
  checkpoints/best_seed{S}.pt

The model, the sentences and the training recipe are those of the re-implementation named in nsp_text.py: mBERT with its
next-sentence head, one positive and one uniformly drawn negative pair per user, 10 epochs, AdamW, learning rate 2e-5 with linear
decay, weight decay 0.01, batch 16. The score of a candidate is logit[1] - logit[0].
What follows the paper instead: the users, targets and candidates, the seeds (42-46), the evaluation, and the checkpoint: the epoch
with the highest validation NDCG@10 under the 1:49 protocol, as for the other models of the paper. The epoch with the lowest
validation loss, the rule of the re-implementation, is evaluated under 1:49 as well and kept as a record (by_val_loss).

The first version of this folder had the two rules the other way round and wrote to the run folder bert_nsp. Where that run chose
one epoch by both rules for a seed, its checkpoint and its results are those of this version, and the seed is taken over from it
instead of being trained again (--earlier-run); the result files say for every seed where it comes from.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import nsp_data
import nsp_eval
import nsp_text
import paths                       # locations of the repository (KATCHERS_DATA, KATCHERS_SPLIT, RESULTS)

CODE = ("nsp_data.py", "nsp_eval.py", "nsp_text.py", "run.py")
ADAM = {"betas": (0.9, 0.999), "eps": 1e-8}      # defaults of the transformers Trainer, which the re-implementation used
MAX_GRAD_NORM = 1.0
SELECT = "highest validation 1:49 NDCG@10"       # the checkpoint of the result: the rule of the other models of the paper
OTHER = "lowest validation loss"                 # evaluated under 1:49 as well and kept as a record
FIRST_VERSION = "bert_nsp"                       # run name of the first version of this folder, whose result was OTHER


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["train", "full", "collect"])
    p.add_argument("--data-dir", default=paths.KC, help="output of preprocess/katchers.py (default: KATCHERS_DATA of paths.py)")
    p.add_argument("--split-dir", default=paths.KC_SPLIT, help="train/val/test_data.csv (default: KATCHERS_SPLIT of paths.py)")
    p.add_argument("--out-dir", default=None, help="default: <RESULTS>/proto_bert_nsp/kc/<name>")
    p.add_argument("--name", default="bert_nsp_ndcg", help="run name; give another one whenever a setting below is changed")
    p.add_argument("--seeds", type=int, nargs="+", default=list(paths.SEEDS))
    p.add_argument("--model", default=nsp_text.MODEL)
    p.add_argument("--tiny", action="store_true", help="a small BERT with random weights instead of the pretrained model (smoke test)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--earlier-run", default=None,
                   help="run folder of the first version of this prototype (its result was the epoch with the lowest validation loss). "
                        "A seed finished there for which both rules chose one epoch is taken over instead of being trained again. "
                        f"default: the folder {FIRST_VERSION} next to the output folder, if there is one; none = train every seed here")
    p.add_argument("--pair-seed", type=int, default=None,
                   help="seed of the negatives of the training and validation pairs (default: the seed of the run)")
    p.add_argument("--num-items", type=int, default=None, help="default: the size of item2idx.json, or N_ITEMS of paths.py without that file")
    p.add_argument("--eval-batch", type=int, default=128, help="pairs per forward pass in evaluation; lower it after an out-of-memory error")
    p.add_argument("--limit-users", type=int, default=None,
                   help="full: only the first N test users, written to files marked _first{N} (for trying the stage out)")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    a.out_dir = a.out_dir or os.path.join(paths.RES, "proto_bert_nsp", "kc", a.name)
    if a.earlier_run is None:
        beside = os.path.join(os.path.dirname(os.path.realpath(a.out_dir)), FIRST_VERSION)
        a.earlier_run = beside if os.path.isdir(beside) else "none"
    elif a.earlier_run != "none" and not os.path.isdir(a.earlier_run):
        p.error(f"--earlier-run: {a.earlier_run} is not a folder")
    a.earlier_run = None if a.earlier_run == "none" or os.path.realpath(a.earlier_run) == os.path.realpath(a.out_dir) \
        else os.path.realpath(a.earlier_run)
    return a


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stream_rng(seed, stream):
    """an independent generator per (seed, purpose), as in the re-implementation (so that --pair-seed 0 draws its negatives)"""
    return np.random.default_rng(np.random.SeedSequence([seed, int.from_bytes(stream.encode("utf-8"), "little") % (2**31)]))


def default_num_items(data_dir):
    """the catalogue size of paths.py, used only when the data folder has no item2idx.json"""
    return None if os.path.exists(os.path.join(data_dir, "item2idx.json")) else paths.N_ITEMS["kc"]


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- the task: users, sentences, pairs
class Task:
    def __init__(self, args):
        self.args = args
        self.ds = nsp_data.load(args.data_dir, args.num_items or default_num_items(args.data_dir))
        self.rows = nsp_data.load_rows(args.split_dir)
        diff = nsp_data.match(self.ds, self.rows)
        if diff:
            sys.exit("the split files and the data folder do not describe the same data:\n  " + "\n  ".join(diff))
        self.texts = nsp_text.Texts(nsp_text.load_tokenizer(args.model))
        self.cand_text = nsp_text.candidate_texts(self.rows)
        if len(self.cand_text) != self.ds.num_items:
            sys.exit(f"{len(self.cand_text)} candidate texts for {self.ds.num_items} items")
        self.cand_ids = [self.texts.ids(t) for t in self.cand_text]
        self.longest_of_catalogue = self.texts.longest(self.cand_text)
        self._manifest = {}

    def prompt(self, split, uid, items):
        """prompt of a user, cut against the longest text of the given candidate items -> (text, orders in it, orders of the user)"""
        orders = self.rows.users[split][uid].orders
        longest = self.longest_of_catalogue if items is None else self.texts.longest([self.cand_text[i] for i in items])
        text, n = nsp_text.history_prompt(orders, self.texts, longest)
        return text, n, len(orders)

    def manifest(self, split, catalogue=False):
        """{uid: prompt ids} of the validation or test users; the prompt is cut against the 50 candidates of the user, or against
        the whole catalogue. Also counts the prompts that do not hold every order, and those without any order."""
        key = (split, catalogue)
        if key not in self._manifest:
            part, ids, cut, bare = getattr(self.ds, split), {}, 0, 0
            for u in part.users:
                text, n, total = self.prompt(split, u, None if catalogue else part.cand[u])
                ids[u], cut, bare = self.texts.ids(text), cut + (n < total), bare + (n == 0)
            self._manifest[key] = (ids, {"users": len(ids), "prompts_without_all_orders": cut, "prompts_without_any_order": bare})
        return self._manifest[key]

    def candidate_pairs(self, split):
        """all (prompt, candidate) pairs of the 1:49 ranking of a split, user after user in the order of the split"""
        part, (prompts, _) = getattr(self.ds, split), self.manifest(split)
        return [(prompts[u], self.cand_ids[i]) for u in part.users for i in part.cand[u]]

    def training_pairs(self, split, pair_seed):
        """one positive and one negative pair per user of the split; the negative is drawn uniformly from all items.
        -> [(prompt ids, candidate ids)], labels (1 = the target), and two text pairs per user for check_pairs"""
        part, rng = getattr(self.ds, split), stream_rng(pair_seed, f"cb_pairs:{split}")
        pairs, labels, sample = [], [], []
        for u in self.rows.users[split]:                       # the order of the split file
            target, negative = part.target[u], int(rng.integers(0, self.ds.num_items))
            text, _, _ = self.prompt(split, u, [target, negative])
            p = self.texts.ids(text)
            pairs += [(p, self.cand_ids[target]), (p, self.cand_ids[negative])]
            labels += [1, 0]
            sample += [(text, self.cand_text[target]), (text, self.cand_text[negative])]
        return pairs, np.asarray(labels, dtype=np.int64), sample


# ---------------------------------------------------------------- model
def new_model(args, texts, pretrained=True):
    from transformers import BertConfig, BertForNextSentencePrediction
    if args.tiny:
        return BertForNextSentencePrediction(BertConfig(vocab_size=texts.tok.vocab_size, hidden_size=32, num_hidden_layers=2,
                                                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=nsp_text.MAX_LEN))
    if pretrained:
        return BertForNextSentencePrediction.from_pretrained(args.model)
    return BertForNextSentencePrediction(BertConfig.from_pretrained(args.model))     # weights follow from a checkpoint


def to_device(arrays, device):
    return {k: torch.from_numpy(v).to(device) for k, v in zip(("input_ids", "attention_mask", "token_type_ids"), arrays)}


@torch.no_grad()
def score_pairs(model, pairs, texts, device, batch, labels=None):
    """scores (logit[1] - logit[0]) of the pairs, in their order; with labels also the mean cross-entropy"""
    model.eval()
    order = np.argsort([-(len(p) + len(c)) for p, c in pairs], kind="stable")       # long pairs first: padding and memory
    scores, loss = np.empty(len(pairs), dtype=np.float32), 0.0
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        logits = model(**to_device(nsp_text.collate([pairs[i] for i in idx], texts), device)).logits.float()
        scores[idx] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        if labels is not None:
            loss += float(F.cross_entropy(logits, torch.from_numpy(labels[idx]).to(device), reduction="sum"))
        if device.type == "mps" and (s // batch + 1) % 10 == 0:
            torch.mps.empty_cache()               # Apple GPUs keep memory for every batch shape they have seen
    return (scores, loss / len(pairs)) if labels is not None else scores


def rank_candidates(model, task, split, device):
    """1:49 results of a split: per-user results, tied users"""
    part = getattr(task.ds, split)
    scores = score_pairs(model, task.candidate_pairs(split), task.texts, device, task.args.eval_batch)
    width = len(part.cand[part.users[0]])
    return nsp_eval.evaluate_candidates({u: scores[r * width:(r + 1) * width] for r, u in enumerate(part.users)}, part)


def optimizer_groups(model, weight_decay):
    """weight decay on everything except biases and LayerNorm parameters (the rule of the transformers Trainer)"""
    decay = [p for n, p in model.named_parameters() if p.requires_grad and not (n.endswith("bias") or "LayerNorm" in n)]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and (n.endswith("bias") or "LayerNorm" in n)]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]


def fit(task, model, args, seed, device, ckpt):
    """-> curve (one entry per epoch), epoch with the highest validation NDCG@10 (its weights are saved as ckpt), epoch with the
    lowest validation loss (saved as ckpt + ".by_val_loss")"""
    pair_seed = seed if args.pair_seed is None else args.pair_seed
    pairs, labels, sample = task.training_pairs("train", pair_seed)
    val_pairs, val_labels, val_sample = task.training_pairs("val", pair_seed)
    step = max(1, len(sample) // 200)
    checked = nsp_text.check_pairs(sample[::step] + val_sample[::step], task.texts)
    print(f"[{args.name}] seed={seed} train pairs {len(pairs)}, validation pairs {len(val_pairs)}; "
          f"{checked} assembled pairs equal the tokenizer's encoding", flush=True)

    opt = torch.optim.AdamW(optimizer_groups(model, args.weight_decay), lr=args.lr, **ADAM)
    steps_per_epoch = math.ceil(len(pairs) / args.batch_size)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, (total - s) / total))        # linear decay, no warm-up
    gen = torch.Generator().manual_seed(seed)
    curve, best_loss, best_ndcg, t0 = [], (float("inf"), None), (float("-inf"), None), time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order, running = torch.randperm(len(pairs), generator=gen).numpy(), 0.0
        for b in range(steps_per_epoch):
            idx = order[b * args.batch_size:(b + 1) * args.batch_size]
            out = model(**to_device(nsp_text.collate([pairs[i] for i in idx], task.texts), device),
                        labels=torch.from_numpy(labels[idx]).to(device))
            if not bool(torch.isfinite(out.loss)):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {b + 1}")
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += float(out.loss.detach())
            if device.type == "mps" and (b + 1) % 10 == 0:
                torch.mps.empty_cache()           # Apple GPUs keep memory for every batch shape they have seen
            if (b + 1) % 200 == 0:
                print(f"[{args.name}] seed={seed} ep={epoch} step {b + 1}/{steps_per_epoch} loss={running / (b + 1):.4f} "
                      f"{(time.time() - t0) / 60:.1f}m", flush=True)
        _, val_loss = score_pairs(model, val_pairs, task.texts, device, args.eval_batch, val_labels)
        entry = {"epoch": epoch, "train_loss": running / steps_per_epoch, "val_loss": val_loss}
        marks = ""
        if val_loss < best_loss[0]:
            best_loss, marks = (val_loss, epoch), " *loss"
            torch.save(model.state_dict(), ckpt + ".by_val_loss")          # its own file: the two rules may choose different epochs
        per_user, _ = rank_candidates(model, task, "val", device)
        entry["val_ndcg@10"] = float(np.mean([r["ndcg@10"] for r in per_user["1to49"].values()]))
        if entry["val_ndcg@10"] > best_ndcg[0]:
            best_ndcg, marks = (entry["val_ndcg@10"], epoch), marks + " *ndcg"
            torch.save(model.state_dict(), ckpt)
        curve.append(entry)
        print(f"[{args.name}] seed={seed} ep={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in entry.items() if k != "epoch")
              + f"{marks} {(time.time() - t0) / 60:.1f}m", flush=True)
    return curve, best_ndcg[1], best_loss[1]


def test_1to49(task, model, device):
    per_user, tied = rank_candidates(model, task, "test", device)
    return {"test_1to49": nsp_eval.summarize_1to49(per_user["1to49"]), "tied_users_1to49": tied}, per_user["1to49"]


def run_seed(task, args, seed, out, device):
    t0 = time.time()
    set_seed(seed)
    model = new_model(args, task.texts).to(device)
    ckpt = os.path.join(out, "checkpoints", f"best_seed{seed}.pt")
    curve, epoch_ndcg, epoch_loss = fit(task, model, args, seed, device, ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    row, p49 = test_1to49(task, model, device)
    at = {e["epoch"]: e for e in curve}
    row = {"seed": seed, "checkpoint": ckpt, "selected_by": "val_ndcg@10", "best_epoch": epoch_ndcg,
           "best_val_ndcg@10": at[epoch_ndcg]["val_ndcg@10"], "best_val_loss": at[epoch_ndcg]["val_loss"], **row}
    per_user = {"1to49": p49}
    other = {"best_epoch": epoch_loss, "best_val_loss": at[epoch_loss]["val_loss"], "same_checkpoint": epoch_loss == epoch_ndcg}
    if epoch_loss == epoch_ndcg:
        other.update({k: row[k] for k in ("test_1to49", "tied_users_1to49")})
        per_user["1to49_by_val_loss"] = p49
        os.remove(ckpt + ".by_val_loss")                    # the same weights as the checkpoint of the result
    else:
        model.load_state_dict(torch.load(ckpt + ".by_val_loss", map_location=device))
        r, per_user["1to49_by_val_loss"] = test_1to49(task, model, device)
        other.update({**r, "checkpoint": ckpt + ".by_val_loss"})       # kept, so that the other rule never needs a new training run
    row["by_val_loss"] = other
    row.update({"minutes": (time.time() - t0) / 60, "val_curve": curve})
    return row, per_user


# ---------------------------------------------------------------- stage 2: the whole catalogue
def rank_catalogue(task, args, seed, out, device, config, ckpt):
    """Scores of all items for every test user, written user by user into scores_full_seed{S}.npy; an interrupted run continues
    with the users that are still missing. The prompt is cut against the longest candidate text of the whole catalogue."""
    part, n = task.ds.test, task.ds.num_items
    prompts, counts = task.manifest("test", catalogue=True)
    users = part.users[:args.limit_users] if args.limit_users else part.users
    mark = f"_first{args.limit_users}" if args.limit_users else ""
    path = os.path.join(out, f"scores_full_seed{seed}{mark}.npy")
    key = {"users": users, "config": config, "checkpoint_sha256": sha256_of(ckpt)}
    if os.path.exists(path) and os.path.exists(path + ".json"):
        if json.load(open(path + ".json")) != key:
            sys.exit(f"{path} was started with other settings, data, code or checkpoint; remove it and its .json to start again")
        scores = np.load(path, mmap_mode="r+")
    else:
        scores = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(users), n))
        scores[:] = np.nan
        scores.flush()
        json.dump(key, open(path + ".json", "w"))
    todo = [r for r in range(len(users)) if np.isnan(scores[r]).any()]
    print(f"[{args.name}] seed={seed} full catalogue: {len(users) - len(todo)} of {len(users)} users done before", flush=True)
    model = new_model(args, task.texts, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)
    t0 = time.time()
    for k, r in enumerate(todo):
        p = prompts[users[r]]
        scores[r] = score_pairs(model, [(p, c) for c in task.cand_ids], task.texts, device, args.eval_batch)
        if (k + 1) % 20 == 0 or k + 1 == len(todo):
            scores.flush()
            m = (time.time() - t0) / 60
            print(f"[{args.name}] seed={seed} full catalogue {k + 1}/{len(todo)} users {m:.1f}m, about {m / (k + 1) * (len(todo) - k - 1):.0f}m left", flush=True)
    sub = part if not args.limit_users else subset(part, users)
    per_user, tied = nsp_eval.evaluate_full({u: np.asarray(scores[r]) for r, u in enumerate(users)}, sub, n)
    row = {"seed": seed, **nsp_eval.summarize_full(per_user["full"]),
           "1to49_from_full": nsp_eval.summarize_1to49(per_user["1to49"]), "tied_users_1to49_from_full": tied,
           "prompts_full": counts, "minutes_full": (time.time() - t0) / 60}
    return row, per_user, mark


def subset(part, users):
    s = nsp_data.Split()
    for u in users:
        s.add(u, part.orders[u], part.target[u], part.cand[u][1:], part.seen[u])
    return s


# ---------------------------------------------------------------- seeds of the first version of this folder
def source(folder, name, config):
    """where a result that is taken over, or compared with, comes from"""
    return {"folder": folder, "file": name, "file_sha256": sha256_of(os.path.join(folder, name)),
            "code_sha256": config["code_sha256"], "select": config["select"]}


def earlier_result(args, seed, config):
    """-> (result_seed{S}.json of the earlier run, True if the seed can be taken over), or (None, False) if that run has no result
    for the seed. The earlier run kept the epoch with the lowest validation loss and evaluated the epoch with the highest
    validation NDCG@10 under 1:49 as well; where both are one epoch, its checkpoint and its results are those of this version.
    Its data, sentences, evaluation and settings must be those of this run: the configs may differ in the name of the run, in
    the rule and in run.py, and in nothing else. A result that fails this stops the run, so that no seed is trained twice
    unnoticed."""
    path = os.path.join(args.earlier_run, f"result_seed{seed}.json") if args.earlier_run else None
    if not path or not os.path.exists(path):
        return None, False
    old = json.load(open(path))
    c, r = old["config"], old["row"]
    if c.get("select") != OTHER or c.get("also_evaluated") != SELECT or "by_val_ndcg" not in r:
        sys.exit(f"{path} is not a result of the first version of this prototype; --earlier-run none trains seed {seed} here")
    differ = sorted(k for k in set(c) | set(config) if k not in ("name", "select", "also_evaluated", "code_sha256") and c.get(k) != config.get(k))
    differ += [f"code_sha256 of {f}" for f in CODE if f != "run.py" and c["code_sha256"].get(f) != config["code_sha256"].get(f)]
    if differ:
        sys.exit(f"{path} was written with other data, settings or code ({', '.join(differ)}); --earlier-run none trains seed {seed} here")
    return old, bool(r["by_val_ndcg"]["same_checkpoint"])


def take_over(args, seed, out, old):
    """stage 1 of a seed for which the earlier run chose one epoch by both rules -> row; writes per_user_seed{S}.json"""
    src, r = args.earlier_run, old["row"]
    per_user = json.load(open(os.path.join(src, f"per_user_seed{seed}.json")))
    epoch = r["by_val_ndcg"]["best_epoch"]
    if epoch != r["best_epoch"] or r["by_val_ndcg"]["test_1to49"] != r["test_1to49"] or per_user["1to49_by_val_ndcg"] != per_user["1to49"]:
        sys.exit(f"seed {seed} of {src}: marked as one checkpoint for both rules, but the results of the two rules differ")
    at = {e["epoch"]: e for e in r["val_curve"]}
    both = {k: r[k] for k in ("test_1to49", "tied_users_1to49")}
    json.dump({"1to49": per_user["1to49"], "1to49_by_val_loss": per_user["1to49"]}, open(os.path.join(out, f"per_user_seed{seed}.json"), "w"))
    return {"seed": seed, "checkpoint": os.path.join(src, "checkpoints", f"best_seed{seed}.pt"), "selected_by": "val_ndcg@10",
            "best_epoch": epoch, "best_val_ndcg@10": at[epoch]["val_ndcg@10"], "best_val_loss": at[epoch]["val_loss"], **both,
            "by_val_loss": {"best_epoch": epoch, "best_val_loss": at[epoch]["val_loss"], "same_checkpoint": True, **both},
            "minutes": r["minutes"], "val_curve": r["val_curve"],
            "taken_over_from": source(src, f"result_seed{seed}.json", old["config"])}


def take_over_full(seed, out, row):
    """stage 2 of a seed taken over in stage 1, if the earlier run finished it -> row of full_seed{S}.json; None if it did not"""
    origin = row["taken_over_from"]
    src, name = origin["folder"], f"full_seed{seed}.json"
    if not os.path.exists(os.path.join(src, name)):
        return None
    old = json.load(open(os.path.join(src, name)))
    result = os.path.join(src, f"result_seed{seed}.json")
    per_user = json.load(open(os.path.join(src, f"per_user_seed{seed}.json")))
    if sha256_of(result) != origin["file_sha256"] or old["config"] != json.load(open(result))["config"] or "full" not in per_user:
        sys.exit(f"seed {seed} of {src}: stage 2 there does not belong to the stage 1 that was taken over")
    path = os.path.join(out, f"per_user_seed{seed}.json")
    merged = json.load(open(path))
    merged.update({"full": per_user["full"], "1to49_from_full": per_user["1to49_from_full"]})
    json.dump(merged, open(path, "w"))
    return {**old["row"], "taken_over_full_from": source(src, name, old["config"])}


def checkpoint_of(row, out, seed):
    """the checkpoint of a result: where the row says, or in this folder if the folder was moved since"""
    return row["checkpoint"] if os.path.exists(row["checkpoint"]) else os.path.join(out, "checkpoints", f"best_seed{seed}.pt")


# ---------------------------------------------------------------- bookkeeping
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_config(task, args):
    _, test_counts = task.manifest("test")
    return {
        "name": args.name, "model": "tiny random BERT" if args.tiny else args.model, "max_len": nsp_text.MAX_LEN,
        "prompt": "every order line, oldest first, cut at 512 tokens", "candidates": "one text per product, without attributes",
        "pairs": "1 positive and 1 uniform negative per user", "pair_seed": "seed of the run" if args.pair_seed is None else args.pair_seed,
        "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "optimizer": f"AdamW {ADAM}, linear decay to 0 without warm-up, gradient norm clipped at {MAX_GRAD_NORM}",
        "select": SELECT, "also_evaluated": OTHER,
        "data": {**nsp_data.describe(task.ds, task.rows), "test_prompts_1to49": test_counts},
        "inputs_sha256": {**task.ds.sha256, **task.rows.sha256},
        "code_sha256": {f: sha256_of(os.path.join(HERE, f)) for f in CODE},
    }


def mean_std(rows, pick):
    v = np.array([pick(r) for r in rows], dtype=float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]


def collect(args, out, config=None):
    """full_eval.json from result_seed{S}.json (stage 1) and full_seed{S}.json (stage 2) of every seed found in the folder, so that
    seeds run by separate calls (e.g. on two GPUs) end up in one file"""
    rows = []
    for seed in sorted(int(m.group(1)) for m in map(re.compile(r"result_seed(\d+)\.json$").match, os.listdir(out)) if m):
        r = os.path.join(out, f"result_seed{seed}.json")
        saved = json.load(open(r))
        if saved["config"].get("select") != SELECT:
            print(f"{r} belongs to another version of this prototype: nothing collected", flush=True)
            return None
        config = config or saved["config"]
        row, f = saved["row"], os.path.join(out, f"full_seed{seed}.json")
        if os.path.exists(f):
            row = {**row, **{k: v for k, v in json.load(open(f))["row"].items() if k != "seed"}}
        rows.append(row)
    if not rows:
        print("no finished seed", flush=True)
        return None
    agg = {f"1to49_ndcg@{k}": mean_std(rows, lambda r, k=k: r["test_1to49"]["overall"][f"ndcg@{k}"]) for k in (10, 5)}
    full = [r for r in rows if "full_masked_ndcg@10_overall" in r]
    if full:
        agg.update({k: mean_std(full, lambda r, k=k: r[k]) for k in full[0] if k.startswith("full_") and isinstance(full[0][k], float)})
    final = {"script": "proto_bert_nsp/run.py", "kind": args.name, "seeds": [r["seed"] for r in rows],
             "seeds_full": [r["seed"] for r in full], "config": config, "per_seed": rows, "aggregate_mean_std": agg}
    final["aggregate_mean_std_by_val_loss"] = {
        f"1to49_ndcg@{k}": mean_std(rows, lambda r, k=k: r["by_val_loss"]["test_1to49"]["overall"][f"ndcg@{k}"]) for k in (10, 5)}
    final["seeds_where_both_rules_pick_one_epoch"] = [r["seed"] for r in rows if r["by_val_loss"]["same_checkpoint"]]
    final["seeds_taken_over"] = [r["seed"] for r in rows if "taken_over_from" in r]                 # stage 1 of the earlier run
    final["seeds_full_taken_over"] = [r["seed"] for r in full if "taken_over_full_from" in r]       # stage 2 of the earlier run
    json.dump(final, open(os.path.join(out, "full_eval.json"), "w"), indent=1)
    print("AGG", json.dumps({k: [round(x, 4) for x in v] for k, v in agg.items()}), flush=True)
    return final


def main():
    args = parse_args()
    out = os.path.realpath(args.out_dir)
    if args.stage == "collect":
        if os.path.isdir(out):
            collect(args, out)
        return
    for folder in (os.path.realpath(args.data_dir), os.path.realpath(args.split_dir)):
        if os.path.commonpath([out, folder]) == folder:
            sys.exit("--out-dir is inside a data folder; the data folders are read-only for this prototype")
    device = pick_device(args.device)
    task = Task(args)
    config = make_config(task, args)
    import transformers
    print(json.dumps({**config, "stage": args.stage, "device": str(device), "torch": torch.__version__,
                      "transformers": transformers.__version__, "data_dir": os.path.realpath(args.data_dir),
                      "split_dir": os.path.realpath(args.split_dir), "out_dir": out}, indent=1, ensure_ascii=False), flush=True)
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)

    for seed in args.seeds:
        result = os.path.join(out, f"result_seed{seed}.json")
        saved = json.load(open(result)) if os.path.exists(result) else None
        if saved and saved["config"] != config:
            sys.exit(f"{result} was written with other settings, data or code; use another --name or remove the folder")
        if args.stage == "train":
            if saved:
                print(f"seed {seed}: finished before, reused", flush=True)
                row = saved["row"]
            else:
                old, same_epoch = earlier_result(args, seed, config)
                if same_epoch:
                    row = take_over(args, seed, out, old)
                    print(f"seed {seed}: taken over from {args.earlier_run} (both rules chose epoch {row['best_epoch']} there)", flush=True)
                else:
                    row, per_user = run_seed(task, args, seed, out, device)
                    json.dump(per_user, open(os.path.join(out, f"per_user_seed{seed}.json"), "w"))
                    if old is not None:               # trained there as well, but the rules chose different epochs: kept for comparison
                        e = old["row"]["by_val_ndcg"]
                        row["earlier_run"] = {**source(args.earlier_run, f"result_seed{seed}.json", old["config"]), "best_epoch": e["best_epoch"],
                                              "ndcg@10": e["test_1to49"]["overall"]["ndcg@10"], "val_curve": old["row"]["val_curve"]}
                json.dump({"config": config, "row": row}, open(result, "w"), indent=1)
            t, o = row["test_1to49"]["overall"], row["by_val_loss"]
            print(f"[{args.name} seed {seed}] epoch {row['best_epoch']} (val NDCG@10 {row['best_val_ndcg@10']:.4f}) | 1:49 NDCG@10 {t['ndcg@10']:.4f} "
                  f"(n={t['n']}, tied {row['tied_users_1to49']}) | by val loss: epoch {o['best_epoch']}, 1:49 NDCG@10 "
                  f"{o['test_1to49']['overall']['ndcg@10']:.4f} | {row['minutes']:.1f} min" + (" | taken over" if "taken_over_from" in row else ""), flush=True)
            if "earlier_run" in row:
                e = row["earlier_run"]
                print(f"[{args.name} seed {seed}] the earlier run chose epoch {e['best_epoch']} by val NDCG@10, 1:49 NDCG@10 {e['ndcg@10']:.4f}; "
                      f"this run differs by {t['ndcg@10'] - e['ndcg@10']:+.4f}", flush=True)
        else:
            if not saved:
                print(f"seed {seed}: stage 1 is not finished, skipped", flush=True)
                continue
            target = os.path.join(out, f"full_seed{seed}.json")
            if os.path.exists(target) and not args.limit_users:
                print(f"seed {seed}: full catalogue ranked before, reused", flush=True)
                row = json.load(open(target))["row"]
            elif "taken_over_from" in saved["row"] and not args.limit_users and (row := take_over_full(seed, out, saved["row"])) is not None:
                json.dump({"config": config, "row": row}, open(target, "w"), indent=1)
                print(f"seed {seed}: full catalogue taken over from {row['taken_over_full_from']['folder']}", flush=True)
            else:
                ckpt = checkpoint_of(saved["row"], out, seed)
                if not os.path.exists(ckpt):
                    print(f"seed {seed}: no checkpoint at {ckpt}, skipped", flush=True)
                    continue
                row, per_user, mark = rank_catalogue(task, args, seed, out, device, config, ckpt)
                if mark:
                    json.dump({"config": config, "row": row, "per_user": per_user}, open(os.path.join(out, f"full_seed{seed}{mark}.json"), "w"), indent=1)
                else:
                    pu_path = os.path.join(out, f"per_user_seed{seed}.json")
                    merged = json.load(open(pu_path))
                    merged.update({"full": per_user["full"], "1to49_from_full": per_user["1to49"]})
                    json.dump(merged, open(pu_path, "w"))
                    json.dump({"config": config, "row": row}, open(target, "w"), indent=1)
            print(f"[{args.name} seed {seed}] full masked NDCG@10 {row['full_masked_ndcg@10_overall']:.4f} (novel "
                  f"{row['full_masked_ndcg@10_novel']:.4f}, repeat {row['full_masked_ndcg@10_repeat']:.4f}) | unmasked "
                  f"{row['full_unmasked_ndcg@10_overall']:.4f} | 1:49 from the same scores {row['1to49_from_full']['overall']['ndcg@10']:.4f} "
                  f"| n={row['n']} | {row['minutes_full']:.1f} min", flush=True)
        if not args.limit_users:
            collect(args, out, config)


if __name__ == "__main__":
    main()
