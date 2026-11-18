"""Macros and a summary of the BERT-NSP runs, so that no number has to be typed by hand. Changes none of the files of the paper:
it writes <RESULTS>/proto_bert_nsp/numbers_bert_nsp.tex (new macros only) and summary.json next to it.

The macros follow tables/make_tables.py, with the same formulas:
  \\kcbaseBERTNSPA / F / Afive / Ffive (+ ...m)   mean +- std over the five seeds (mean only): 1:49 and full catalogue (masked),
                                                 NDCG@10 and NDCG@5, as for the other baselines
  \\kcbaseBERTNSPAnovel / Arepeat / Fnovel / Frepeat   means over the seeds for targets outside / inside the history
  \\kcdvsBERTNSPA / F, \\kcsigvsBERTNSPA / F, \\kccivsBERTNSPA / F   Text+Graph minus BERT-NSP on seed-averaged per-user NDCG@10: mean,
                                                 significance, bootstrap 95% interval (paired() of make_tables.py). Needs the
                                                 per-user dumps of the paper's model (<RESULTS>/analysis/pu/); [TODO] without them
  \\kcBERTNSPcutusers / cutshare                  test users whose prompt does not hold every order (512 tokens), and their share
Anything that is not complete for all five seeds is written as [TODO], as in make_tables.py.
The second line of the file names the rule by which the checkpoints were chosen and says which seeds were taken over from the run of
the first version of this folder (run.py); summary.json has the same and the 1:49 result of the other rule.

usage: python proto_bert_nsp/make_numbers.py [--name bert_nsp_ndcg] [--results <RESULTS>/proto_bert_nsp]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--name", default="bert_nsp_ndcg")
p.add_argument("--results", default=os.path.join(paths.RES, "proto_bert_nsp"))
p.add_argument("--paper-pu", default=paths.PU, help="per-user dumps of the paper's models")
p.add_argument("--paper-run", default="text_co_lgcn_late", help="name of the Text+Graph run in the per-user dumps")
A = p.parse_args()
run_dir = os.path.join(A.results, "kc", A.name)
SEEDS = list(paths.SEEDS)
TODO = "[TODO]"
macros, summary = {}, {"run": run_dir, "seeds": SEEDS}


def origin(rows, config):
    """one line on how the checkpoints were chosen and where the seeds come from"""
    taken = [s for s in sorted(rows) if "taken_over_from" in rows[s]]
    here = [s for s in sorted(rows) if "taken_over_from" not in rows[s]]
    text = f"checkpoint: {config.get('select')}; run {A.name}"
    if taken:
        earlier = rows[taken[0]]["taken_over_from"]
        text += (f"; seed{'s' * (len(taken) > 1)} {', '.join(map(str, taken))} taken over from the run {os.path.basename(earlier['folder'])} "
                 f"(run.py {earlier['code_sha256']['run.py'][:12]}), where both rules chose one epoch")
        text += f"; seed{'s' * (len(here) > 1)} {', '.join(map(str, here))} trained by run.py {config['code_sha256']['run.py'][:12]}" if here else ""
    return text

final = json.load(open(os.path.join(run_dir, "full_eval.json"))) if os.path.exists(os.path.join(run_dir, "full_eval.json")) else None
rows = {r["seed"]: r for r in final["per_seed"]} if final else {}
summary["origin"] = origin(rows, final["config"]) if final else None
summary["select"] = final["config"].get("select") if final else None
summary["seeds_taken_over"] = {str(s): rows[s]["taken_over_from"] for s in sorted(rows) if "taken_over_from" in rows[s]}
summary["seeds_full_taken_over"] = {str(s): rows[s]["taken_over_full_from"] for s in sorted(rows) if "taken_over_full_from" in rows[s]}
have_149 = all(s in rows for s in SEEDS)
have_full = have_149 and all("full_masked_ndcg@10_overall" in rows[s] for s in SEEDS)
summary.update({"seeds_with_1to49": sorted(s for s in SEEDS if s in rows),
                "seeds_with_full": sorted(s for s in SEEDS if s in rows and "full_masked_ndcg@10_overall" in rows[s])})


def pm(values, with_pm=True):
    v = np.array(values, dtype=float)
    return f"{v.mean():.4f} $\\pm$ {v.std(ddof=1):.4f}" if with_pm else f"{v.mean():.4f}"


def both(name, ok, pick):
    values = [pick(rows[s]) for s in SEEDS] if ok else None
    macros[name], macros[name + "m"] = (pm(values), pm(values, False)) if ok else (TODO, TODO)
    summary[name] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "per_seed": values} if ok else None


both("kcbaseBERTNSPA", have_149, lambda r: r["test_1to49"]["overall"]["ndcg@10"])
both("kcbaseBERTNSPAfive", have_149, lambda r: r["test_1to49"]["overall"]["ndcg@5"])
both("kcbaseBERTNSPF", have_full, lambda r: r["full_masked_ndcg@10_overall"])
both("kcbaseBERTNSPFfive", have_full, lambda r: r["full_masked_ndcg@5_overall"])
for g in ("novel", "repeat"):
    macros[f"kcbaseBERTNSPA{g}"] = pm([rows[s]["test_1to49"][g]["ndcg@10"] for s in SEEDS], False) if have_149 else TODO
    macros[f"kcbaseBERTNSPF{g}"] = pm([rows[s][f"full_masked_ndcg@10_{g}"] for s in SEEDS], False) if have_full else TODO
if have_149:
    counts = final["config"]["data"]["test_prompts_1to49"]
    macros["kcBERTNSPcutusers"] = f"{counts['prompts_without_all_orders']:,}"
    macros["kcBERTNSPcutshare"] = f"{100 * counts['prompts_without_all_orders'] / counts['users']:.1f}\\%"
    summary["test_prompts_1to49"] = counts
    summary["tied_users_1to49"] = [rows[s]["tied_users_1to49"] for s in SEEDS]
    summary["selected_epochs"] = [rows[s]["best_epoch"] for s in SEEDS]
    for key, rule in (("by_val_loss", "lowest_validation_loss"), ("by_val_ndcg", "highest_validation_ndcg")):     # the other rule
        if all(key in rows[s] for s in SEEDS):
            v = [rows[s][key]["test_1to49"]["overall"]["ndcg@10"] for s in SEEDS]
            summary[f"1to49_ndcg@10_of_the_checkpoint_with_the_{rule}"] = {
                "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "per_seed": v, "epochs": [rows[s][key]["best_epoch"] for s in SEEDS]}
    summary["again_trained_seeds_against_the_earlier_run"] = {
        str(s): {"earlier_epoch": rows[s]["earlier_run"]["best_epoch"], "earlier_ndcg@10": rows[s]["earlier_run"]["ndcg@10"],
                 "epoch": rows[s]["best_epoch"], "ndcg@10": rows[s]["test_1to49"]["overall"]["ndcg@10"]} for s in SEEDS if "earlier_run" in rows[s]}
if have_full:
    v = [rows[s]["1to49_from_full"]["overall"]["ndcg@10"] for s in SEEDS]
    summary["1to49_ndcg@10_from_the_scores_of_all_items"] = {"mean": float(np.mean(v)), "per_seed": v}


# ---------------------------------------------------------------- Text+Graph minus BERT-NSP, as paired() of tables/make_tables.py
def per_user(proto):
    """-> (BERT-NSP, Text+Graph): per seed {user: NDCG@10}, or None where files are missing"""
    mine, theirs = [], []
    for s in SEEDS:
        a = os.path.join(run_dir, f"per_user_seed{s}.json")
        b = f"{A.paper_pu}kc_{A.paper_run}_{proto}_seed{s}.json"
        if not os.path.exists(a) or not os.path.exists(b):
            return None
        d = json.load(open(a))
        if proto == "full" and "full" not in d:
            return None
        src, key = (d["1to49"], "ndcg@10") if proto == "1to49" else (d["full"], "masked@10")
        mine.append({u: x[key] for u, x in src.items()})
        theirs.append({u: x["n10"] for u, x in json.load(open(b)).items()})
    return mine, theirs


def paired(proto):
    got = per_user(proto)
    if got is None:
        return None
    X, Y = got
    users = sorted(set.intersection(*[set(x) for x in X], *[set(y) for y in Y]))
    MA = np.array([[x[u] for u in users] for x in X])
    MB = np.array([[y[u] for u in users] for y in Y])
    d = MB.mean(0) - MA.mean(0)
    rng = np.random.default_rng(paths.BOOT_SEED)
    n = len(d)
    boot = d[rng.integers(0, n, (paths.BOOT_N, n))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nz = d[np.abs(d) > 1e-15]
    pv = float(wilcoxon(nz).pvalue)
    return dict(n=n, delta=float(d.mean()), lo=float(lo), hi=float(hi), p=pv, seeds_pos=int(((MB - MA).mean(1) > 0).sum()),
                sig=bool((lo > 0 or hi < 0) and pv < 0.05))


for proto, P_ in (("1to49", "A"), ("full", "F")):
    r = paired(proto) if (have_149 if proto == "1to49" else have_full) else None
    macros[f"kcdvsBERTNSP{P_}"] = TODO if r is None else f"{r['delta']:+.4f}"
    macros[f"kcsigvsBERTNSP{P_}"] = TODO if r is None else ("yes" if r["sig"] else "no")
    macros[f"kccivsBERTNSP{P_}"] = TODO if r is None else f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"
    summary[f"text_graph_minus_bert_nsp_{proto}"] = r

os.makedirs(A.results, exist_ok=True)
with open(os.path.join(A.results, "numbers_bert_nsp.tex"), "w") as f:
    f.write("% BERT-NSP (Park and Oh, 2024) retrained in the setting of the paper: proto_bert_nsp/make_numbers.py\n")
    f.write(f"% {summary['origin'] or 'no finished seed'}\n")
    f.writelines(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
json.dump(summary, open(os.path.join(A.results, "summary.json"), "w"), indent=1)
for k, v in macros.items():
    print(f"{k:28s} {v}")
missing = [k for k, v in macros.items() if v == TODO]
print(f"{len(macros) - len(missing)} macros written, {len(missing)} still [TODO]"
      + (": " + ", ".join(missing) if missing else "") + f" -> {os.path.join(A.results, 'numbers_bert_nsp.tex')}")
