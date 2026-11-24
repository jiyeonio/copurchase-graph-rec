"""Katcher's: how many of the 49 validation negatives are items of the user's own history.
The validation candidates are built as in model/model_katchers.py (option --val-exclude-history off, as in the paper): users of val_data.json
with a non-empty history (last MAX_ORDERS orders) and a valid first target item; 49 negatives drawn from all items except the target with a
generator seeded by VAL_NEGATIVE_SEED + crc32(user id). History = every item of the user's val_history.json (the items removed under
full-catalogue ranking); the count over the model input (last MAX_ORDERS orders) is given as well. CPU only. -> <AN>/val_neg_history.json"""
import hashlib, json, os, sys, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import KC, AN, N_ITEMS

VAL_NEGATIVE_SEED, MAX_ORDERS = 42, 50          # as in model/model_katchers.py and model/core.py
NUM_ITEMS = N_ITEMS['kc']
val_data = json.load(open(KC + 'val_data.json', encoding='utf-8'))
val_history = json.load(open(KC + 'val_history.json', encoding='utf-8'))
key = lambda e: (e.get('day', 0), str(e.get('order_id', '')))


def window_items(events):                      # items of the model input (last MAX_ORDERS orders); None if empty
    s = set()
    for e in sorted(events, key=key)[-MAX_ORDERS:]:
        s |= {int(i) for i in e.get('items', []) if 0 <= int(i) < NUM_ITEMS}
    return s or None


def val_negatives(uid, target, n=49):          # as model/model_katchers.py with the option off
    pool = np.array([i for i in range(NUM_ITEMS) if i != target], dtype=np.int64)
    rng = np.random.default_rng((VAL_NEGATIVE_SEED + zlib.crc32(str(uid).encode('utf-8'))) % (2**32))
    return rng.choice(pool, size=n, replace=False).tolist()


cache = {str(u): w for u, ev in val_history.items() if (w := window_items(ev)) is not None}
cand = {}
for uid, ev in val_data.items():
    uid = str(uid); ti = ev.get('items', [])
    if uid not in cache or not ti or not (0 <= int(ti[0]) < NUM_ITEMS):
        continue
    cand[uid] = [int(ti[0]), *val_negatives(uid, int(ti[0]), 49)]
seen = {u: {int(i) for e in val_history.get(u, []) for i in e.get('items', []) if 0 <= int(i) < NUM_ITEMS} for u in cand}


def summary(hist):
    k = np.array([sum(i in hist[u] for i in c[1:]) for u, c in cand.items()])
    pos = k[k > 0]
    return dict(users=len(k), users_with_any=int((k > 0).sum()), share_with_any=float((k > 0).mean()),
                mean_per_user=float(k.mean()), mean_per_user_among_with_any=float(pos.mean()) if len(pos) else None,
                max_per_user=int(k.max()), distribution={int(v): int((k == v).sum()) for v in np.unique(k)})


out = dict(num_items=NUM_ITEMS, val_negative_seed=VAL_NEGATIVE_SEED,
           candidates_sha256=hashlib.sha256(json.dumps(cand, sort_keys=True).encode()).hexdigest(),
           target_in_history=int(sum(c[0] in seen[u] for u, c in cand.items())),
           history_all=summary(seen), history_model_window=summary(cache),
           mean_history_size=float(np.mean([len(seen[u]) for u in cand])),
           expected_mean_if_uniform=float(np.mean([49 * len(seen[u] - {c[0]}) / (NUM_ITEMS - 1) for u, c in cand.items()])))
json.dump(out, open(AN + 'val_neg_history.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
