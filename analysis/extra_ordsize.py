"""Number of test users by the mean number of items per order of the input history. -> <EXTRA>/ordsize.json"""
import json, numpy as np
from common import *
OUT = {}
for DS in ('kc', 'ic'):
    hist, target, trh = data(DS); m = np.array([np.mean([len(o) for o in h]) for h in hist.values()])
    OUT[DS] = dict(n=int(len(m)), single=int((m < 1.0001).sum()), below15=int((m < 1.5).sum()), atleast15=int((m >= 1.5).sum()))
json.dump(OUT, open(EXTRA + 'ordsize.json', 'w'), indent=0); print(OUT)
