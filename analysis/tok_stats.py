"""Tokenisation of the Instacart item sentences under the two text encoders (tokenizer only): tokens per sentence and share of words kept
whole. -> <EXTRA_ENC>/tok_stats.json"""
import json, numpy as np
from transformers import AutoTokenizer
from common import *
m = json.load(open(IC + 'item_meta.json')); T = [m[str(i)]['text'] for i in range(len(m))]; OUT = {}
for tag, name in (('KR-ELECTRA', 'snunlp/KR-ELECTRA-discriminator'), ('ELECTRA-base (en)', 'google/electra-base-discriminator')):
    tok = AutoTokenizer.from_pretrained(name); nt = np.array([len(tok(t)['input_ids']) for t in T])
    words = [w for t in T for w in t.replace(' - ', ' ').split() if any(c.isalpha() for c in w)]; pc = {w: len(tok.tokenize(w)) for w in sorted(set(w.lower() for w in words))}
    per = np.array([pc[w.lower()] for w in words])
    OUT[tag] = dict(model=name, tokens_mean=float(nt.mean()), tokens_max=int(nt.max()), pieces_per_word=float(per.mean()), words_whole_share=float((per == 1).mean()))
    print(tag, OUT[tag])
json.dump(OUT, open(EXTRA_ENC + 'tok_stats.json', 'w'), ensure_ascii=False, indent=1)
