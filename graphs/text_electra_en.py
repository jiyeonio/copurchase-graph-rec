"""Appendix comparison experiment only: Instacart item text embeddings with google/electra-base-discriminator ([CLS], max_length 64, same
input sentences as the main embedding). -> IC_TEXT_EN of paths.py"""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import IC, IC_TEXT_EN
from transformers import AutoModel, AutoTokenizer
O = os.path.dirname(IC_TEXT_EN) + '/'; os.makedirs(O, exist_ok=True); assert not os.path.exists(IC_TEXT_EN)
m = json.load(open(IC + 'item_meta.json')); T = [m[str(i)]['text'] for i in range(len(m))]
name = 'google/electra-base-discriminator'; tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).cuda().eval(); e = []
for s in range(0, len(T), 256):
    x = tok(T[s:s + 256], padding=True, truncation=True, max_length=64, return_tensors='pt').to('cuda')
    with torch.no_grad(): e.append(model(**x).last_hidden_state[:, 0].cpu().numpy())
e = np.vstack(e).astype(np.float32); assert e.shape == (len(T), 768) and np.isfinite(e).all(); np.save(IC_TEXT_EN, e)
n = e / np.linalg.norm(e, axis=1, keepdims=True); S = n @ n.T; iu = np.triu_indices(len(e), 1)
info = dict(model=name, pool='cls', max_length=64, n=len(T), norm=float(np.linalg.norm(e, axis=1).mean()), offdiag_cos_mean=float(S[iu].mean())); json.dump(info, open(O + 'info.json', 'w'), indent=1); print(info)
