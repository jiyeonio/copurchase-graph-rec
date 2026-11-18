"""Inference cost: proposed scorer (one pass -> scores for all items) vs a BERT next-sentence-prediction scorer (one pass per candidate),
as in Park & Oh (2024): bert-base-multilingual-cased, inputs up to 512 tokens. Architecture-level timing with random weights on the same GPU.
Paths from paths.py (KATCHERS_DATA, INSTACART_DATA, RESULTS). -> <RESULTS>/extra/bench_cost.json"""
import json, os, sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import KC, IC, N_ITEMS, EXTRA
dev = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True
def sync(): torch.cuda.synchronize()
def timeit(fn, n=30, warm=5):
    for _ in range(warm): fn()
    sync(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)
class Ours(nn.Module):
    def __init__(s, N, d=128):
        super().__init__(); s.N = N; s.d = d
        s.text = nn.Parameter(torch.randn(N, 768), requires_grad=False); s.graph = nn.Parameter(torch.randn(N, 64), requires_grad=False)
        s.pt = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d), nn.GELU()); s.pg = nn.Sequential(nn.Linear(64, d), nn.LayerNorm(d), nn.GELU())
        s.pos = nn.Embedding(50, d); s.phi = nn.Parameter(torch.zeros(2))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True), 2, enable_nested_tensor=False); s.ln = nn.LayerNorm(d)
    def forward(s, items, tpos):                      # items, tpos: (B, L)
        Rt, Rg = s.pt(s.text), s.pg(s.graph)          # item matrices (recomputed here; they can be cached at deployment)
        x = torch.cat([Rt[items], Rg[items]], 0) + s.pos(tpos).repeat(2, 1, 1)
        m = (tpos[:, None, :] > tpos[:, :, None]).repeat(2 * 4, 1, 1)      # block-causal: attend to own and earlier orders
        h = s.ln(s.enc(x, mask=m))[:, -1]; B = items.shape[0]; w = torch.softmax(s.phi, 0)
        return w[0] * h[:B] @ Rt.T / s.d ** 0.5 + w[1] * h[B:] @ Rg.T / s.d ** 0.5
def hist_lengths(ds):
    if ds == 'kc':
        th = json.load(open(KC + 'test_history.json')); return [[len(set(o['items'])) for o in th[u]][-50:] for u in th], N_ITEMS['kc']
    sp = json.load(open(IC + 'split.json')); h = json.load(open(IC + 'user_item_history.json'))
    return [[len(set(o)) for o in h[str(u)]][-50:] for u in sp['test']], N_ITEMS['ic']
from transformers import BertConfig, BertForNextSentencePrediction
bert = BertForNextSentencePrediction(BertConfig(vocab_size=119547)).to(dev).eval().half()      # bert-base-multilingual-cased architecture
nb = sum(p.numel() for p in bert.parameters())
OUT = {'bert_params': nb, 'gpu': torch.cuda.get_device_name(0)}
@torch.inference_mode()
def bert_user(C, L, bs=128):
    ids = torch.randint(1000, 100000, (C, L), device=dev)
    for i in range(0, C, bs): bert(input_ids=ids[i:i + bs]).logits
for L in (128, 256, 512):
    OUT[f'bert_L{L}_C50_ms'] = timeit(lambda: bert_user(50, L), n=10, warm=2)
for ds in ('kc', 'ic'):
    sizes, N = hist_lengths(ds); model = Ours(N).to(dev).eval()
    OUT[f'{ds}_ours_trained_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad); OUT[f'{ds}_N'] = N
    toks = [sum(s) for s in sizes]; OUT[f'{ds}_tokens_mean'] = float(np.mean(toks)); OUT[f'{ds}_tokens_max'] = int(max(toks)); OUT[f'{ds}_orders_mean'] = float(np.mean([len(s) for s in sizes]))
    rng = np.random.default_rng(0); idx = rng.choice(len(sizes), 200, replace=False); ts = []
    @torch.inference_mode()
    def one(s):
        L = sum(s); items = torch.randint(0, N, (1, L), device=dev); tpos = torch.tensor([[k for k, c in enumerate(s) for _ in range(c)]], device=dev)
        with torch.autocast('cuda', dtype=torch.float16): return model(items, tpos)
    for _ in range(5): one(sizes[idx[0]])
    for i in idx:
        sync(); t = time.perf_counter(); one(sizes[i]); sync(); ts.append(time.perf_counter() - t)
    OUT[f'{ds}_ours_ms_per_user_all_items'] = float(np.mean(ts) * 1000)
    for L in (128, 512):
        OUT[f'{ds}_bert_L{L}_all_items_ms'] = timeit(lambda: bert_user(N, L), n=3, warm=1)
    del model
os.makedirs(EXTRA, exist_ok=True)
json.dump(OUT, open(EXTRA + 'bench_cost.json', 'w'), indent=1); print(json.dumps(OUT, indent=1))
