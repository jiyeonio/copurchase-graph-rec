"""Partial fine-tuning of the text encoder: output of the frozen lower layers of KR-ELECTRA for every item, computed once.
Same input sentence, tokenizer, max_length and batch size as graphs/text_electra.py (Katcher's: katchers_texts, batches of 32;
Instacart: instacart_texts, batches of 256; padding to the longest sentence of each batch); the hidden states after layer
--frozen-layers (default 10 of 12) are saved zero-padded to max_length, with the attention mask and the batch size.
core.py --text-ft top runs the remaining layers on this cache, in the same batches, and reads [CLS].
Appendix H. The meta of the Katcher's cache has no dataset field (same file as the cache behind the results of Appendix H).
Run: python proto_textft/text_electra_partial.py --dataset {katchers|instacart} --data-dir <folder with the metadata file> --out <file .pt>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graphs"))
from text_electra import TEXT_MODEL_NAME, TEXT_DEFAULTS, katchers_texts, instacart_texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["katchers", "instacart"], default="katchers")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frozen-layers", type=int, default=10)
    p.add_argument("--model-name", default=TEXT_MODEL_NAME)
    a = p.parse_args()
    d = TEXT_DEFAULTS[a.dataset]; L, bs = d["max_length"], d["batch_size"]
    texts = katchers_texts(a.data_dir) if a.dataset == "katchers" else instacart_texts(a.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModel.from_pretrained(a.model_name).to(device).eval()
    H = torch.zeros(len(texts), L, model.config.hidden_size); M = torch.zeros(len(texts), L, dtype=torch.bool)
    ids = torch.zeros(len(texts), L, dtype=torch.long); cls = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], padding=True, truncation=True, max_length=L, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        n, S = enc["input_ids"].shape
        H[s:s + n, :S] = out.hidden_states[a.frozen_layers].cpu()       # hidden_states[0] = embeddings, [k] = output of layer k
        M[s:s + n, :S] = enc["attention_mask"].bool().cpu(); ids[s:s + n, :S] = enc["input_ids"].cpu()
        cls.append(out.last_hidden_state[:, 0].cpu().numpy())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model_name, "frozen_layers": a.frozen_layers, "num_layers": model.config.num_hidden_layers, "max_length": L,
            "batch_size": bs, "n": len(texts), "tokens_max": int(M.sum(1).max()), "device": str(device),
            "commit_hash": getattr(model.config, "_commit_hash", None)}
    if a.dataset != "katchers":
        meta["dataset"] = a.dataset                                       # no dataset field for Katcher's (same file as the cache of Appendix H)
    torch.save({"hidden": H, "mask": M, "input_ids": ids, "meta": meta}, a.out)
    np.save(a.out.with_suffix(".cls.npy"), np.vstack(cls))         # [CLS] of the full model in the same pass (= text_electra.py output)
    json.dump(meta, open(a.out.with_suffix(".json"), "w"), indent=1); print(meta)


if __name__ == "__main__":
    main()
