"""The sentences of BERT-NSP (Park and Oh, 2024), in the corrected form of the re-implementation this prototype follows
(https://github.com/jangmino/txncausal, src/mvcp/baselines/concurrentbert.py with `--text repaired --cand canonical`).

  prompt      "## 주문 히스토리" and one line per order, oldest first:
              "\n<payment time>의 주문 내역[ [동시구매]]: <category> - <name>[ <attributes>], ..."
              ([동시구매] marks an order with more than one purchased row). A line is added only while the prompt and the longest
              candidate of the user stay below 512 tokens as a pair; the first line that does not fit ends the prompt, so a long
              history loses its most recent orders. This is the rule of the released code and is kept.
  candidate   "## 다음 구매 상품 예측\n<category> - <name>": one text per product for the target and for every negative. The name is the
              most frequent product_name of the product over the three split files with leading [..] tags removed, the category its
              most frequent category_name. Attributes are left out, because they vary from one purchase to the next.

The two corrections relative to the released code of the 2024 study: every order line is in the prompt (the released builder kept the
history only when it overflowed), and the target is written from the same source as the negatives (it was written from the user's own
purchase row, the negatives from a representative row of the product).

A pair is given to BERT as [CLS] prompt [SEP] candidate [SEP]. Prompts and candidates are tokenised once and the pairs are assembled
from the token ids; check_pairs() compares the assembled pairs with the tokenizer's own pair encoding.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

MODEL = "google-bert/bert-base-multilingual-cased"
MAX_LEN = 512
PROMPT_HEAD = "## 주문 히스토리"
CANDIDATE_HEAD = "## 다음 구매 상품 예측\n"
_TAGS = re.compile(r"^\s*(\[[^\]]*\]\s*)+")


def load_tokenizer(name=MODEL):
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(name)
    tok.model_max_length = 10**9       # only silences the length warning of pair_fits, which measures prompts beyond 512 tokens;
    return tok                         # nothing is truncated by this attribute (check_pairs passes max_length itself)


class Texts:
    """tokenizer with the two operations the prompt builder needs, and a cache of the token ids of single texts"""

    def __init__(self, tokenizer):
        self.tok, self._ids = tokenizer, {}
        self.cls, self.sep, self.pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id

    def ids(self, text):
        """token ids of one text without [CLS] and [SEP]"""
        if text not in self._ids:
            self._ids[text] = np.asarray(self.tok(text, add_special_tokens=False).input_ids, dtype=np.int64)
        return self._ids[text]

    def longest(self, texts):
        """the text with the most tokens; of two equally long texts the later one (the comparison of the released code, folded)"""
        best = texts[0]
        for t in texts[1:]:
            if not len(self.ids(best)) > len(self.ids(t)):
                best = t
        return best

    def pair_fits(self, text, other):
        """True if the pair (text, other) with its three special tokens has fewer than 512 tokens"""
        return len(self.tok(text, other, padding=True, truncation=False).input_ids) < MAX_LEN


def order_line(paid_at, records):
    items = [f"{cat} - {name}" + ("" if attr is None else f" {attr}") for _, cat, name, attr in records]
    return f"\n{paid_at}의 주문 내역" + (" [동시구매]" if len(records) > 1 else "") + ": " + ", ".join(items)


def history_prompt(orders, texts, longest_candidate):
    """-> (prompt, number of orders in it). orders: UserRows.orders of nsp_data.py"""
    prompt, n = PROMPT_HEAD, 0
    for paid_at, records in orders:
        longer = prompt + order_line(paid_at, records)
        if not texts.pair_fits(longer, longest_candidate):
            break
        prompt, n = longer, n + 1
    return prompt, n


def candidate_texts(rows):
    """item index -> candidate text, from the three split files (rows: nsp_data.Rows)"""
    union = pd.concat([rows.frames[s][["product_id", "product_name", "category_name"]] for s in rows.frames], ignore_index=True)
    out = {}
    for pid, g in union.groupby("product_id"):
        names = Counter(str(n) for n in g["product_name"].dropna())
        raw = names.most_common(1)[0][0] if names else str(pid)          # ties: the name that occurs first
        cats = g["category_name"].dropna()
        cat = str(cats.mode().iloc[0]) if len(cats) else ""              # ties: the first in sorted order
        out[rows.item2idx[int(pid)]] = f"{CANDIDATE_HEAD}{cat} - {_TAGS.sub('', raw).strip() or raw}"
    return [out[i] for i in range(len(rows.item2idx))]


# ---------------------------------------------------------------- pairs as token ids
def pair_arrays(prompt_ids, candidate_ids, texts):
    """[CLS] prompt [SEP] candidate [SEP] -> (input ids, token type ids) as two int64 arrays"""
    ids = np.concatenate(([texts.cls], prompt_ids, [texts.sep], candidate_ids, [texts.sep]))
    types = np.zeros(len(ids), dtype=np.int64)
    types[len(prompt_ids) + 2:] = 1
    if len(ids) > MAX_LEN:
        raise ValueError(f"a pair of {len(ids)} tokens: the prompt was not cut to the budget of its candidates")
    return ids, types


def collate(pairs, texts):
    """pairs: [(prompt ids, candidate ids)] -> padded input_ids, attention_mask, token_type_ids (numpy, right-padded)"""
    arrays = [pair_arrays(p, c, texts) for p, c in pairs]
    width = max(len(a[0]) for a in arrays)
    ids = np.full((len(arrays), width), texts.pad, dtype=np.int64)
    mask, types = np.zeros_like(ids), np.zeros_like(ids)
    for r, (i, t) in enumerate(arrays):
        ids[r, :len(i)], mask[r, :len(i)], types[r, :len(i)] = i, 1, t
    return ids, mask, types


def check_pairs(text_pairs, texts):
    """the assembled pairs against the tokenizer's own encoding of the same pairs (truncation on, as in the released training)"""
    for prompt, cand in text_pairs:
        enc = texts.tok(prompt, cand, truncation=True, max_length=MAX_LEN)
        ids, types = pair_arrays(texts.ids(prompt), texts.ids(cand), texts)
        if enc.input_ids != ids.tolist() or enc.token_type_ids != types.tolist():
            raise ValueError("the assembled token ids differ from the tokenizer's pair encoding")
    return len(text_pairs)
