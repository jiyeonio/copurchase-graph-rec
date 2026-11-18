# From Sentences to Structure: Co-Purchase Graphs for Next-Purchase Recommendation

Jiyeon Kim, Jangmin Oh (School of AI Convergence, Sungshin Women's University). Submitted to *Applied Sciences* (MDPI).

Code that produces the tables, figures and in-text numbers of the paper.

## Model

Each item has two fixed representations. One is the [CLS] vector of the item description encoded with KR-ELECTRA (text representation); the other is the vector learned by LightGCN on the item–item co-purchase graph built from the orders of the training users (graph representation). Each representation is projected to d dimensions with Linear + LayerNorm + GELU. The order-level Transformer groups the items of an order under the same position and reads them as a set, and with block-causal attention it sees only up to the previous orders. The same Transformer reads the text representation and the graph representation separately and produces scores for all items, and the two scores are added with learned weights (late fusion). Training predicts the items of the next order at every order of the history and gives weight 2 to the term of the last target. A transaction in the code and documentation is an order in the paper.

## Requirements

Versions are pinned in `requirements.txt`. The target sampling and the split in `preprocess/katchers_split.py` depend on the random number implementations of pandas and scikit-learn, so use these versions (pandas 3.0.5, scikit-learn 1.9.0).

| Package | Version |
|---|---|
| Python | 3.13.11 |
| torch | 2.13.0 (CUDA 13.0) |
| torch-geometric | 2.8.0.post1 |
| numpy / scipy / pandas | 2.5.2 / 1.18.1 / 3.0.5 |
| networkx | 3.6.1 |
| transformers | 5.17.0 |
| scikit-learn | 1.9.0 |
| matplotlib | only needed for `analysis/emb_vis2.py` (Figure 5) |
| GPU | one NVIDIA RTX A5000 24 GB |

## Data

- Katcher's: obtained from the repository of the prior work [1] (https://github.com/msparkDev/ConcurrentBERT-Recommender)
- Instacart: the Instacart Online Grocery Shopping Dataset 2017, released through the Kaggle competition "Instacart Market Basket Analysis" (`orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, `departments.csv`)

[1] Park, M.; Oh, J. Enhancing e-commerce recommendation systems with multiple item purchase data: A bidirectional encoder representations from transformers-based approach. *Applied Sciences* 2024, 14, 7255. https://doi.org/10.3390/app14167255

## How to run

All paths are defined in `paths.py` (Python) and `config.sh` (shell). The defaults are `data/katchers_split`, `data/katchers`, `data/instacart`, `results/` and `paper/` under this folder, and they can be changed with the environment variables `KATCHERS_SPLIT`, `KATCHERS_DATA`, `INSTACART_DATA`, `RESULTS`, `PAPER` and `PY`.

0. Download `raw_data_katchers.csv` (in the `data/` folder) from the repository of [1].

1. Katcher's user filtering and 60/20/20 split

```bash
python preprocess/katchers_split.py --raw <raw_data_katchers.csv> --out-dir data/katchers_split
```

2. Preprocessing (targets, 1:49 test candidates, co-purchase graph)

```bash
python preprocess/katchers.py  --data-dir data/katchers_split --output-dir data/katchers
python preprocess/instacart.py --raw-dir <Instacart csv folder> --out-dir data/instacart
python graphs/lightgcn_g.py --dataset instacart --data-dir data/instacart --freq 3 --emb-dir data/instacart \
    --seeds 42 43 44 45 46 --checkpoint-dir results/ic_graphs/ckpt
```

3. Graphs, text embeddings, training, analysis, tables (GPU jobs one at a time, in order)

```bash
cd scripts
bash 01_graphs_katcher.sh        # Katcher's co-purchase graphs (f≥1, f≥2, f≥3), LightGCN, rewired graphs, edge-removed graphs
bash 02_graphs_instacart.sh      # Instacart rewired graphs and LightGCN
bash 03_text_embeddings.sh       # text embeddings (KR-ELECTRA; Instacart ELECTRA-base)
bash 04_train_katcher.sh         # Katcher's: Text+Graph, Text only, Graph only, ID only
bash 05_train_instacart.sh       # Instacart: the same configurations
bash 06_controls.sh              # retraining with the rewired graphs
bash 07_variants.sh              # variants: item-level, last order only, mean readout, final-target weight 1 and 4
bash 08_threshold_katcher.sh     # Katcher's co-purchase thresholds f≥2, f≥3
bash 09_baselines_full.sh        # baselines (SASRec, BERT4Rec, GRU4Rec, LightGCN)
bash 10_edge_removal.sh          # Katcher's edge removal
bash 11_english_encoder.sh       # Instacart ELECTRA-base
bash 12_analysis.sh              # rank dumps, group analyses, nearest neighbours, Figures 5 and 6
bash 13_tables.sh                # paper/tables/*.tex, paper/numbers.tex
```

To run a single model directly:

```bash
python model/model_katchers.py  --exp kc11/NAME --src text+graph --text-path results/text/kc/item_embeddings_text.npy \
    --graph-dir results/graphs/g1/emb --out-root results/runs --test
python model/model_instacart.py --exp ic11/NAME --src text+graph --graph-dir data/instacart --out-root results/runs --test
```

`--src {text+graph|text|graph|id}` is the item representation, and the other defaults are the settings of the paper.
`--item-level` is the item-level variant, `--last-order-only` the last-order variant, `--readout mean` the mean readout of the order state and `--final-weight` the weight of the final-target term (default 2).
`--eval-only --ckpt-dir <folder>` evaluates saved checkpoints without training.

## Tables and figures → scripts

The numbering is that of the paper. Unless a row says otherwise, the table files (`paper/tables/T_*.tex`) and the macros (`paper/numbers.tex`) are written by `tables/make_tables.py` (13). The macros of BERT-NSP are written by `proto_bert_nsp/make_numbers.py` into a file of their own. For Tables 1, 2, 5 and A11 the scripts write the values to a JSON file, and the table in the paper (for Appendix H also its macros) is typeset from that file. Tables 3 and 4 and Figures 1–4 are written in the manuscript and have no script. For tables filled by macros, the output column gives the prefix of the macro names.

| Paper | Output | Run (scripts) | Code |
|---|---|---|---|
| Table 1 | `paper/tables/check_stats.json`, `\kcisolated`, `\icisolated` | 13, 12 | `tables/check_stats.py`, `analysis/link_an_lg.py` |
| Table 2 | `paper/tables/check_stats.json` (key `r1`) | 01, 02, 13 | `tables/check_stats.py` |
| Figure 1, 2, 3, 4 | — | — | TikZ figures (not in this repository) |
| Table 3 | — | — | notation, written in the manuscript |
| Table 4 | — | — | settings and environment, written in the manuscript; the parameter count is the macro `\oursparams` (in-text numbers, last row) |
| Table 5 | `results/extra/bench_cost.json` | `python analysis/bench_cost.py` (GPU; not called by a numbered script) | `analysis/bench_cost.py` |
| Table 6 | macros `\kc…`, `\ic…`, `\kcbase…`, `\icbase…` | 04, 05, 09 | `tables/make_tables.py`, `baselines/base_full_eval.py` |
| Table 6 and Table A1, row BERT-NSP (Katcher's), and the BERT-NSP numbers in the text | `results/proto_bert_nsp/numbers_bert_nsp.tex`: macros `\kcbaseBERTNSP…`, `\kcdvsBERTNSP…`, `\kccivsBERTNSP…`, `\kcBERTNSPcut…` | 04, then `proto_bert_nsp/run_all.sh` (stages 1 and 2) | `proto_bert_nsp/run.py`, `proto_bert_nsp/make_numbers.py` (see `proto_bert_nsp/README.md`) |
| Table 7 | `T_knn.tex` | 12 | `analysis/kc_item_meta.py`, `analysis/emb_vis2.py` |
| Figure 5 | `paper/figs/emb_tsne_ic.pdf` | 12 | `analysis/emb_vis2.py` |
| Figure 6 | `paper/figs/fig_case.tex` | 12 | `analysis/make_case.py` |
| Table 8 | `T_link.tex` | 04, 05, 06, 12 | `analysis/link_an_lg.py` |
| Table 9 | macros `\kcst…` | 06, 12 | `analysis/story_an.py` |
| Table 10 | macros `\kcctl…`, `\kcgctl…`, `\icctl…`, `\icgctl…` | 01, 02, 06, 12 | `analysis/link_an_lg.py` |
| Table 11 | `T_degree.tex` | 12 | `analysis/link_an_lg.py` |
| Table 12 | macros `\kcabl…`, `\icabl…` | 07 | `analysis/ablation_an.py` |
| Table 12, row of the mean readout (runs `T_readout_mean`) | macros `\kcablreadoutmean…`, `\icablreadoutmean…` | 07 | `model/core.py` (`--readout mean`), `analysis/ablation_an.py`, `tables/make_tables.py` |
| Table A1 | macros `…Afive`, `…Ffive` | 04, 05, 09 | `tables/make_tables.py`, `baselines/base_full_eval.py` |
| Table A2 | `T_sig.tex` | 04, 05 | `tables/make_tables.py` |
| Table A3 | `T_unmasked.tex` | 04, 05, 09 | `tables/make_tables.py`, `baselines/base_full_eval.py` |
| Table A4 | `T_link_unmasked.tex` | 12 | `analysis/link_an_lg.py` |
| Table A5 | `T_control_full.tex` | 06, 12 | `analysis/link_an_lg.py` |
| Table A6 | macros `\kcst…` | 06, 12 | `analysis/story_an.py` |
| Table A7 | `T_edges.tex` | 01, 10 | `graphs/drop_edges.py`, `analysis/edge_removal_an.py` |
| Table A8 | `T_fsens.tex` | 01, 08 | `graphs/build_graph_katchers.py` |
| Table A9 | `T_ic_enc.tex` | 03, 11 | `graphs/text_electra_en.py`, `tables/make_tables.py` |
| Appendix G, Table A10 (runs `T_finalw1`, `T_finalw4`) | macros `\kcablfinalw…`, `\icablfinalw…` | 07 | `model/core.py` (`--final-weight`), `analysis/ablation_an.py`, `tables/make_tables.py` |
| Appendix H, Table A11 | `results/analysis/textft_an.json` | 01, 03, 04, 05, then `proto_textft/run_all.sh` (cache, train, an) | `proto_textft/core.py`, `proto_textft/text_electra_partial.py`, `proto_textft/textft_an.py` |
| Section 5.3, weighted degrees in the real and the rewired graphs | `paper/tables/rewire_degree.json` (the script also writes `paper/tables/T_rewire_degree.tex`) | 01, 02, 13 | `tables/rewire_degree_check.py` |
| In-text numbers (Holm correction, group shares, comparison by order size, token statistics, parameter counts, etc.) | `paper/numbers.tex`, `paper/tables/holm.json` | 12, 13 | `tables/make_tables.py`, `analysis/extra_an2.py`, `analysis/extra_ordsize.py`, `analysis/tok_stats.py` |

## Citation

```bibtex
@article{kim2026sentences,
  title   = {From Sentences to Structure: Co-Purchase Graphs for Next-Purchase Recommendation},
  author  = {Kim, Jiyeon and Oh, Jangmin},
  journal = {Applied Sciences},
  year    = {2026},
  note    = {Submitted}
}
```
