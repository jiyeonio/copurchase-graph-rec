# proto_textft: partial fine-tuning of KR-ELECTRA with the recommender (Appendix H, Table A11)

The paper keeps the text representation fixed. This folder trains part of the text encoder together with the recommender and compares the result with the fixed text representation of the paper, on Katcher's and on Instacart (Appendix H, Table A11).

## What it does

- Of the 12 layers of KR-ELECTRA (`snunlp/KR-ELECTRA-discriminator`), the **last 2 (layers 11–12) are trained end-to-end with the recommender**. Layers 1–10 are frozen; their output is computed once and cached (`text_electra_partial.py`). The [CLS] vector is read, as in the paper.
- At every training step all item descriptions are passed through the 2 trained layers to rebuild the text table. At evaluation the table is rebuilt from the final weights at every call of `model()` (eval mode, so it equals a table computed once).
- Settings: `--enc-lr 2e-5`, `--enc-dropout 0`. Everything else is the default of `model/core.py` (the settings of the paper): user batch 64, recommender lr 2.5e-4, at most 100 epochs, checkpoint chosen by validation 1:49 NDCG@10 (every 2 epochs on Katcher's, every 4 on Instacart), patience 5, gradient clipping 5.0 over all parameters, final weight 2. The training loss is not compiled.
- On Instacart `--enc-checkpoint` is on: activation checkpointing of the 2 trained layers during training (their intermediate values are recomputed in the backward pass). Instacart has 3,542 items, and without it Text+Graph does not fit in a 24 GB GPU (Text only 21.6 GB, Text+Graph out of memory; with it 12.7 GB and 22.5 GB). The computation is the same; results differ only as much as GPU training differs between two runs anyway. The option is off by default and off on Katcher's.
- Models: Text only and Text+Graph, seeds 42–46. They are compared per user with the same models of the paper (fixed text representation), and on Instacart also with ID only (`textft_an.py`).

## Files

| File | Content |
|---|---|
| `core.py` | `model/core.py` with the option `--text-ft top` (partial fine-tuning, encoder lr and dropout) and `--enc-checkpoint` (off by default). |
| `model_katchers.py`, `model_instacart.py` | Byte-identical copies of the same files in `model/`; `import core` reads `core.py` of this folder. |
| `text_electra_partial.py` | Cache of the output of the frozen layers 1–10 (same input sentences, tokenizer, max_length and batches as `graphs/text_electra.py`). |
| `textft_an.py` | Paired tests per user against the fixed text representation (bootstrap and Wilcoxon as in the paper, `analysis/common.py`), values per target group (`analysis/link_an_lg.py`), learned fusion weights, results per seed. Output: `<RESULTS>/analysis/textft_an.json` (Table A11). |
| `run_all.sh` | Steps `cache`, `train`, `an`. |

Outside this folder the scripts only import `paths.py`, `graphs/text_electra.py`, `analysis/common.py` and `analysis/link_an_lg.py`.

## How to run

The `an` step compares with the runs of the paper, so these must exist first (see "How to run" in the main README):

- `scripts/04_train_katcher.sh` and `scripts/05_train_instacart.sh`: runs `<ds>11/I_text`, `I_text_co_lgcn_late`, `K_id` (`<RESULTS>/runs/`) and their per-user dumps `<RESULTS>/analysis/pu/<ds>_{text,text_co_lgcn_late,id_fix}_{1to49,full}_seed{S}.json`.
- Katcher's text embedding (`scripts/03_text_embeddings.sh`) and graph embeddings (`scripts/01_graphs_katcher.sh`). Instacart text and graph embeddings are read from `INSTACART_DATA`, as in the runs of the paper.

```bash
export PY=python KATCHERS_DATA=data/katchers INSTACART_DATA=data/instacart RESULTS=results    # as paths.py
export KC_TEXT=results/text/kc/item_embeddings_text.npy KC_GRAPH=results/graphs/g1/emb KC_TEXT_DATA=data/katchers
export CACHE=results/textft_cache                     # caches: Katcher's 0.3 GB, Instacart 0.7 GB
bash proto_textft/run_all.sh cache                    # frozen layers 1-10, both datasets (GPU, under a minute each)
bash proto_textft/run_all.sh train                    # both datasets, seeds 42-46, Text only and Text+Graph (GPU)
bash proto_textft/run_all.sh an                       # paired tests (CPU) -> results/analysis/textft_an.json
DATASETS=ic SEEDS=42 bash proto_textft/run_all.sh train   # one dataset, one seed
```

Outputs: `<RESULTS>/runs/<ds>11/T_textft_text*/result_U_seed{S}.json`, checkpoints `best_base_real_seed{S}.pt` (about 59 MB each), per-user dumps `<RESULTS>/analysis/pu2/<ds>_T_textft_*`, logs `<RESULTS>/analysis/<ds>_T_textft_*.seeds_<S>.log`, `<RESULTS>/analysis/textft_an.json`.

## Cost (one RTX A5000 24 GB)

- Trained parameters: 14,175,744 in the 2 encoder layers plus about 0.5M in the recommender.
- Katcher's: about 12 minutes per seed and run, GPU memory about 7 GB (fixed encoder of the paper: 0.4–0.6 minutes).
- Instacart (with `--enc-checkpoint`): about 138 minutes per seed for Text only and 144 for Text+Graph, GPU memory 12.7 GB and 22.5 GB (fixed encoder of the paper: 5–9 minutes). All Instacart runs reached the limit of 100 epochs.

## Code of the results

The table gives the sha256 of the files with which the results of Appendix H were produced and of the files as they are now. Only comments, docstrings and documentation changed in between; the training and evaluation code (`core.py`, `model_*.py`) is unchanged.

| File | sha256 when the results were produced | sha256 now |
|---|---|---|
| `core.py` | `38778d1e9e21cea49a0d1316e53159366ccad66263fd2acfc236626e12970488` | same |
| `model_katchers.py` | `4c330fa408efc3fd2575b828917786a7e9e5394ff951cf95405dd7b51ea5a2fe` | same |
| `model_instacart.py` | `000ba6661079ce55d279b284288c267602b64199c41e05ee524bd6f807910c47` | same |
| `text_electra_partial.py` | `9447323b8b850891420d6c6e3cdc24883ae94335490644c3f1aa2063a4d245a8` | `90d3c75d4c33547a39abf6ca84d3cdded66ab6bafb940f85dc499f3efa6d07c5` (docstring and comment) |
| `textft_an.py` | `b1c2a01adf60c39e9b13447999f78196a273e0bbaafea5cdfe8138a9f5f23ed3` | `5b811f658745f67a78bc7dfd943e70f89999e056c81ffa01a31520bd045e05b8` (docstring) |
| `run_all.sh` | `2a6fd9694b3f3aad3a4da675068199a13ecea1df40fbe5546c0364d7b5d01448` | `fd5704c6772302ae07cdca1d3de803ab3cb27465d9255780bf0782ac1b93dd7c` (comments) |
| `README.md` | `e80150535fdba1dc76f9d4e1bb20d863f55e23e68ea9c4401e7268dcbb1ac9cf` | this file (rewritten) |
