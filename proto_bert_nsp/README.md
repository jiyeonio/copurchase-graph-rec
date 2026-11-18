# proto_bert_nsp: BERT-NSP in the setting of the paper

BERT-NSP is the sentence-based recommender of Park and Oh (2024): the purchase history is written as a prompt, a candidate product as a second sentence, and a BERT with a next-sentence head scores the pair. This folder trains and evaluates it on the Katcher's data of this paper, with the same users, targets, 1:49 candidates and evaluation as every other model. It is self-contained: nothing outside the folder is changed, and from the rest of the repository only `paths.py` is imported.

The implementation follows the corrected re-implementation of the 2024 model in <https://github.com/jangmino/txncausal> (`src/mvcp/baselines/concurrentbert.py`, options `--text repaired --cand canonical`).

## What is kept from where

| | Rule | Source |
|---|---|---|
| Users, targets, 1:49 candidates | the files of `preprocess/katchers.py`, read as the baselines read them; none of them is made again | paper |
| Seeds | 42–46 | paper |
| Evaluation | 1:49 with the fixed candidates; full catalogue, masked and unmasked; novel / repeat; NDCG@5 and @10 | paper |
| Model | `google-bert/bert-base-multilingual-cased` with its next-sentence head; score of a candidate = logit[1] − logit[0] | 2024 model |
| Training pairs | per user one positive pair (prompt, target) and one negative pair (prompt, an item drawn uniformly from all items) | 2024 model |
| Training | 10 epochs, AdamW, learning rate 2e-5 with linear decay, weight decay 0.01, batch 16 | 2024 model |
| Checkpoint | the epoch with the highest validation NDCG@10 under the 1:49 protocol, as for the other models | paper |
| Prompt | one line per order, oldest first, with the payment time, a `[동시구매]` mark for orders of several purchased rows, and category, name and attributes of every row; cut where prompt and longest candidate would reach 512 tokens | 2024 model, corrected |
| Candidate text | category and name of the product, one text per product for the target and for the negatives | corrected |

Two things were corrected relative to the code released with the 2024 study. Its prompt builder kept the history only when the history overflowed, so most prompts held the heading alone; here every order line is in the prompt. And it wrote the target from the user's own purchase row but the negatives from a representative row of each product, so the two could be told apart by their source; here every candidate is written from the same per-product text.

The 2024 model kept the epoch with the lowest validation loss. That checkpoint is evaluated under 1:49 as well and stored next to the result (`by_val_loss`); where the two rules choose different epochs, its weights are kept too. It is a record of how much the rule matters, not the result.

## Inputs

Both folders are only read; a run refuses an output folder inside one of them.

- `KATCHERS_DATA` (`data/katchers`): the output of `preprocess/katchers.py`. Defines the split, the targets and the candidates.
- `KATCHERS_SPLIT` (`data/katchers_split`): `train_data.csv`, `val_data.csv`, `test_data.csv`. BERT-NSP needs the text of every purchased row (`category_name`, `product_name`, `attribute_values`) and its payment time, which the first folder does not hold.

Before anything is trained, `check_protocol.py` verifies that the two folders describe the same users, orders and targets, that this folder reads the data as `baselines/original/kc_sasrec.py` does, and that its evaluation gives the per-user NDCG of the baselines' evaluators. It needs the tokenizer but no GPU.

## Run

`PY`, `KATCHERS_DATA`, `KATCHERS_SPLIT` and `RESULTS` are those of `config.sh` / `paths.py`. Beyond `requirements.txt` nothing is needed; the pretrained model (about 0.7 GB) is downloaded from the Hugging Face hub on first use.

```bash
# 1. does the code run here? (small random data in a temporary folder; touches neither the data nor results/)
bash proto_bert_nsp/smoke_test.sh              # last line: SMOKE TEST OK

# 2. the run: stage 1 (training and 1:49, seeds 42-46), then stage 2 (full catalogue, seeds 42-46)
bash proto_bert_nsp/run_all.sh
```

Stage 1 and stage 2 can be started separately (`run_all.sh 1`, `run_all.sh 2`), and for some seeds only, for example on a second GPU: `CUDA_VISIBLE_DEVICES=1 SEEDS="45 46" bash proto_bert_nsp/run_all.sh 2`. A finished seed is skipped, and stage 2 continues an interrupted seed with the users that are still missing, so the same command can be started again after an interruption. After every stage the results so far are packed into `<RESULTS>/proto_bert_nsp_results.tgz`.

A single call, for other settings (give a new `--name` whenever a setting changes; a result folder refuses a run with other settings, data or code):

```bash
python proto_bert_nsp/run.py train --seeds 42            # stage 1 of one seed
python proto_bert_nsp/run.py full  --seeds 42            # stage 2 of that seed
```

On the RTX A5000 of the paper, stage 1 took about 3.3 hours per seed (ten epochs, with the 1:49 ranking of the validation users after every epoch) and stage 2 about 6 hours per seed. The logs print the elapsed minutes, and stage 2 prints the minutes left.

### Seeds of the first version

The first version of this folder had the two checkpoint rules the other way round: its result was the epoch with the lowest validation loss, and the epoch with the highest validation NDCG@10 was only evaluated under 1:49. It wrote to `kc/bert_nsp`. For a seed where both rules chose one epoch there, the checkpoint and all results of that run are those of this version as well, so `run.py` takes such a seed over from that folder instead of training it again, together with its full-catalogue ranking if that is finished. A seed for which the rules chose different epochs is trained again, because the first version did not keep the weights of the other epoch; its 1:49 and full-catalogue results then both come from the new training run.

This happens by itself when `kc/bert_nsp` is next to the output folder (`--earlier-run` names another folder, `--earlier-run none` trains every seed). A seed is taken over only if the earlier run used the same input files, settings, sentences and evaluation: the two configurations may differ in the name of the run, in the rule and in `run.py`, and in nothing else. If they differ otherwise, the run stops. The earlier folder is only read. Every result row says where it comes from (`taken_over_from` and `taken_over_full_from`, with the sha256 of the file and of the code that wrote it; `earlier_run` for a seed that was trained again), and `full_eval.json` lists the seeds concerned.

Between the two versions only `run.py` changed, and in it only which of the two checkpoints is the result. On small random data the two versions give the same loss and validation NDCG@10 in every epoch and the same per-user results for both rules.

## Output

`<RESULTS>/proto_bert_nsp/kc/bert_nsp_ndcg/`

| File | Content |
|---|---|
| `full_eval.json` | per-seed rows and mean / std over the seeds, with the keys of `baselines/base_full_eval.py`; `seeds_full` lists the seeds whose stage 2 is finished, `seeds_taken_over` and `seeds_full_taken_over` those that come from the first version |
| `per_user_seed{S}.json` | per-user NDCG: `1to49`, `full` (masked / unmasked), in the layout of `base_full_eval.py`; also `1to49_by_val_loss` and `1to49_from_full` |
| `result_seed{S}.json`, `full_seed{S}.json` | stage 1 and stage 2 of one seed with the settings, the sha256 of every input file and of the code |
| `scores_full_seed{S}.npy` | scores of all items for every test user (rows in the order of the test users); not packed |
| `checkpoints/best_seed{S}.pt` | the selected checkpoint, 0.7 GB per seed; `….by_val_loss` where the other rule chose another epoch; not packed. A seed taken over keeps its checkpoint and its scores in the earlier folder |
| `../../numbers_bert_nsp.tex`, `../../summary.json` | written by `make_numbers.py`: macros in the style of `tables/make_tables.py` and the paired test against Text+Graph. The second line of the first file names the checkpoint rule and the origin of the seeds |

`make_numbers.py` writes new macros only (`\kcbaseBERTNSPA`, `\kcbaseBERTNSPF`, …, `\kcdvsBERTNSPA`, …) into a file of its own and leaves `numbers.tex` alone. The paired test uses the per-user files of the paper's model in `<RESULTS>/analysis/pu/`; a macro whose inputs are incomplete is written as `[TODO]`.

## Properties of the model in this setting

- **Long histories lose their most recent orders.** The prompt is filled from the oldest order and stops at 512 tokens. On Katcher's this affects 543 of the 3,358 test users (16.2%); 8 of them have no order in the prompt. This is the rule of the 2024 model and is kept. The paper's models read the most recent 50 orders, which no Katcher's user exceeds.
- **Products with the same text tie.** 122 of the 1,608 products share category and name with another product. BERT-NSP cannot tell them apart, so a target and such a negative get the same score; 7 test users have such a negative among their 49, and an eighth has one whose name differs only in spacing, which the tokenizer does not see. Ties get the mid-rank, and `tied_users_1to49` counts the users concerned. For the other models of the paper no ties occur and the mid-rank equals the rank they use.
- **Rows of one order are written in the order of the split file.** The implementation this follows sorted them with an unstable sort, which does not fix their order (on the machine where the two were compared, about 4% of the prompts differ in the order of the items inside a line). The lines hold the same items either way.
- **The prompt of the full-catalogue ranking is cut against the longest candidate text of the catalogue**, the prompt of the 1:49 ranking against the longest of the user's 50 candidates, so the two prompts of a user can differ by an order line (564 against 543 test users lose an order). `1to49_from_full` gives 1:49 from the full-catalogue scores as a check.
- **Katcher's only.** The prompt and the candidate texts are Korean sentences, and Instacart would need a design of its own.

## What was checked

Against the implementation this follows, on a copy of the Katcher's split files: with the rows of an order in the order that implementation produced, all 20,142 training pairs, all 6,716 validation pairs and all 3,358 test prompts with their 50 candidates are identical, text for text, to the files its `prep` step writes (including the negatives it draws with seed 0, `--pair-seed 0`). The training loop gives the weights of the `transformers` Trainer it replaces and the same validation loss per epoch (a small BERT, same batches, dropout off: the weights differ by 2e-7 where they moved by 3e-4, and with a hundred times the learning rate by 2e-6 where they moved by 3e-2).

Against the paper: `check_protocol.py`, run by `run_all.sh` before training.

The take-over of seeds, on small random data: the earlier folder is unchanged afterwards, the seeds taken over have the per-user results of the earlier run, a seed trained again has the results of a run of this version without an earlier folder, and the macros of `make_numbers.py` keep their names.

## Code of the results

Every `result_seed{S}.json` and `full_seed{S}.json` records the sha256 of the four files below (`code_sha256`). The results of the paper were written with the files of the middle column. Since then one docstring was reworded in `nsp_data.py` and one in `nsp_eval.py`; the code itself is unchanged.

| File | sha256 when the results were produced | sha256 now |
|---|---|---|
| `nsp_data.py` | `348b7758d4d3a5e12af73981a4913529899bf8bb1fab894f93068302749bf82c` | `32fc5bd403fcab8bb7336c9a935362a502bbfbf904967ea4aaec5814efa61dce` (docstring) |
| `nsp_eval.py` | `1dc27258823e39cdfbe6b6b312313d8372edd5fad0a6fb4993b2a9e471d7764f` | `7c9d76acf64e00268863ee41a3d0d83c9caa09f73d06837d3afb3cc79ff4d7e8` (docstring) |
| `nsp_text.py` | `3e28ee88a62b14bc0ebd6d91e1e68dbec73eac65949289495bc705a324181223` | same |
| `run.py` | `1b76ccb62f5b8f79aa120065e44f4f898103d8c67c73e22ef180a72e0e279573` | same |
