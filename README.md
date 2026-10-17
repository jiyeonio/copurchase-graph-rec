# From Sentences to Structure: Co-Purchase Graphs for Next-Purchase Recommendation

Jiyeon Kim, Jangmin Oh (School of AI Convergence, Sungshin Women's University). Submitted to *Applied Sciences* (MDPI).

논문의 표·그림·본문 수치를 만드는 코드다.

## 모델

상품마다 두 가지 고정 표현을 쓴다. 하나는 상품 설명문을 KR-ELECTRA로 인코딩한 [CLS] 벡터(텍스트 표현), 다른 하나는 학습 사용자의 주문으로 만든 상품–상품 동시구매 그래프에서 LightGCN이 학습한 벡터(그래프 표현)다. 각 표현은 Linear + LayerNorm + GELU로 d차원에 사영된다. 주문 단위 Transformer는 한 주문의 상품을 같은 위치로 묶어 집합으로 읽고, block-causal 어텐션으로 이전 주문까지만 본다. 텍스트 표현과 그래프 표현은 같은 Transformer가 각각 읽어 전체 상품의 점수를 내고, 학습되는 가중치로 두 점수를 더한다(late fusion). 학습은 이력의 모든 주문에서 다음 주문의 상품을 예측하고, 마지막 target의 항에 가중치 2를 준다. 코드와 문서의 transaction은 논문의 order(주문)다.

## Requirements

`requirements.txt`에 버전을 고정했다. `preprocess/katchers_split.py`의 정답 샘플링과 분할은 pandas·scikit-learn의 난수 구현을 따르므로 이 버전(pandas 3.0.5, scikit-learn 1.9.0)을 쓴다.

| 항목 | 버전 |
|---|---|
| Python | 3.13.11 |
| torch | 2.13.0 (CUDA 13.0) |
| torch-geometric | 2.8.0.post1 |
| numpy / scipy / pandas | 2.5.2 / 1.18.1 / 3.0.5 |
| networkx | 3.6.1 |
| transformers | 5.17.0 |
| scikit-learn | 1.9.0 |
| matplotlib | `analysis/emb_vis2.py`(그림 4)에만 필요 |
| GPU | NVIDIA RTX A5000 24 GB 한 장 |

## Data

- Katcher's: 선행 연구 [1] 레포에서 받음 (https://github.com/msparkDev/ConcurrentBERT-Recommender)
- Instacart: https://www.kaggle.com/c/instacart-market-basket-analysis/data (`orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, `departments.csv`)

[1] Park, M.; Oh, J. Enhancing e-commerce recommendation systems with multiple item purchase data: A bidirectional encoder representations from transformers-based approach. *Applied Sciences* 2024, 14, 7255. https://doi.org/10.3390/app14167255

## How to run

모든 경로는 `paths.py`(Python)와 `config.sh`(shell)에 정의되어 있다. 기본값은 이 폴더 아래의 `data/katchers_split`, `data/katchers`, `data/instacart`, `results/`, `paper/`이고, 환경변수 `KATCHERS_SPLIT`, `KATCHERS_DATA`, `INSTACART_DATA`, `RESULTS`, `PAPER`, `PY`로 바꿀 수 있다.

0. [1] 레포에서 `raw_data_katchers.csv`(`data/` 폴더)를 받는다.

1. Katcher's 사용자 필터링과 60/20/20 분할

```bash
python preprocess/katchers_split.py --raw <raw_data_katchers.csv> --out-dir data/katchers_split
```

2. 전처리 (target, 1:49 test 후보, 동시구매 그래프)

```bash
python preprocess/katchers.py  --data-dir data/katchers_split --output-dir data/katchers
python preprocess/instacart.py --raw-dir <Instacart csv 폴더> --out-dir data/instacart
python graphs/lightgcn_g.py --dataset instacart --data-dir data/instacart --freq 3 --emb-dir data/instacart \
    --seeds 42 43 44 45 46 --checkpoint-dir results/ic_graphs/ckpt
```

3. 그래프, 텍스트 임베딩, 학습, 분석, 표 (GPU 작업은 하나씩 순서대로)

```bash
cd scripts
bash 01_graphs_katcher.sh        # Katcher's 동시구매 그래프(f≥1, f≥2, f≥3), LightGCN, 재배선 그래프, 간선 제거 그래프
bash 02_graphs_instacart.sh      # Instacart 재배선 그래프와 LightGCN
bash 03_text_embeddings.sh       # 텍스트 임베딩 (KR-ELECTRA; Instacart ELECTRA-base)
bash 04_train_katcher.sh         # Katcher's: Text+Graph, Text only, Graph only, ID only
bash 05_train_instacart.sh       # Instacart: 같은 구성
bash 06_controls.sh              # 재배선 그래프로 재학습
bash 07_variants.sh              # item-level 변형, last-order 변형
bash 08_threshold_katcher.sh     # Katcher's 동시구매 임계값 f≥2, f≥3
bash 09_baselines_full.sh        # 베이스라인 (SASRec, BERT4Rec, GRU4Rec, LightGCN)
bash 10_edge_removal.sh          # Katcher's 간선 제거
bash 11_english_encoder.sh       # Instacart ELECTRA-base
bash 12_analysis.sh              # 순위 덤프, 집단별 분석, 최근접 이웃, 그림 4·5
bash 13_tables.sh                # paper/tables/*.tex, paper/numbers.tex
```

모델 하나를 직접 실행하려면:

```bash
python model/model_katchers.py  --exp kc11/NAME --src text+graph --text-path results/text/kc/item_embeddings_text.npy \
    --graph-dir results/graphs/g1/emb --out-root results/runs --test
python model/model_instacart.py --exp ic11/NAME --src text+graph --graph-dir data/instacart --out-root results/runs --test
```

`--src {text+graph|text|graph|id}`가 상품 표현이고, 나머지 기본값이 논문의 설정이다.
`--item-level`은 item-level 변형, `--last-order-only`는 last-order 변형이다.
`--eval-only --ckpt-dir <폴더>`는 학습 없이 저장된 checkpoint를 평가한다.

## 표·그림 → 스크립트

번호는 논문의 번호다. 표 파일과 매크로(`paper/numbers.tex`)는 모두 `tables/make_tables.py`(13)가 쓴다. 매크로로 들어가는 표는 출력 열에 매크로 이름의 앞부분을 적었다.

| 논문 | 출력 | 실행 (scripts) | 계산 코드 |
|---|---|---|---|
| Table 1 | `paper/tables/check_stats.json`, `\kcisolated`, `\icisolated` | 13, 12 | `tables/check_stats.py`, `analysis/link_an_lg.py` |
| Figure 1, 2, 3 | — | — | TikZ 그림 (이 저장소에 없음) |
| Table 2 | 매크로 `\kc…`, `\ic…`, `\kcbase…`, `\icbase…` | 04, 05, 09 | `tables/make_tables.py`, `baselines/base_full_eval.py` |
| Table 3 | `T_knn.tex` | 12 | `analysis/kc_item_meta.py`, `analysis/emb_vis2.py` |
| Figure 4 | `paper/figs/emb_tsne_ic.pdf` | 12 | `analysis/emb_vis2.py` |
| Figure 5 | `paper/figs/fig_case.tex` | 12 | `analysis/make_case.py` |
| Table 4 | `T_link.tex` | 04, 05, 06, 12 | `analysis/link_an_lg.py` |
| Table 5 | 매크로 `\kcst…` | 06, 12 | `analysis/story_an.py` |
| Table 6 | 매크로 `\kcctl…`, `\kcgctl…`, `\icctl…`, `\icgctl…` | 01, 02, 06, 12 | `analysis/link_an_lg.py` |
| Table 7 | `T_degree.tex` | 12 | `analysis/link_an_lg.py` |
| Table 8 | 매크로 `\kcabl…`, `\icabl…` | 07 | `analysis/ablation_an.py` |
| Table A1 | 매크로 `…Afive`, `…Ffive` | 04, 05, 09 | `tables/make_tables.py`, `baselines/base_full_eval.py` |
| Table A2 | `T_sig.tex` | 04, 05 | `tables/make_tables.py` |
| Table A3 | `T_unmasked.tex` | 04, 05, 09 | `tables/make_tables.py`, `baselines/base_full_eval.py` |
| Table A4 | `T_link_unmasked.tex` | 12 | `analysis/link_an_lg.py` |
| Table A5 | `T_control_full.tex` | 06, 12 | `analysis/link_an_lg.py` |
| Table A6 | 매크로 `\kcst…` | 06, 12 | `analysis/story_an.py` |
| Table A7 | `T_edges.tex` | 01, 10 | `graphs/drop_edges.py`, `analysis/edge_removal_an.py` |
| Table A8 | `T_fsens.tex` | 01, 08 | `graphs/build_graph_katchers.py` |
| Table A9 | `T_ic_enc.tex` | 03, 11 | `graphs/text_electra_en.py`, `tables/make_tables.py` |
| 본문 수치 (Holm 보정, 집단 비율, 주문 크기별 비교, 토큰 통계, 파라미터 수 등) | `paper/numbers.tex`, `paper/tables/holm.json` | 12, 13 | `tables/make_tables.py`, `analysis/extra_an2.py`, `analysis/extra_ordsize.py`, `analysis/tok_stats.py` |

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
