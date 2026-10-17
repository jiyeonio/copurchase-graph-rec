"""Locations of the data and of the result files. The Python scripts import this file; config.sh holds the same values for the shell
scripts. The defaults are folders of this repository; override the roots with the environment variables KATCHERS_SPLIT, KATCHERS_DATA,
INSTACART_DATA, RESULTS and PAPER."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _dir(var, default):
    return os.path.join(os.path.abspath(os.path.expanduser(os.environ.get(var, os.path.join(ROOT, default)))), "")


KC_SPLIT = _dir("KATCHERS_SPLIT", "data/katchers_split")   # Katcher's train/val/test_data.csv: output of preprocess/katchers_split.py
KC = _dir("KATCHERS_DATA", "data/katchers")       # Katcher's data: output of preprocess/katchers.py
IC = _dir("INSTACART_DATA", "data/instacart")     # Instacart data: output of preprocess/instacart.py, text and graph embeddings
RES = _dir("RESULTS", "results")                  # everything the scripts write
PAPER = _dir("PAPER", "paper")                    # tables/, figs/, numbers.tex

RUNS = RES + "runs/"                         # <RUNS>{kc11,ic11}/<run>/result_U_seed{S}.json, checkpoints
AN = RES + "analysis/"                       # analysis files of the main runs
PU, PU2, RK = AN + "pu/", AN + "pu2/", AN + "rank/"       # per-user dumps: main runs / controls and variants / ranks
RUNS_EDGE, PU_EDGE = RES + "runs_edge/", AN + "pu_edge/"   # removal of edges
RUNS_ENC, PU_ENC = RES + "runs_en/", AN + "pu_en/"         # English encoder on Instacart
EXTRA, EXTRA_ENC = RES + "extra/", RES + "extra_en/"       # analysis files behind single statements
GRAPHS = RES + "graphs/"                     # Katcher's graphs and graph embeddings
IC_GRAPHS = RES + "ic_graphs/"               # Instacart rewired graphs and their embeddings
BASELINES = RES + "baselines_full/"
KC_META = KC                                 # Katcher's item_metadata.csv
KC_TEXT = RES + "text/kc/item_embeddings_text.npy"
IC_TEXT = IC + "item_embeddings_text.npy"
IC_TEXT_EN = RES + "text/ic_en/item_embeddings_text.npy"

SEEDS = [42, 43, 44, 45, 46]
N_ITEMS = {"kc": 1608, "ic": 3542}
FMIN = {"kc": 1, "ic": 3}                    # minimum pair count of the co-purchase graph
BOOT_N, BOOT_SEED = 10000, 2026              # paired bootstrap over users
