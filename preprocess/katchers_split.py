"""Katcher's: filtering of the raw purchase log as in [1] and the user-level 60/20/20 split -> train_data.csv, val_data.csv, test_data.csv,
the input of preprocess/katchers.py.

Input: raw_data_katchers.csv (tab-separated) from the repository of [1], https://github.com/msparkDev/ConcurrentBERT-Recommender
usage: python preprocess/katchers_split.py --raw <raw_data_katchers.csv> --out-dir <folder>
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

p = argparse.ArgumentParser()
p.add_argument("--raw", required=True, help="raw_data_katchers.csv (tab-separated)")
p.add_argument("--out-dir", required=True, help="train_data.csv, val_data.csv and test_data.csv are written here")
args = p.parse_args()

# =========================
# 1. paths
# =========================
raw_path = args.raw
save_dir = args.out_dir
os.makedirs(save_dir, exist_ok=True)

# =========================
# 2. raw data
# =========================
data = pd.read_csv(raw_path, sep="\t", encoding="utf-8", low_memory=False)

print("raw data:", data.shape)
print(data.columns.tolist())

# =========================
# 3. filtering as in [1]
# =========================

# Step 1: users with purchases on at least two dates
data["unique_purchase_dates"] = data.groupby("user_id")["initial_paid_at"].transform(lambda x: x.nunique())
data = data[data["unique_purchase_dates"] > 1].drop(columns=["unique_purchase_dates"])

# Step 2: most recent purchase date of each user
df_grouped = data.groupby("user_id")["initial_paid_at"].max().reset_index(name="initial_paid_at")

# Step 3: all items bought on the most recent purchase date
df_max_all = data.merge(df_grouped, how="inner", on=["user_id", "initial_paid_at"])

# Step 4: one target item per user, drawn at random from the items of the most recent purchase date
df_max = df_max_all.groupby("user_id").sample(n=1, random_state=42).reset_index(drop=True)

# Step 5: purchases on the other dates
df_joined = data.merge(df_grouped, how="inner", on="user_id", suffixes=["", "_y"])
df_joined["initial_paid_at_mismatch"] = df_joined["initial_paid_at"] != df_joined["initial_paid_at_y"]

df_not_max = df_joined[df_joined["initial_paid_at_mismatch"]].drop(
    columns=["initial_paid_at_y", "initial_paid_at_mismatch"]
)

# Step 6: the target item and the earlier purchases
df_processed = pd.concat([df_max, df_not_max]).reset_index(drop=True)

# Step 7: users with more than 1 and fewer than 68 purchased items
df_processed["total_order_cnt"] = df_processed.groupby("user_id")["product_id"].transform("count")
df_processed = df_processed[
    (df_processed["total_order_cnt"] > 1) &
    (df_processed["total_order_cnt"] < 68)
].drop(columns=["total_order_cnt"]).reset_index(drop=True)

print("after filtering:", df_processed.shape)
print("users:", df_processed["user_id"].nunique())
print("items:", df_processed["product_id"].nunique())

# =========================
# 4. user-level split as in [1]
# =========================

# first split: train+val 80%, test 20%
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_val_idx, test_idx = next(
    gss.split(df_processed, groups=df_processed["user_id"])
)

train_val_data = df_processed.iloc[train_val_idx].reset_index(drop=True)
test_data = df_processed.iloc[test_idx].reset_index(drop=True)

# second split: train 75%, val 25% of train+val
# overall: train 60%, val 20%, test 20%
gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
train_idx, val_idx = next(
    gss.split(train_val_data, groups=train_val_data["user_id"])
)

train_data = train_val_data.iloc[train_idx].reset_index(drop=True)
val_data = train_val_data.iloc[val_idx].reset_index(drop=True)

# =========================
# 5. save
# =========================
train_data.to_csv(os.path.join(save_dir, "train_data.csv"), index=False, encoding="utf-8-sig")
val_data.to_csv(os.path.join(save_dir, "val_data.csv"), index=False, encoding="utf-8-sig")
test_data.to_csv(os.path.join(save_dir, "test_data.csv"), index=False, encoding="utf-8-sig")

# =========================
# 6. checks
# =========================
print("\nsaved")
print("train:", train_data.shape, "users:", train_data["user_id"].nunique())
print("val  :", val_data.shape, "users:", val_data["user_id"].nunique())
print("test :", test_data.shape, "users:", test_data["user_id"].nunique())

train_users = set(train_data["user_id"])
val_users = set(val_data["user_id"])
test_users = set(test_data["user_id"])

print("\nuser overlap")
print("train & val :", len(train_users & val_users))
print("train & test:", len(train_users & test_users))
print("val & test  :", len(val_users & test_users))

print("\nsaved to:", save_dir)
