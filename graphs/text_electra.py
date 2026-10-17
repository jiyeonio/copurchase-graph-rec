"""Text representation of the items: fixed [CLS] embedding of the item description, 768-d, no fine-tuning.
Both datasets: snunlp/KR-ELECTRA-discriminator, [CLS], max_length 64.
Input sentence: Katcher's = product name + root category + category + attribute values (item_metadata.csv); Instacart = text field of item_meta.json.
Run: python graphs/text_electra.py --dataset {katchers|instacart} --data-dir <folder with the metadata file> --emb-dir <output folder>
     -> <output folder>/item_embeddings_text.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# [CLS], max_length 64 for both datasets (batch size does not change the result beyond numerical noise)
TEXT_MODEL_NAME = "snunlp/KR-ELECTRA-discriminator"
TEXT_RAW_DIM = 768
TEXT_DEFAULTS = {"katchers": dict(max_length=64, batch_size=32),
                 "instacart": dict(max_length=64, batch_size=256)}


def katchers_texts(data_dir):
    item_info = pd.read_csv(
        data_dir / "item_metadata.csv"
    )
    item_info = (
        item_info
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    expected = list(range(len(item_info)))
    if (
        item_info["item_idx"]
        .astype(int)
        .tolist()
        != expected
    ):
        raise RuntimeError(
            "item_metadata.csv must be ordered by contiguous item_idx"
        )

    cols = [
        "product_name",
        "root_category_name",
        "category_name",
        "attribute_values",
    ]

    texts = []
    for _, row in item_info.iterrows():
        parts = [
            str(row[col])
            for col in cols
            if pd.notna(row[col])
        ]
        texts.append(" ".join(parts))

    return texts


def instacart_texts(data_dir):
    with (
        data_dir / "item_meta.json"
    ).open("r", encoding="utf-8") as f:
        item_meta = json.load(f)

    return [
        item_meta[str(i)]["text"]
        for i in range(len(item_meta))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["katchers", "instacart"],
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--emb-dir",
        type=Path,
        default=None,
        help="output directory for item_embeddings_text.npy (default: --data-dir)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--model-name",
        default=None,
    )
    args = parser.parse_args()

    d = TEXT_DEFAULTS[args.dataset]
    batch_size = args.batch_size if args.batch_size is not None else d["batch_size"]
    max_length = args.max_length if args.max_length is not None else d["max_length"]
    model_name = args.model_name if args.model_name is not None else TEXT_MODEL_NAME

    if args.dataset == "katchers":
        texts = katchers_texts(args.data_dir)
    else:
        texts = instacart_texts(args.data_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    model = AutoModel.from_pretrained(
        model_name
    ).to(device)
    model.eval()

    chunks = []
    for start in range(
        0, len(texts), batch_size
    ):
        batch = texts[
            start:start + batch_size
        ]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            chunks.append(
                outputs.last_hidden_state[:, 0, :]
                .cpu()
                .numpy()
            )

        print(
            f"encoded "
            f"{min(start + batch_size, len(texts))}"
            f"/{len(texts)}"
        )

    embeddings = np.vstack(chunks)
    expected_shape = (
        len(texts),
        TEXT_RAW_DIM,
    )
    if embeddings.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected text embedding shape: {embeddings.shape}, "
            f"expected {expected_shape}"
        )

    emb_dir = args.emb_dir if args.emb_dir is not None else args.data_dir
    emb_dir.mkdir(parents=True, exist_ok=True)
    output_path = emb_dir / "item_embeddings_text.npy"
    np.save(output_path, embeddings)
    print(
        f"saved {embeddings.shape} -> {output_path}"
    )


if __name__ == "__main__":
    main()
