"""
build_pairs.py - Construction des paires metadata -> description pour le
fine-tuning SentenceTransformer.

Logique :
  - sentence1 = metadonnees structurees de l'offre cote requete ;
  - sentence2 = competences + description nettoyee de l'offre cote corpus ;
  - toutes les offres normalisees sont retenues pour les paires ;
  - split stratifie par secteur_principal (70/15/15).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_FT,
    FT_RANDOM_SEED,
    FT_TRAIN_RATIO,
    FT_VAL_RATIO,
    OFFRES_PROC,
    PAIRS_META,
    PAIRS_TEST,
    PAIRS_TRAIN,
    PAIRS_VAL,
)
from utils import log


def load_offres_processed() -> pd.DataFrame:
    log.info(f"Chargement offres processed : {OFFRES_PROC}")
    df = pd.read_parquet(OFFRES_PROC)
    log.info(f"  -> {len(df)} offres")
    return df


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les paires a partir du DataFrame offres normalise.
    Le filtre informatif sur les longueurs de texte est desactive.
    """
    df_ft = df[df["ft_eligible"] == True].copy()
    log.info(f"  Paires retenues sans filtre informatif : {len(df_ft)}")

    df_ft["sentence1"] = df_ft["metadata_str"].fillna("")
    df_ft["sentence2"] = df_ft["text_to_embed"].fillna("")

    return df_ft[
        [
            "offre_id",
            "sentence1",
            "sentence2",
            "secteur_principal",
            "titre_poste",
            "ville_principale",
        ]
    ]


def stratified_split(df: pd.DataFrame, seed: int = FT_RANDOM_SEED) -> tuple:
    """Split stratifie sur secteur_principal."""
    rng = np.random.default_rng(seed)

    train_rows, val_rows, test_rows = [], [], []

    for _, group in df.groupby("secteur_principal", dropna=False):
        idx = group.index.tolist()
        rng.shuffle(idx)

        n = len(idx)
        n_train = max(1, round(n * FT_TRAIN_RATIO))
        n_val = max(1, round(n * FT_VAL_RATIO))

        train_rows.extend(idx[:n_train])
        val_rows.extend(idx[n_train:n_train + n_val])
        test_rows.extend(idx[n_train + n_val:])

    train = df.loc[train_rows].reset_index(drop=True)
    val = df.loc[val_rows].reset_index(drop=True)
    test = df.loc[test_rows].reset_index(drop=True)

    log.info(f"  Split : train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "offre_id": row["offre_id"],
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "titre": row.get("titre_poste", ""),
                "secteur": row.get("secteur_principal", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"  Sauvegarde -> {path.name} ({len(df)} paires)")


def save_metadata(train, val, test, df_all):
    dist_secteurs = df_all["secteur_principal"].value_counts().to_dict()

    meta = {
        "total_paires": len(df_all),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "split_ratios": {
            "train": FT_TRAIN_RATIO,
            "val": FT_VAL_RATIO,
            "test": 1 - FT_TRAIN_RATIO - FT_VAL_RATIO,
        },
        "random_seed": FT_RANDOM_SEED,
        "min_s1_len": None,
        "min_s2_len": None,
        "filtre_informatif": False,
        "format": "JSONL - InputExample sentence-transformers",
        "sentence1_role": "metadata structurees cote requete",
        "sentence2_role": "skills_raw + details_clean cote corpus",
        "modele_cible": "all-MiniLM-L6-v2",
        "perte": "MultipleNegativesRankingLoss",
        "distribution_secteurs": dist_secteurs,
        "stats_longueur": {
            "s1_mean": round(df_all["sentence1"].str.len().mean(), 1),
            "s1_min": int(df_all["sentence1"].str.len().min()),
            "s1_max": int(df_all["sentence1"].str.len().max()),
            "s2_mean": round(df_all["sentence2"].str.len().mean(), 1),
            "s2_min": int(df_all["sentence2"].str.len().min()),
            "s2_max": int(df_all["sentence2"].str.len().max()),
        },
    }

    PAIRS_META.parent.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info(f"  Metadata sauvegardee -> {PAIRS_META.name}")

    return meta


def run(save=True) -> dict:
    log.info("=" * 60)
    log.info("PIPELINE ETL - PAIRES FINE-TUNING SENTENCETRANSFORMER")
    log.info("=" * 60)

    df_offres = load_offres_processed()
    df_pairs = build_pairs(df_offres)

    train, val, test = stratified_split(df_pairs)

    if save:
        save_jsonl(train, PAIRS_TRAIN)
        save_jsonl(val, PAIRS_VAL)
        save_jsonl(test, PAIRS_TEST)
        meta = save_metadata(train, val, test, df_pairs)
        log.info(
            f"\n  Resume : {meta['total_paires']} paires | "
            f"train={meta['n_train']} | val={meta['n_val']} | test={meta['n_test']}"
        )

    log.info("OK - Paires fine-tuning construites")
    return {"train": train, "val": val, "test": test, "all": df_pairs}


if __name__ == "__main__":
    run()
