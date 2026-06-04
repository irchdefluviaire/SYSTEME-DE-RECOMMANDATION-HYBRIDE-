"""
normalize_candidats.py — Pipeline ETL complet pour le dataset des demandeurs d'emploi

Stratégie de nettoyage :
  1. Nettoyage de base (strip, espaces insécables)
  2. Audit Matricule sans suppression de lignes
  3. Mapping Niveau Étude → code NCF (1-9)
  4. Mapping Diplôme → code NCF (confirmé / priorité sur niveau_etude)
  5. Normalisation Mobilité géographique → booléen / liste villes
  6. Normalisation Secteur demandé (valeurs "Non déclaré" → None)
  7. Construction text_to_embed côté requête (format metadata)
  8. candidat_id technique unique, matricule_raw conservé
  9. Sauvegarde en Parquet et rapports qualité/audit
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEMANDEUR_RAW, DEMANDEUR_PROC, DATA_PROC,
    NIVEAU_ETUDES_CAND_TO_NCF, DIPLOME_TO_NCF,
    FT_MAX_META_CHARS,
)
from utils import (
    clean_whitespace, normalize_ville,
    profil_qualite, log_etape, log,
)


# ─────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────

def load_raw(path=DEMANDEUR_RAW) -> pd.DataFrame:
    log.info(f"Chargement demandeurs : {path}")
    df = pd.read_excel(path)
    log.info(f"  → {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────────────────────────────
# ÉTAPE 1 — NETTOYAGE DE BASE
# ─────────────────────────────────────────────────────────────────

def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "source_row_number", range(1, len(df) + 1))
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].apply(
            lambda x: clean_whitespace(str(x)) if pd.notna(x) else pd.NA
        )
        df[col] = df[col].replace("", pd.NA).replace("nan", pd.NA)
    return df


def audit_matricules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marque les matricules dupliques sans supprimer de lignes.

    Le matricule reste conserve comme identifiant metier brut. L'identifiant
    technique candidat_id est rendu unique plus loin dans finalize_schema.
    """
    df = df.copy()
    if "Matricule" not in df.columns:
        raise ValueError("Colonne Matricule absente du fichier demandeur")

    valid = df["Matricule"].notna()
    counts = df.loc[valid].groupby("Matricule")["Matricule"].transform("size")
    ranks = df.loc[valid].groupby("Matricule").cumcount() + 1

    df["matricule_duplicate_count"] = 1
    df.loc[valid, "matricule_duplicate_count"] = counts.astype("int64")
    df["matricule_duplicate_rank"] = 1
    df.loc[valid, "matricule_duplicate_rank"] = ranks.astype("int64")
    df["is_matricule_duplicate"] = df["matricule_duplicate_count"].gt(1)

    log.info(
        "  Doublons Matricule audites : "
        f"{int(df['is_matricule_duplicate'].sum())} lignes; aucune ligne supprimee"
    )
    return df


# ─────────────────────────────────────────────────────────────────
# ÉTAPE 2 — MAPPING NIVEAU ÉTUDES → CODE NCF
# ─────────────────────────────────────────────────────────────────

def map_ncf_niveau(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Depuis niveau_etude (champ structuré)
    df["ncf_code_niveau_etude"] = (
        df["niveau_etude"]
        .map(NIVEAU_ETUDES_CAND_TO_NCF)
        .astype("Int64")
    )

    # Depuis Diplome (plus précis, priorité)
    df["ncf_code_diplome"] = (
        df["Diplome"]
        .map(DIPLOME_TO_NCF)
        .astype("Int64")
    )

    # Code NCF final : diplome en priorité, niveau_etude en fallback
    df["ncf_niveau_final"] = df["ncf_code_diplome"].where(
        df["ncf_code_diplome"].notna(),
        df["ncf_code_niveau_etude"]
    )

    log.info(f"  NCF code résolu : {df['ncf_niveau_final'].notna().sum()} / {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────
# ÉTAPE 3 — MOBILITÉ GÉOGRAPHIQUE
# ─────────────────────────────────────────────────────────────────

def normalize_mobilite(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Mobilité géographique' : 'Oui', 'Non', 'Non déclaré'
    → booléen + liste de villes acceptées (si disponible)
    """
    df = df.copy()

    def parse_mobilite(val):
        if not isinstance(val, str):
            return None, []
        val_low = val.strip().lower()
        if val_low == "oui":
            return True, []  # mobile partout
        if val_low == "non":
            return False, []  # pas mobile
        if "non déclaré" in val_low or val_low == "":
            return None, []  # inconnu
        # Cas rare : liste de villes "Oui - Douala, Yaoundé"
        if "oui" in val_low:
            parts = val.split("-", 1)
            if len(parts) > 1:
                villes = normalize_ville(parts[1])
                return True, villes
        return None, []

    parsed = df["Mobilité géographique"].apply(parse_mobilite)
    df["mobilite_geo_bool"]  = parsed.apply(lambda x: x[0])
    df["mobilite_geo_villes"] = parsed.apply(lambda x: x[1])

    log.info(f"  Mobilité : Oui={df['mobilite_geo_bool'].eq(True).sum()}, "
             f"Non={df['mobilite_geo_bool'].eq(False).sum()}, "
             f"Inconnu={df['mobilite_geo_bool'].isna().sum()}")
    return df


# ─────────────────────────────────────────────────────────────────
# ÉTAPE 4 — NORMALISATION DES CHAMPS "NON DÉCLARÉ"
# ─────────────────────────────────────────────────────────────────

def normalize_non_declare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs 'Non déclaré' par pd.NA pour les champs :
    Secteur demandé, Mobilité géographique (déjà géré),
    et autres champs avec cette sentinelle.
    """
    df = df.copy()
    NON_DECLARE_VALUES = {"Non déclaré", "non déclaré", "Non Déclaré", "Divers "}

    for col in ["Secteur demandé", "Mobilité géographique"]:
        if col in df.columns:
            df[col] = df[col].where(
                ~df[col].isin(NON_DECLARE_VALUES), other=pd.NA
            )

    return df


# ─────────────────────────────────────────────────────────────────
# ÉTAPE 5 — SCHÉMA FINAL + RENOMMAGE
# ─────────────────────────────────────────────────────────────────

def finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Renommage → schéma normalisé
    rename_map = {
        "Matricule":                           "matricule_raw",
        "Age":                                  "age",
        "Qualification":                        "qualification_declaree",
        "Secteur d'activité":                  "secteur_activite_cand",
        "Objectif":                             "objectif",
        "Diplome":                              "diplome_raw",
        "Genre":                                "genre",
        "niveau_etude":                         "niveau_etude_raw",
        "qualification_metier":                 "qualification_metier",
        "secteur_metier":                       "secteur_metier",
        "Filière / Spécialité":                 "filiere_specialite",
        "Secteur demandé":                      "secteur_demande",
        "Métier visé / Qualification visée":    "metier_vise",
        "Mobilité géographique":                "mobilite_geo_raw",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["candidat_id"] = df.apply(_build_candidat_id, axis=1)

    # Ordonnancement final
    cols = [
        "candidat_id",
        "matricule_raw",
        "source_row_number",
        "age",
        "genre",
        "diplome_raw",
        "ncf_niveau_final",
        "ncf_code_niveau_etude",
        "ncf_code_diplome",
        "niveau_etude_raw",
        "qualification_declaree",
        "qualification_metier",
        "secteur_metier",
        "secteur_activite_cand",
        "filiere_specialite",
        "secteur_demande",
        "metier_vise",
        "objectif",
        "mobilite_geo_bool",
        "mobilite_geo_villes",
        "mobilite_geo_raw",
        "matricule_duplicate_count",
        "matricule_duplicate_rank",
        "is_matricule_duplicate",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def _build_candidat_id(row: pd.Series) -> str:
    matricule = row.get("matricule_raw")
    if pd.isna(matricule) or str(matricule).strip() == "":
        return f"CAND_ROW_{int(row['source_row_number']):06d}"

    candidat_id = str(matricule).strip()
    if int(row.get("matricule_duplicate_count", 1)) > 1:
        candidat_id = f"{candidat_id}__OCC{int(row['matricule_duplicate_rank']):02d}"
    return candidat_id


# ─────────────────────────────────────────────────────────────────
# ÉTAPE 6 — TEXT_TO_EMBED (côté requête ST fine-tuning)
# ─────────────────────────────────────────────────────────────────

def build_text_to_embed_candidat(row: pd.Series) -> str:
    """
    Construit le texte côté requête pour le fine-tuning et les embeddings pgvector.
    Format métadonnées structurées — aligné sur sentence1 du fine-tuning ST.
    """
    parts = []

    if pd.notna(row.get("metier_vise")):
        parts.append(f"Poste: {row['metier_vise']}")

    if pd.notna(row.get("secteur_metier")):
        parts.append(f"Secteur: {row['secteur_metier']}")
    elif pd.notna(row.get("secteur_activite_cand")):
        parts.append(f"Secteur: {row['secteur_activite_cand']}")

    if pd.notna(row.get("ncf_niveau_final")):
        parts.append(f"Niveau_NCF: {row['ncf_niveau_final']}")

    if pd.notna(row.get("niveau_etude_raw")):
        parts.append(f"Études: {row['niveau_etude_raw']}")

    if pd.notna(row.get("filiere_specialite")):
        parts.append(f"Filière: {row['filiere_specialite']}")

    if pd.notna(row.get("qualification_metier")):
        parts.append(f"Qualification: {row['qualification_metier']}")

    # Objectif en texte libre — apporte contexte sémantique
    objectif_str = ""
    if pd.notna(row.get("objectif")):
        objectif_str = f". {row['objectif']}"

    return (" | ".join(parts) + objectif_str).strip()[:FT_MAX_META_CHARS]


def add_embed_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_to_embed"] = df.apply(build_text_to_embed_candidat, axis=1)
    log.info(f"  text_to_embed moyen : {df['text_to_embed'].str.len().mean():.0f} chars")
    return df


def save_matricule_audit(df: pd.DataFrame) -> None:
    audit_path = DATA_PROC / "rapport_audit_matricules_candidats.csv"
    audit_cols = [
        "matricule_raw",
        "matricule_duplicate_count",
        "matricule_duplicate_rank",
        "source_row_number",
        "candidat_id",
        "age",
        "genre",
        "diplome_raw",
        "metier_vise",
        "secteur_demande",
    ]
    audit = df.loc[df["is_matricule_duplicate"], [c for c in audit_cols if c in df.columns]]
    audit.to_csv(audit_path, index=False)
    log.info(f"Rapport audit matricules sauvegardé → {audit_path} ({len(audit)} lignes)")


# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def run(save=True) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("PIPELINE ETL — DEMANDEURS D'EMPLOI")
    log.info("=" * 60)

    df = load_raw()
    profil_qualite(df, "demandeurs_raw")

    df = clean_base(df)
    log.info(f"[1/6] Nettoyage de base terminé")

    df = audit_matricules(df)
    log.info("  Audit Matricule terminé")

    df = normalize_non_declare(df)
    log.info(f"[2/6] Valeurs 'Non déclaré' normalisées")

    df = map_ncf_niveau(df)
    log.info(f"[3/6] Mapping NCF terminé")

    df = normalize_mobilite(df)
    log.info(f"[4/6] Mobilité géographique normalisée")

    df = finalize_schema(df)
    log.info(f"[5/6] Schéma finalisé")

    df = add_embed_text(df)
    log.info(f"[6/6] Textes d'embedding construits")

    if save:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DEMANDEUR_PROC, index=False)
        log.info(f"Sauvegardé → {DEMANDEUR_PROC}")

        rapport = profil_qualite(df, "candidats_processed")
        rapport.to_csv(DATA_PROC / "rapport_qualite_candidats.csv", index=False)
        save_matricule_audit(df)

    log.info(f"✓ Candidats traités : {len(df)} lignes")
    return df


if __name__ == "__main__":
    run()
