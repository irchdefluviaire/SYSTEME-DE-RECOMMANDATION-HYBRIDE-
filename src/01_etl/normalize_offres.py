"""
normalize_offres.py - Pipeline ETL pour le dataset des offres d'emploi.

Strategie ETL retenue :
  1. Validation du schema source minimum.
  2. Nettoyage texte : espaces, valeurs vides explicites.
  3. Conservation de toutes les observations : aucune deduplication destructive.
  4. Audit des lignes strictement identiques sur toutes les variables source.
  5. Score de completude sur les champs structurants.
  6. Nettoyage du champ "Details de l'Annonce" : bruit de scraping.
  7. Normalisation multi-valeurs : villes, secteurs, competences.
  8. Mapping niveaux NCF, experience et contrats.
  9. Identifiant offre stable et schema final.
 10. Construction des textes d'embedding et rapports qualite.
"""

import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PROC,
    EXPERIENCE_TO_INT,
    FT_MAX_DESC_CHARS,
    FT_MAX_META_CHARS,
    FT_MIN_DETAILS_CHARS,
    FT_MIN_META_SKILLS_CHARS,
    GROUPE_CONTRAT_NORMALIZE,
    NIVEAU_ETUDES_OFFRES_TO_NCF,
    OFFRES_PROC,
    OFFRES_RAW,
    TYPE_CONTRAT_NORMALIZE,
)
from utils import (
    clean_details_annonce,
    clean_whitespace,
    log,
    log_etape,
    normalize_secteurs,
    normalize_skills,
    normalize_ville,
    profil_qualite,
)


REQUIRED_RAW_COLUMNS = [
    "Titre du Poste",
    "Employeur",
    "Ville / Région",
    "Secteur d'Activité",
    "Détails de l'Annonce",
]

QUALITY_COLUMNS = [
    "Titre du Poste",
    "Employeur",
    "Ville / Région",
    "Secteur d'Activité",
    "Niveau d'Études",
    "Niveau d'Expérience",
    "Compétences / Skills",
    "Détails de l'Annonce",
]


def load_raw(path=OFFRES_RAW) -> pd.DataFrame:
    log.info(f"Chargement offres : {path}")
    df = pd.read_excel(path, dtype=str)
    log.info(f"  -> {df.shape[0]} lignes, {df.shape[1]} colonnes")
    validate_raw_schema(df)
    return df


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Verifie que les champs minimums existent avant transformation."""
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Colonnes offres manquantes dans la source brute : "
            + ", ".join(missing)
        )


def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    """Strip + espaces insecables sur toutes les colonnes texte."""
    df = df.copy()
    df.insert(0, "source_row_number", range(1, len(df) + 1))

    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].apply(
            lambda x: clean_whitespace(x) if isinstance(x, str) else x
        )
        df[col] = df[col].replace("", pd.NA).replace("nan", pd.NA).replace("NaN", pd.NA)
    return df


def _canonical_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _stable_hash(parts: list[str], size: int = 16) -> str:
    payload = "||".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:size]


def _cell_for_duplicate_key(value) -> str:
    if value is None or pd.isna(value):
        return "<NA>"
    return str(value)


def audit_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marque les lignes strictement identiques sans supprimer de lignes.

    Une offre n'est candidate doublon que si toutes les variables source sont
    identiques apres le nettoyage de base. Le numero technique source_row_number
    est exclu de la comparaison.
    """
    n_before = len(df)
    df = df.copy()

    duplicate_key_columns = [
        col for col in df.columns
        if col != "source_row_number"
    ]
    key_parts = [
        df[col].map(_cell_for_duplicate_key)
        for col in duplicate_key_columns
    ]
    df["duplicate_candidate_key"] = (
        pd.concat(key_parts, axis=1)
        .agg("||".join, axis=1)
        .map(lambda value: _stable_hash([value], size=20))
    )

    valid_key = df["duplicate_candidate_key"].ne("")
    group_sizes = (
        df.loc[valid_key]
        .groupby("duplicate_candidate_key")["duplicate_candidate_key"]
        .transform("size")
    )
    group_ranks = df.loc[valid_key].groupby("duplicate_candidate_key").cumcount() + 1

    df["duplicate_candidate_count"] = 1
    df.loc[valid_key, "duplicate_candidate_count"] = group_sizes.astype("int64")
    df["duplicate_candidate_rank"] = 1
    df.loc[valid_key, "duplicate_candidate_rank"] = group_ranks.astype("int64")
    df["is_duplicate_candidate"] = df["duplicate_candidate_count"].gt(1)

    available_quality_cols = [col for col in QUALITY_COLUMNS if col in df.columns]
    df["etl_completeness_score"] = (
        df[available_quality_cols].notna().mean(axis=1).round(3)
        if available_quality_cols
        else 0.0
    )

    n_groups = int(
        df.loc[df["is_duplicate_candidate"], "duplicate_candidate_key"].nunique()
    )
    n_rows = int(df["is_duplicate_candidate"].sum())
    log.info(
        "Audit doublons exacts ligne complete : "
        f"{n_rows} lignes dans {n_groups} groupes; aucune ligne supprimee"
    )
    log_etape("Audit observations", pd.DataFrame(index=range(n_before)), df)
    return df


def clean_details(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le champ texte brut scrape."""
    df = df.copy()
    df["details_clean"] = df["Détails de l'Annonce"].apply(
        lambda x: clean_details_annonce(x) if isinstance(x, str) else ""
    )
    df["details_truncated"] = df["details_clean"].str[:FT_MAX_DESC_CHARS]
    log.info(
        f"  Details nettoyes : {(df['details_clean'] != '').sum()} / {len(df)} non vides"
    )
    return df


def explode_multivalues(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les champs multi-valeurs."""
    df = df.copy()

    df["villes_list"] = df["Ville / Région"].apply(
        lambda x: normalize_ville(x) if isinstance(x, str) else []
    )
    df["ville_principale"] = df["villes_list"].apply(lambda lst: lst[0] if lst else None)

    df["secteurs_list"] = df["Secteur d'Activité"].apply(
        lambda x: normalize_secteurs(x) if isinstance(x, str) else []
    )
    df["secteur_principal"] = df["secteurs_list"].apply(lambda lst: lst[0] if lst else None)

    df["skills_list"] = df["Compétences / Skills"].apply(
        lambda x: normalize_skills(x) if isinstance(x, str) else []
    )

    log.info(f"  Villes : {df['ville_principale'].notna().sum()} precisees")
    log.info(f"  Secteurs : {df['secteur_principal'].notna().sum()} precises")
    return df


def map_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ncf_niveau_code"] = (
        df["Niveau d'Études"]
        .map(NIVEAU_ETUDES_OFFRES_TO_NCF)
        .astype("Int64")
    )

    df["experience_min_ans"] = (
        df["Niveau d'Expérience"]
        .map(EXPERIENCE_TO_INT)
        .astype("Int64")
    )

    df["type_contrat_norm"] = (
        df["Type de Contrat"]
        .map(TYPE_CONTRAT_NORMALIZE)
        .fillna(df["Type de Contrat"])
    )

    df["groupe_contrat_norm"] = (
        df["Groupe de Contrat"]
        .map(GROUPE_CONTRAT_NORMALIZE)
    )

    df["type_entreprise_norm"] = df["Type d'Entreprise"].apply(
        lambda x: _normalize_type_entreprise(x) if isinstance(x, str) else None
    )

    return df


def _normalize_type_entreprise(val: str) -> str:
    val_low = val.lower()
    if "ong" in val_low or "international" in val_low:
        return "ONG/International"
    if "public" in val_low or "para" in val_low:
        return "Public/Para-public"
    return "Privé"


def _build_stable_offre_id(row: pd.Series) -> str:
    parts = [
        str(row.get("source_row_number", "")),
        _canonical_text(row.get("Source")),
        _canonical_text(row.get("Lien / Référence")),
        _canonical_text(row.get("Titre du Poste")),
        _canonical_text(row.get("Employeur")),
        _canonical_text(row.get("Ville / Région")),
        _canonical_text(row.get("Secteur d'Activité")),
        _canonical_text(row.get("Détails de l'Annonce"))[:500],
    ]
    return f"OFFRE_{int(row['source_row_number']):06d}_{_stable_hash(parts, size=12)}"


def finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme les colonnes vers le schema du projet, genere des IDs stables,
    selectionne et ordonne les colonnes finales.
    """
    df = df.copy()
    df["offre_id"] = df.apply(_build_stable_offre_id, axis=1)

    rename_map = {
        "Source": "source",
        "Lien / Référence": "lien_reference",
        "Titre du Poste": "titre_poste",
        "Employeur": "employeur",
        "Pays": "pays",
        "Groupe de Contrat": "groupe_contrat_raw",
        "Type de Contrat": "type_contrat_raw",
        "Niveau d'Expérience": "niveau_experience_raw",
        "Niveau d'Études": "niveau_etudes_raw",
        "Compétences / Skills": "skills_raw",
        "Détails de l'Annonce": "details_raw",
        "Type d'Entreprise": "type_entreprise_raw",
        "Secteur d'Activité": "secteur_activite_raw",
        "Ville / Région": "ville_region_raw",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    cols = [
        "offre_id",
        "source_row_number",
        "source",
        "titre_poste",
        "employeur",
        "type_entreprise_norm",
        "pays",
        "ville_principale",
        "villes_list",
        "secteur_principal",
        "secteurs_list",
        "groupe_contrat_norm",
        "type_contrat_norm",
        "ncf_niveau_code",
        "niveau_etudes_raw",
        "experience_min_ans",
        "niveau_experience_raw",
        "skills_list",
        "skills_raw",
        "details_clean",
        "details_truncated",
        "details_raw",
        "lien_reference",
        "groupe_contrat_raw",
        "type_contrat_raw",
        "secteur_activite_raw",
        "ville_region_raw",
        "type_entreprise_raw",
        "duplicate_candidate_key",
        "duplicate_candidate_count",
        "duplicate_candidate_rank",
        "is_duplicate_candidate",
        "etl_completeness_score",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def build_text_to_embed_offre(row: pd.Series) -> str:
    """Construit le texte corpus pour le fine-tuning du SentenceTransformer."""
    parts = []

    if isinstance(row.get("skills_list"), list) and row["skills_list"]:
        parts.append("Compétences requises : " + ", ".join(row["skills_list"]))

    if row.get("details_clean"):
        parts.append(row["details_clean"][:FT_MAX_DESC_CHARS])

    return " ".join(parts).strip()


def build_pair_query_offre(row: pd.Series) -> str:
    """Construit la future sentence1 : metadonnees + competences."""
    parts = []
    metadata = build_metadata_str_offre(row)
    if metadata:
        parts.append(metadata)
    if isinstance(row.get("skills_list"), list) and row["skills_list"]:
        parts.append("Competences: " + ", ".join(row["skills_list"]))
    return " | ".join(parts).strip()[:FT_MAX_META_CHARS]


def build_metadata_str_offre(row: pd.Series) -> str:
    """Construit le texte requete a partir des metadonnees structurees."""
    parts = []
    if pd.notna(row.get("titre_poste")):
        parts.append(f"Poste: {row['titre_poste']}")
    if pd.notna(row.get("secteur_principal")):
        parts.append(f"Secteur: {row['secteur_principal']}")
    if pd.notna(row.get("type_contrat_norm")):
        parts.append(f"Contrat: {row['type_contrat_norm']}")
    if pd.notna(row.get("niveau_etudes_raw")):
        parts.append(f"Études: {row['niveau_etudes_raw']}")
    if pd.notna(row.get("niveau_experience_raw")):
        parts.append(f"Expérience: {row['niveau_experience_raw']}")
    if pd.notna(row.get("ville_principale")):
        parts.append(f"Ville: {row['ville_principale']}")
    return " | ".join(parts)


def add_embed_texts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_to_embed"] = df.apply(build_text_to_embed_offre, axis=1)
    df["metadata_str"] = df.apply(build_metadata_str_offre, axis=1)
    df["pair_query_text"] = df.apply(build_pair_query_offre, axis=1)
    df["pair_details_text"] = df["details_clean"].fillna("").str[:FT_MAX_DESC_CHARS]
    df["ft_eligible"] = (
        df["skills_list"].apply(lambda value: isinstance(value, list) and len(value) > 0)
        & df["pair_query_text"].str.len().ge(FT_MIN_META_SKILLS_CHARS)
        & df["pair_details_text"].str.len().ge(FT_MIN_DETAILS_CHARS)
    )
    log.info(
        "  Paires FT eligibles: "
        f"{df['ft_eligible'].sum()} / {len(df)} "
        f"(min metadata+competences={FT_MIN_META_SKILLS_CHARS}, "
        f"min details={FT_MIN_DETAILS_CHARS})"
    )
    return df


def save_duplicate_audit(df: pd.DataFrame) -> None:
    audit_path = DATA_PROC / "rapport_audit_doublons_offres.csv"
    audit_cols = [
        "duplicate_candidate_key",
        "duplicate_candidate_count",
        "duplicate_candidate_rank",
        "source_row_number",
        "offre_id",
        "source",
        "lien_reference",
        "titre_poste",
        "employeur",
        "type_entreprise_raw",
        "pays",
        "groupe_contrat_raw",
        "type_contrat_raw",
        "niveau_experience_raw",
        "niveau_etudes_raw",
        "skills_raw",
        "details_raw",
        "ville_region_raw",
        "secteur_activite_raw",
        "etl_completeness_score",
    ]
    audit = df.loc[df["is_duplicate_candidate"], [c for c in audit_cols if c in df.columns]]
    audit.to_csv(audit_path, index=False)
    log.info(f"Rapport audit doublons sauvegarde -> {audit_path} ({len(audit)} lignes)")


def run(save=True) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("PIPELINE ETL - OFFRES D'EMPLOI")
    log.info("=" * 60)

    df = load_raw()
    profil_qualite(df, "offres_raw")

    df = clean_base(df)
    log.info("[1/8] Nettoyage de base termine")

    df = audit_observations(df)
    log.info("[2/8] Audit observations termine")

    df = clean_details(df)
    log.info("[3/8] Nettoyage details annonce termine")

    df = explode_multivalues(df)
    log.info("[4/8] Explosion multi-valeurs terminee")

    df = map_categorical(df)
    log.info("[5/8] Mapping categoriels termine")

    df = finalize_schema(df)
    log.info("[6/8] Schema finalise")

    df = add_embed_texts(df)
    log.info("[7/8] Textes d'embedding construits")

    if save:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OFFRES_PROC, index=False)
        log.info(f"Sauvegarde -> {OFFRES_PROC}")

        rapport_final = profil_qualite(df, "offres_processed")
        rapport_final.to_csv(DATA_PROC / "rapport_qualite_offres.csv", index=False)
        save_duplicate_audit(df)
        log.info("[8/8] Rapports qualite sauvegardes")

    log.info(f"OK - Offres traitees : {len(df)} lignes")
    return df


if __name__ == "__main__":
    run()
