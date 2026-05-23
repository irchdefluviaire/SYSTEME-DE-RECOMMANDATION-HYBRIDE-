"""
routers/offre.py — Endpoints GET /offre/{id} et GET /offres
"""
import logging
from typing import Optional, List
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from schemas import OffreDetail

log    = logging.getLogger(__name__)
router = APIRouter()

ROOT  = Path(__file__).resolve().parent.parent.parent.parent
_df_offres: Optional[pd.DataFrame] = None


def _get_offres() -> pd.DataFrame:
    """Cache du DataFrame offres (chargé une seule fois)."""
    global _df_offres
    if _df_offres is None:
        path = ROOT / "data" / "processed" / "offres_normalized.parquet"
        _df_offres = pd.read_parquet(path)
        log.info(f"Offres chargées en cache : {len(_df_offres):,}")
    return _df_offres


def _safe_list(val) -> List[str]:
    """Convertit une colonne (liste ou None) en list[str] propre."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    return [str(val)]


@router.get(
    "/{offre_id}",
    response_model=OffreDetail,
    summary="Détails d'une offre d'emploi",
    description="""
Retourne les détails complets d'une offre d'emploi par son UUID.

Inclut :
- Métadonnées structurées (titre, employeur, secteur, ville, contrat, NCF)
- Liste des compétences requises (skills_list)
- Description nettoyée (details_clean — boilerplate supprimé)
""",
    responses={404: {"description": "Offre introuvable"}},
)
async def get_offre(offre_id: str):
    df = _get_offres()
    row = df[df["offre_id"] == offre_id]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Offre introuvable : {offre_id}"
        )

    r = row.iloc[0]

    def safe_int(v):
        try: return int(v) if pd.notna(v) else None
        except: return None

    return OffreDetail(
        offre_id=str(r["offre_id"]),
        titre_poste=str(r.get("titre_poste", "") or ""),
        employeur=str(r.get("employeur", "") or "") or None,
        type_entreprise=str(r.get("type_entreprise_norm", "") or "") or None,
        secteur_principal=str(r.get("secteur_principal", "") or "") or None,
        secteurs_list=_safe_list(r.get("secteurs_list")),
        ville_principale=str(r.get("ville_principale", "") or "") or None,
        villes_list=_safe_list(r.get("villes_list")),
        type_contrat=str(r.get("type_contrat_norm", "") or "") or None,
        groupe_contrat=str(r.get("groupe_contrat_norm", "") or "") or None,
        ncf_niveau_code=safe_int(r.get("ncf_niveau_code")),
        niveau_etudes_raw=str(r.get("niveau_etudes_raw", "") or "") or None,
        experience_min_ans=safe_int(r.get("experience_min_ans")),
        skills_list=_safe_list(r.get("skills_list")),
        details_clean=str(r.get("details_clean", "") or "")[:1000] or None,
        metadata_str=str(r.get("metadata_str", "") or "") or None,
    )


@router.get(
    "/",
    response_model=List[OffreDetail],
    summary="Recherche d'offres (filtres simples)",
    description="Recherche d'offres par secteur, ville et niveau NCF.",
)
async def search_offres(
    secteur: Optional[str] = Query(None, description="Secteur d'activité"),
    ville: Optional[str] = Query(None, description="Ville principale"),
    ncf_min: Optional[int] = Query(None, ge=1, le=9, description="Niveau NCF minimum"),
    ncf_max: Optional[int] = Query(None, ge=1, le=9, description="Niveau NCF maximum"),
    limit: int = Query(20, ge=1, le=100),
):
    df = _get_offres().copy()

    if secteur:
        df = df[df["secteur_principal"].str.contains(secteur, case=False, na=False)]
    if ville:
        df = df[df["ville_principale"].str.contains(ville, case=False, na=False)]
    if ncf_min is not None:
        df = df[df["ncf_niveau_code"].fillna(0) >= ncf_min]
    if ncf_max is not None:
        df = df[df["ncf_niveau_code"].fillna(9) <= ncf_max]

    results = []
    for _, r in df.head(limit).iterrows():
        results.append(OffreDetail(
            offre_id=str(r["offre_id"]),
            titre_poste=str(r.get("titre_poste", "") or ""),
            employeur=str(r.get("employeur", "") or "") or None,
            secteur_principal=str(r.get("secteur_principal", "") or "") or None,
            ville_principale=str(r.get("ville_principale", "") or "") or None,
            type_contrat=str(r.get("type_contrat_norm", "") or "") or None,
            ncf_niveau_code=int(r["ncf_niveau_code"]) if pd.notna(r.get("ncf_niveau_code")) else None,
            skills_list=_safe_list(r.get("skills_list")),
        ))

    return results
