"""
routers/recommend.py — Endpoint POST /recommend
"""
import time
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src" / "05_graphrag"))

from schemas import (
    RecommendRequest, RecommendResponse,
    OffreRecommandee, SkillGapAnalysis, Roadmap, RoadmapEtape
)
from dependencies import get_engine, get_st_model

log = logging.getLogger(__name__)
router = APIRouter()


def _parse_offre(raw: dict, rang: int) -> OffreRecommandee:
    """Convertit un dict brut du moteur en OffreRecommandee Pydantic."""
    return OffreRecommandee(
        rang=rang,
        offre_id=raw.get("offre_id", ""),
        titre_poste=raw.get("titre", raw.get("titre_poste", "")),
        secteur=raw.get("secteur"),
        ville=raw.get("ville"),
        type_contrat=raw.get("type_contrat", raw.get("contrat")),
        score_hybride=round(float(raw.get("score_hybride", 0)), 4),
        score_semantique=round(float(raw.get("score_sem", 0)), 4),
        score_graphe=round(float(raw.get("taux_match", 0)), 4),
        taux_matching=round(float(raw.get("taux_match", 0)), 4),
        competences_acquises=raw.get("acquises", [])[:6],
        competences_manquantes=raw.get("manquantes", [])[:6],
        essentielles_manquantes=raw.get("ess_manq", [])[:4],
    )


def _parse_skill_gap(raw: dict) -> Optional[SkillGapAnalysis]:
    if not raw:
        return None
    return SkillGapAnalysis(
        taux_matching=raw.get("taux_matching", 0.0),
        niveau_gap=raw.get("niveau_gap", "modéré"),
        eligible_maintenant=bool(raw.get("eligible_maintenant", False)),
        competences_critiques=raw.get("competences_critiques", [])[:5],
        score_projete_apres_formation=raw.get("score_projete_apres_formation"),
        message_candidat=raw.get("message_candidat"),
    )


def _parse_roadmap(raw: dict) -> Optional[Roadmap]:
    if not raw:
        return None
    etapes = []
    for e in raw.get("etapes", [])[:8]:
        form = e.get("formation", {})
        etapes.append(RoadmapEtape(
            priorite=e.get("priorite", 0),
            competence_cible=e.get("competence_cible", ""),
            importance=e.get("importance", "optionnelle"),
            formation_nom=form.get("nom"),
            etablissement=form.get("etablissement"),
            duree=form.get("duree"),
            cout_estimatif=form.get("cout_estimatif"),
            modalite=form.get("modalite"),
            delai_acquisition=e.get("delai_acquisition"),
            impact_score=e.get("impact_score"),
        ))
    return Roadmap(
        poste_cible=raw.get("poste_cible", ""),
        score_matching_actuel=raw.get("score_matching_actuel", 0.0),
        score_matching_projete=raw.get("score_matching_projete", 0.0),
        duree_totale_estimee=raw.get("duree_totale_estimee"),
        etapes=etapes,
        ressources_gratuites=raw.get("ressources_gratuites", [])[:5],
        certifications_utiles=raw.get("certifications_utiles", [])[:5],
        conseil_candidature_immediate=raw.get("conseil_candidature_immediate"),
        message_motivation=raw.get("message_motivation"),
    )


@router.post(
    "/",
    response_model=RecommendResponse,
    summary="Recommandation d'offres d'emploi",
    description="""
Retourne les top-k offres d'emploi recommandées pour un candidat,
avec score hybride (sémantique + graphe + collaboratif), analyse du skill gap
et roadmap de formation personnalisée (formations NCF camerounaises).

Le pipeline GraphRAG :
1. ANN pgvector → top-20 offres sémantiquement proches
2. Cypher Neo4j → skill gap exact + compatibilité NCF
3. Score hybride = 0.40×sémantique + 0.35×graphe + 0.25×collaboratif
4. LLM 2 (Mistral-7B / GPT-4o) → génération recommandations + roadmap
    """,
    responses={
        200: {"description": "Recommandations générées avec succès"},
        404: {"description": "Candidat non trouvé"},
        503: {"description": "Service indisponible (moteur non initialisé)"},
    },
)
async def recommend(
    request: RecommendRequest,
    engine=Depends(get_engine),
):
    t0 = time.time()
    try:
        # Changer le top_k et le backend à la volée si demandé
        engine.top_k = request.top_k
        result = engine.recommend(request.profil.candidat_id)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Erreur recommendation : {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Erreur moteur : {str(e)}")

    latence_ms = round((time.time() - t0) * 1000, 1)

    # Parser les offres
    top_offres = [
        _parse_offre(o, rang=i + 1)
        for i, o in enumerate(result.get("top_offres", []))
    ]

    # Extraire les métadonnées LLM
    rec_data = result.get("recommandations", {})

    return RecommendResponse(
        candidat_id=request.profil.candidat_id,
        n_offres_analysees=result.get("n_offres_ann", len(top_offres)),
        top_offres=top_offres,
        analyse_globale=rec_data.get("analyse_globale"),
        score_employabilite_global=rec_data.get("score_employabilite_global"),
        conseil_global=rec_data.get("conseil_global"),
        prochaine_action=rec_data.get("prochaine_action"),
        skill_gap=_parse_skill_gap(result.get("skill_gap", {}))
            if request.include_skill_gap else None,
        roadmap=_parse_roadmap(result.get("roadmap", {}))
            if request.include_roadmap else None,
        latence_ms=latence_ms,
        llm_backend="ollama:qwen2:1.5b",
    )


@router.get(
    "/candidat/{candidat_id}",
    response_model=RecommendResponse,
    summary="Recommandation rapide par ID candidat (GET)",
)
async def recommend_by_id(
    candidat_id: str,
    top_k: int = Query(5, ge=1, le=20),
    include_roadmap: bool = Query(True),
    engine=Depends(get_engine),
):
    """Version GET simplifiée — profil chargé depuis la base Parquet."""
    t0 = time.time()
    try:
        engine.top_k = top_k
        result = engine.recommend(candidat_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    latence_ms = round((time.time() - t0) * 1000, 1)
    top_offres = [_parse_offre(o, i + 1) for i, o in enumerate(result.get("top_offres", []))]
    rec_data = result.get("recommandations", {})

    return RecommendResponse(
        candidat_id=candidat_id,
        n_offres_analysees=result.get("n_offres_ann", 0),
        top_offres=top_offres,
        analyse_globale=rec_data.get("analyse_globale"),
        score_employabilite_global=rec_data.get("score_employabilite_global"),
        conseil_global=rec_data.get("conseil_global"),
        prochaine_action=rec_data.get("prochaine_action"),
        roadmap=_parse_roadmap(result.get("roadmap", {})) if include_roadmap else None,
        latence_ms=latence_ms,
        llm_backend="ollama:qwen2:1.5b",
    )
