"""
routers/skill_gap.py — Endpoint POST /skill-gap
"""
import time, logging, sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src" / "05_graphrag"))

from schemas import SkillGapRequest, SkillGapResponse, SkillGapAnalysis, Roadmap, RoadmapEtape
from dependencies import get_engine

log    = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=SkillGapResponse,
    summary="Analyse du skill gap entre un candidat et une offre",
    description="""
Calcule l'écart de compétences entre un candidat et une offre spécifique.

Sources :
- **Neo4j** : intersection exacte des URIs ESCO (acquis / manquant)
- **pgvector** : proximité sémantique des compétences partiellement correspondantes
- **LLM 2** : synthèse narrative + estimation du délai d'acquisition

Retourne :
- Taux de matching (exact + hiérarchique)
- Liste des compétences acquises / manquantes / essentielles manquantes
- Roadmap de formation personnalisée (formations NCF camerounaises)
""",
    responses={
        200: {"description": "Analyse skill gap réalisée"},
        404: {"description": "Candidat ou offre introuvable"},
    },
)
async def skill_gap_analysis(
    request: SkillGapRequest,
    engine=Depends(get_engine),
):
    t0 = time.time()

    try:
        # Utiliser le context builder du moteur
        candidat = engine._load_candidat(request.candidat_id)
        ctx = engine.builder.build_context(request.candidat_id, candidat)

        # Trouver l'offre demandée parmi le top ou lancer une recherche dédiée
        offre = next(
            (o for o in ctx["top_offres"] if o["offre_id"] == request.offre_id),
            ctx["top_offres"][0] if ctx["top_offres"] else None,
        )
        if not offre:
            raise ValueError(f"Offre {request.offre_id} introuvable")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # LLM 2 → Skill Gap
    from prompt_templates import SYSTEM_SKILL_GAP, USER_SKILL_GAP
    user_sg = USER_SKILL_GAP.format(
        candidat_id=request.candidat_id,
        metier_vise=candidat.get("metier_vise", ""),
        ncf_niveau=candidat.get("ncf_niveau_final", "?"),
        titre_offre=offre.get("titre", ""),
        secteur=offre.get("secteur", ""),
        n_acquises=len(offre.get("acquises", [])),
        n_manquantes=len(offre.get("manquantes", [])),
        acquises_list="\n".join(f"  - {c}" for c in offre.get("acquises", [])[:5]),
        manquantes_list="\n".join(f"  - {c}" for c in offre.get("manquantes", [])[:5]),
        taux_match=offre.get("taux_match", 0.5),
    )

    raw_sg  = engine.llm.generate(SYSTEM_SKILL_GAP, user_sg)
    sg_data = engine._safe_parse_json(raw_sg) or {}

    # LLM 2 → Roadmap
    from prompt_templates import SYSTEM_ROADMAP, USER_ROADMAP, get_formations
    import json
    user_rm = USER_ROADMAP.format(
        candidat_profile=json.dumps(candidat, ensure_ascii=False)[:600],
        offre_profile=json.dumps({"titre": offre.get("titre",""), "secteur": offre.get("secteur","")},
                                  ensure_ascii=False),
        competences_manquantes="\n".join(f"  - {c}" for c in offre.get("manquantes", [])[:5]),
        score_actuel=offre.get("score_hybride", 0.5),
        score_projete=sg_data.get("score_projete_apres_formation",
                                   min(offre.get("score_hybride", 0.5) + 0.2, 0.95)),
    )
    raw_rm  = engine.llm.generate(SYSTEM_ROADMAP, user_rm)
    rm_data = engine._safe_parse_json(raw_rm) or {}

    latence_ms = round((time.time() - t0) * 1000, 1)

    # Construire Roadmap Pydantic
    etapes = []
    for e in rm_data.get("etapes", [])[:6]:
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

    return SkillGapResponse(
        candidat_id=request.candidat_id,
        offre_id=request.offre_id,
        analyse=SkillGapAnalysis(
            taux_matching=sg_data.get("taux_matching", offre.get("taux_match", 0)),
            niveau_gap=sg_data.get("niveau_gap", "modéré"),
            eligible_maintenant=bool(sg_data.get("eligible_maintenant", False)),
            competences_critiques=sg_data.get("competences_critiques", [])[:5],
            score_projete_apres_formation=sg_data.get("score_projete_apres_formation"),
            message_candidat=sg_data.get("message_candidat"),
        ),
        roadmap=Roadmap(
            poste_cible=rm_data.get("poste_cible", offre.get("titre", "")),
            score_matching_actuel=rm_data.get("score_matching_actuel", 0.0),
            score_matching_projete=rm_data.get("score_matching_projete", 0.0),
            duree_totale_estimee=rm_data.get("duree_totale_estimee"),
            etapes=etapes,
            ressources_gratuites=rm_data.get("ressources_gratuites", [])[:5],
            certifications_utiles=rm_data.get("certifications_utiles", [])[:3],
            conseil_candidature_immediate=rm_data.get("conseil_candidature_immediate"),
            message_motivation=rm_data.get("message_motivation"),
        ) if rm_data else None,
        latence_ms=latence_ms,
    )
