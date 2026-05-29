"""
schemas.py â€” SchÃ©mas Pydantic v2 pour l'API FastAPI
Module 06 â€” SystÃ¨me de Recommandation Emploi-CompÃ©tences Â· Cameroun

Valide automatiquement :
  - Les entrÃ©es (profil candidat, paramÃ¨tres de requÃªte)
  - Les sorties (recommandations, skill gap, roadmap, embeddings)
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field, model_validator


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ENTRÃ‰ES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class CandidatProfile(BaseModel):
    """Profil d'un demandeur d'emploi â€” entrÃ©e principale du systÃ¨me."""

    candidat_id: str = Field(
        ...,
        description="Matricule unique du candidat (ex: PPKOU2501080016340)",
        examples=["PPKOU2501080016340"],
    )
    metier_vise: Optional[str] = Field(
        None, description="IntitulÃ© du mÃ©tier recherchÃ©", max_length=200
    )
    secteur_metier: Optional[str] = Field(
        None, description="Secteur d'activitÃ© souhaitÃ©", max_length=100
    )
    ncf_niveau_final: Optional[int] = Field(
        None, description="Code NCF niveau d'Ã©tudes (1=primaire â€¦ 9=doctorat)", ge=1, le=9
    )
    filiere_specialite: Optional[str] = Field(
        None, description="FiliÃ¨re / spÃ©cialitÃ© de formation", max_length=200
    )
    objectif: Optional[str] = Field(
        None, description="Objectif professionnel dÃ©clarÃ©", max_length=500
    )
    diplome_raw: Optional[str] = Field(
        None, description="DiplÃ´me obtenu (libellÃ© brut)", max_length=200
    )
    secteur_demande: Optional[str] = Field(
        None, description="Secteur d'emploi demandÃ©", max_length=100
    )
    mobilite_geo_bool: Optional[bool] = Field(
        None, description="Accepte la mobilitÃ© gÃ©ographique"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "candidat_id": "PPKOU2501080016340",
            "metier_vise": "Data Analyst",
            "secteur_metier": "Finance",
            "ncf_niveau_final": 8,
            "filiere_specialite": "Statistiques et sciences apparentÃ©es",
            "objectif": "IntÃ©grer une Ã©quipe data pour contribuer Ã  la prise de dÃ©cision stratÃ©gique",
            "mobilite_geo_bool": True,
        }
    }}


class RecommendRequest(BaseModel):
    """Corps de la requÃªte POST /recommend"""
    profil: CandidatProfile
    top_k: int = Field(5, description="Nombre d'offres Ã  retourner", ge=1, le=20)
    llm_backend: str = Field(
        "openai/gpt-oss-20b:free",
        description="Modele LLM unique via OpenRouter",
        pattern="^openai/gpt-oss-20b:free$",
    )
    include_roadmap: bool = Field(True, description="Inclure la roadmap de formation")
    include_skill_gap: bool = Field(True, description="Inclure l'analyse skill gap")


class SkillGapRequest(BaseModel):
    """Corps de la requÃªte POST /skill-gap"""
    candidat_id: str = Field(..., description="Matricule du candidat")
    offre_id: str = Field(..., description="UUID de l'offre d'emploi")
    llm_backend: str = Field("openai/gpt-oss-20b:free", pattern="^openai/gpt-oss-20b:free$")


class EmbedRequest(BaseModel):
    """Corps de la requÃªte POST /embed"""
    textes: List[str] = Field(
        ..., description="Liste de textes Ã  encoder", min_length=1, max_length=100
    )
    entity_kind: str = Field(
        "OFFRE_EMPLOI",
        description="Type d'entitÃ© : OFFRE_EMPLOI | CANDIDAT | COMPETENCE | METIER",
    )
    normalize: bool = Field(True, description="Normaliser les vecteurs (norme=1)")


class ChatRequest(BaseModel):
    """Corps de la requete POST /chat pour l'agent GraphRAG."""

    query: str = Field(..., description="Question utilisateur en langage naturel", min_length=3)
    candidat_id: Optional[str] = Field(None, description="Matricule candidat optionnel")
    top_k: int = Field(5, description="Nombre de resultats a recuperer", ge=1, le=20)
    include_traces: bool = Field(True, description="Inclure traces et critique dans la reponse")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SORTIES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class OffreRecommandee(BaseModel):
    """Une offre d'emploi dans le classement de recommandation."""
    rang: int
    offre_id: str
    titre_poste: str
    secteur: Optional[str] = None
    ville: Optional[str] = None
    type_contrat: Optional[str] = None
    score_hybride: float = Field(..., ge=0, le=1)
    score_semantique: float = Field(..., ge=0, le=1)
    score_graphe: float = Field(..., ge=0, le=1)
    taux_matching: float = Field(..., ge=0, le=1)
    competences_acquises: List[str] = Field(default_factory=list)
    competences_manquantes: List[str] = Field(default_factory=list)
    essentielles_manquantes: List[str] = Field(default_factory=list)


class RoadmapEtape(BaseModel):
    """Une Ã©tape de la roadmap de formation."""
    priorite: int
    competence_cible: str
    importance: str  # "essentielle" | "optionnelle"
    formation_nom: Optional[str] = None
    etablissement: Optional[str] = None
    duree: Optional[str] = None
    cout_estimatif: Optional[str] = None
    modalite: Optional[str] = None  # "presentiel" | "en ligne" | "hybride"
    delai_acquisition: Optional[str] = None
    impact_score: Optional[float] = None


class Roadmap(BaseModel):
    """Plan d'action personnalisÃ© pour amÃ©liorer le profil."""
    poste_cible: str
    score_matching_actuel: float
    score_matching_projete: float
    duree_totale_estimee: Optional[str] = None
    etapes: List[RoadmapEtape] = Field(default_factory=list)
    ressources_gratuites: List[str] = Field(default_factory=list)
    certifications_utiles: List[str] = Field(default_factory=list)
    conseil_candidature_immediate: Optional[str] = None
    message_motivation: Optional[str] = None


class SkillGapAnalysis(BaseModel):
    """Analyse complÃ¨te de l'Ã©cart de compÃ©tences."""
    taux_matching: float
    niveau_gap: str  # "faible" | "modÃ©rÃ©" | "important" | "critique"
    eligible_maintenant: bool
    competences_critiques: List[dict] = Field(default_factory=list)
    score_projete_apres_formation: Optional[float] = None
    message_candidat: Optional[str] = None


class RecommendResponse(BaseModel):
    """RÃ©ponse complÃ¨te de l'endpoint POST /recommend"""
    candidat_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    n_offres_analysees: int
    top_offres: List[OffreRecommandee]
    analyse_globale: Optional[str] = None
    score_employabilite_global: Optional[float] = None
    conseil_global: Optional[str] = None
    prochaine_action: Optional[str] = None
    skill_gap: Optional[SkillGapAnalysis] = None
    roadmap: Optional[Roadmap] = None
    latence_ms: float
    llm_backend: str
    model_version: str = "all-MiniLM-L6-v2-ft-offres-cm"


class SkillGapResponse(BaseModel):
    """RÃ©ponse de l'endpoint POST /skill-gap"""
    candidat_id: str
    offre_id: str
    analyse: SkillGapAnalysis
    roadmap: Optional[Roadmap] = None
    latence_ms: float


class EmbedResponse(BaseModel):
    """RÃ©ponse de l'endpoint POST /embed"""
    n_textes: int
    dimension: int = 384
    embeddings: List[List[float]]
    model_id: str = "all-MiniLM-L6-v2-ft-offres-cm"
    latence_ms: float


class ChatResponse(BaseModel):
    """Reponse de l'agent GraphRAG conversationnel."""

    answer: str
    use_case: Optional[str] = None
    candidat_id: Optional[str] = None
    top_k: int = 5
    traces: List[str] = Field(default_factory=list)
    critic: dict = Field(default_factory=dict)
    latence_ms: float


class OffreDetail(BaseModel):
    """DÃ©tails complets d'une offre d'emploi (GET /offre/{id})"""
    offre_id: str
    titre_poste: str
    employeur: Optional[str] = None
    type_entreprise: Optional[str] = None
    secteur_principal: Optional[str] = None
    secteurs_list: Optional[List[str]] = None
    ville_principale: Optional[str] = None
    villes_list: Optional[List[str]] = None
    type_contrat: Optional[str] = None
    groupe_contrat: Optional[str] = None
    ncf_niveau_code: Optional[int] = None
    niveau_etudes_raw: Optional[str] = None
    experience_min_ans: Optional[int] = None
    skills_list: Optional[List[str]] = None
    details_clean: Optional[str] = None
    metadata_str: Optional[str] = None


class HealthResponse(BaseModel):
    """RÃ©ponse de l'endpoint GET /health"""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    services: dict = Field(
        default_factory=lambda: {
            "neo4j":     "unknown",
            "pgvector":  "unknown",
            "st_model":  "unknown",
            "llm":       "openrouter:openai/gpt-oss-20b:free",
        }
    )


class ErrorResponse(BaseModel):
    """Format d'erreur standardisÃ©."""
    code: int
    message: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

