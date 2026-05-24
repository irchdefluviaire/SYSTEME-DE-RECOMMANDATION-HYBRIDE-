"""Outils LangChain pour l'agent ReAct Agentic GraphRAG.

Chaque outil est décoré @tool et appelé dynamiquement par llama3.1.
La recherche sémantique utilise le SentenceTransformer fine-tuné réel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
for _p in [SRC / "05_graphrag", SRC / "03_knowledge_graph", SRC / "04_pgvector"]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# ── Modèle sémantique (chargé une seule fois) ─────────────────────────────────
_ST_MODEL = None

def _model():
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    from sentence_transformers import SentenceTransformer
    ft = ROOT / "models" / "st_finetuned" / "checkpoints" / "checkpoint-292"
    path = str(ft) if ft.exists() else "sentence-transformers/all-MiniLM-L6-v2"
    _ST_MODEL = SentenceTransformer(path)
    return _ST_MODEL


def _encode(texts: list[str]) -> np.ndarray:
    return _model().encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def _load_offers() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / "offres_normalized.parquet")


def _load_candidates() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / "candidats_normalized.parquet")


# ── Helpers scoring ────────────────────────────────────────────────────────────

def _ncf_score(cand_ncf, offre_ncf) -> tuple[float, int | None]:
    try:
        c, o = int(cand_ncf), int(offre_ncf)
        gap = o - c
        if gap <= 0:
            return 1.0, gap
        if gap == 1:
            return 0.72, gap
        if gap == 2:
            return 0.45, gap
        return 0.20, gap
    except (TypeError, ValueError):
        return 0.5, None


def _sector_score(cand_text: str, offre_text: str) -> float:
    def tok(t):
        stop = {"et","de","du","des","la","le","les","en","a","au","aux","and","the","pour"}
        return {w for w in t.lower().split() if len(w) >= 4 and w not in stop}
    ct, ot = tok(cand_text), tok(offre_text)
    if not ct or not ot:
        return 0.45
    return min(1.0, len(ct & ot) / max(min(len(ct), len(ot)), 1))


def _skill_overlap(cand_text: str, skills_raw: str) -> tuple[list[str], list[str], float]:
    skills = {s.strip() for s in str(skills_raw).replace(",", " ").split() if len(s.strip()) >= 4}
    ctoks  = {t for t in cand_text.lower().split() if len(t) >= 4}
    acq    = sorted(skills & ctoks)[:6]
    manq   = sorted(skills - ctoks)[:6]
    taux   = round(len(acq) / max(len(skills), 1), 3)
    return acq, manq, taux


def _verdict(score: float, ncf_gap: int | None, n_manq: int) -> str:
    if score >= 0.65 and (ncf_gap is None or ncf_gap <= 0) and n_manq <= 2:
        return "pret_a_postuler"
    if score >= 0.50 and (ncf_gap is None or ncf_gap <= 2):
        return "postuler_avec_plan"
    if score >= 0.38:
        return "vivier_a_developper"
    return "hors_cible_actuel"


def _enrich_offer(row: pd.Series, sem_score: float,
                  cand_ncf=None, cand_profile_text: str = "") -> dict:
    offre_ncf   = row.get("ncf_niveau_code")
    offre_text  = " ".join([str(row.get("titre_poste","")), str(row.get("secteur_principal",""))])
    ns, gap     = _ncf_score(cand_ncf, offre_ncf)
    ss          = _sector_score(cand_profile_text, offre_text)
    acq, manq, taux = _skill_overlap(cand_profile_text, str(row.get("skills_raw","")))
    score = round(0.40 * sem_score + 0.30 * taux + 0.20 * ns + 0.10 * ss, 4)
    return {
        "offre_id":      str(row.get("offre_id", "")),
        "titre":         str(row.get("titre_poste", "")),
        "secteur":       str(row.get("secteur_principal", "")),
        "ville":         str(row.get("ville_principale", "")),
        "type_contrat":  str(row.get("type_contrat_norm", "")),
        "ncf_requis":    str(offre_ncf) if offre_ncf else None,
        "details":       str(row.get("details_clean", ""))[:250],
        "score_sem":     round(float(sem_score), 4),
        "score_hybride": score,
        "ncf_score":     round(ns, 3),
        "ncf_gap":       gap,
        "sector_score":  round(ss, 3),
        "skill_taux":    taux,
        "acquises":      acq,
        "manquantes":    manq,
        "verdict":       _verdict(score, gap, len(manq)),
    }


# ── Outils exposés à l'agent ───────────────────────────────────────────────────

@tool
def search_offers(query: str, top_k: int = 5) -> str:
    """Recherche sémantique d'offres d'emploi à partir d'une description en langage naturel.
    Utilise le SentenceTransformer fine-tuné pour calculer la similarité cosinus.
    Utilise cet outil pour toute requête générale sans ID candidat.

    Args:
        query: Description du profil, du poste ou du secteur recherché.
        top_k: Nombre d'offres à retourner (défaut 5).
    """
    df      = _load_offers()
    texts   = df["text_to_embed"].fillna("").astype(str).tolist()[:600]
    q_vec   = _encode([query])[0]
    o_vecs  = _encode(texts)
    sims    = o_vecs @ q_vec
    idx     = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        row = df.iloc[i]
        results.append({
            "offre_id":     str(row.get("offre_id", "")),
            "titre":        str(row.get("titre_poste", "")),
            "secteur":      str(row.get("secteur_principal", "")),
            "ville":        str(row.get("ville_principale", "")),
            "type_contrat": str(row.get("type_contrat_norm", "")),
            "ncf_requis":   str(row.get("ncf_niveau_code", "")) or None,
            "skills_raw":   str(row.get("skills_raw", ""))[:200],
            "details":      str(row.get("details_clean", ""))[:200],
            "score_sem":    round(float(sims[i]), 4),
        })
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def get_candidate_profile(candidat_id: str) -> str:
    """Charge le profil complet d'un candidat à partir de son identifiant.
    Utilise toujours cet outil en premier quand un ID candidat est fourni.

    Args:
        candidat_id: Identifiant du candidat (ex: PP001, CAND_042, C001).
    """
    df  = _load_candidates()
    row = df[df["candidat_id"].astype(str) == str(candidat_id)]
    if row.empty:
        available = df["candidat_id"].astype(str).head(8).tolist()
        return json.dumps({"erreur": f"Candidat '{candidat_id}' introuvable.",
                           "ids_disponibles": available})
    r = row.iloc[0]
    return json.dumps({
        "candidat_id":       str(r.get("candidat_id", candidat_id)),
        "metier_vise":       str(r.get("metier_vise", "") or ""),
        "secteur_metier":    str(r.get("secteur_metier", "") or ""),
        "ncf_niveau_final":  int(r["ncf_niveau_final"]) if pd.notna(r.get("ncf_niveau_final")) else None,
        "filiere_specialite":str(r.get("filiere_specialite", "") or ""),
        "diplome_raw":       str(r.get("diplome_raw", "") or ""),
        "objectif":          str(r.get("objectif", "") or "")[:300],
        "text_to_embed":     str(r.get("text_to_embed", "") or "")[:350],
    }, ensure_ascii=False, indent=2)


@tool
def search_offers_for_candidate(candidat_id: str, top_k: int = 5) -> str:
    """Recherche et classe les meilleures offres pour un candidat spécifique.
    Calcule un score hybride : 40% sémantique + 30% compétences + 20% NCF + 10% secteur.
    Utilise cet outil après avoir chargé le profil du candidat.

    Args:
        candidat_id: Identifiant du candidat.
        top_k: Nombre d'offres à retourner (défaut 5).
    """
    df_c = _load_candidates()
    df_o = _load_offers()
    row  = df_c[df_c["candidat_id"].astype(str) == str(candidat_id)]
    if row.empty:
        return json.dumps({"erreur": f"Candidat '{candidat_id}' introuvable."})

    r           = row.iloc[0]
    query       = str(r.get("text_to_embed","") or r.get("objectif","") or r.get("metier_vise",""))
    cand_ncf    = int(r["ncf_niveau_final"]) if pd.notna(r.get("ncf_niveau_final")) else None
    cand_text   = " ".join([str(r.get("metier_vise","")), str(r.get("secteur_metier","")),
                             str(r.get("filiere_specialite","")), str(r.get("objectif",""))])

    texts  = df_o["text_to_embed"].fillna("").astype(str).tolist()[:600]
    q_vec  = _encode([query])[0]
    o_vecs = _encode(texts)
    sims   = o_vecs @ q_vec
    pool_n = min(top_k * 6, len(df_o))
    idx    = np.argsort(-sims)[:pool_n]

    enriched = [_enrich_offer(df_o.iloc[i], float(sims[i]), cand_ncf, cand_text) for i in idx]
    enriched.sort(key=lambda x: x["score_hybride"], reverse=True)
    return json.dumps(enriched[:top_k], ensure_ascii=False, indent=2)


@tool
def get_skill_gap(candidat_id: str, offre_id: str) -> str:
    """Analyse détaillée de l'écart de compétences entre un candidat et une offre précise.
    Retourne les compétences acquises, manquantes, le taux de correspondance et la compatibilité NCF.

    Args:
        candidat_id: Identifiant du candidat.
        offre_id: Identifiant de l'offre d'emploi.
    """
    df_c = _load_candidates()
    df_o = _load_offers()
    crow = df_c[df_c["candidat_id"].astype(str) == str(candidat_id)]
    orow = df_o[df_o["offre_id"].astype(str) == str(offre_id)]
    if crow.empty:
        return json.dumps({"erreur": f"Candidat '{candidat_id}' introuvable."})
    if orow.empty:
        return json.dumps({"erreur": f"Offre '{offre_id}' introuvable."})

    c = crow.iloc[0]
    o = orow.iloc[0]
    cand_text = " ".join([str(c.get("metier_vise","")), str(c.get("filiere_specialite","")),
                          str(c.get("objectif","")), str(c.get("text_to_embed",""))]).lower()
    acq, manq, taux = _skill_overlap(cand_text, str(o.get("skills_raw","")))
    cand_ncf  = int(c["ncf_niveau_final"]) if pd.notna(c.get("ncf_niveau_final")) else None
    offre_ncf = o.get("ncf_niveau_code")
    ns, gap   = _ncf_score(cand_ncf, offre_ncf)

    return json.dumps({
        "candidat_id":           candidat_id,
        "offre_id":              offre_id,
        "titre_offre":           str(o.get("titre_poste","")),
        "taux_correspondance":   taux,
        "competences_acquises":  acq,
        "competences_manquantes":manq,
        "ncf_candidat":          cand_ncf,
        "ncf_offre_requis":      str(offre_ncf) if offre_ncf else None,
        "ncf_score":             round(ns, 3),
        "ncf_gap":               gap,
        "verdict":               _verdict(0.55 * taux + 0.20 * ns, gap, len(manq)),
    }, ensure_ascii=False, indent=2)


@tool
def generate_roadmap(candidat_id: str, offre_id: str) -> str:
    """Génère un plan de formation structuré pour qu'un candidat comble ses lacunes et soit éligible à une offre.
    Utilise cet outil en dernier, après avoir obtenu le skill gap.

    Args:
        candidat_id: Identifiant du candidat.
        offre_id: Identifiant de l'offre cible.
    """
    from roadmap_generator import generate_roadmap as _gen

    df_c = _load_candidates()
    df_o = _load_offers()
    crow = df_c[df_c["candidat_id"].astype(str) == str(candidat_id)]
    orow = df_o[df_o["offre_id"].astype(str) == str(offre_id)]
    if crow.empty or orow.empty:
        return json.dumps({"erreur": "Candidat ou offre introuvable."})

    c = crow.iloc[0].to_dict()
    o = orow.iloc[0].to_dict()
    cand_text = str(c.get("text_to_embed","") or "").lower()
    skills    = [s.strip() for s in str(o.get("skills_raw","")).replace(",", " ").split() if len(s.strip()) >= 4]
    manq      = [{"label": s, "importance": "essential"} for s in skills if s.lower() not in cand_text][:8]
    roadmap   = _gen(candidat=c, top_offre=o, competences_manquantes=manq, score_actuel=0.5)
    return json.dumps(roadmap, ensure_ascii=False, indent=2, default=str)
