"""
roadmap_generator.py
===========================================================================
Module 05 — Générateur de Roadmap personnalisée (composant GraphRAG)

Produit un plan de formation structuré à partir du skill gap identifié,
intégrant les référentiels NCF camerounais et les ressources locales.
Appelé par recommendation_engine.py après la génération des recommandations.
===========================================================================
"""

import json
import logging
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent.parent
PROC    = ROOT / "data" / "processed"

# Chargement paresseux des référentiels NCF
_NCF_DET = None

def _get_ncf():
    global _NCF_DET
    if _NCF_DET is None:
        _NCF_DET = pd.read_parquet(PROC / "ncf_dom_detailles.parquet")
    return _NCF_DET

# Ressources de formation disponibles au Cameroun (enrichies)
FORMATIONS_CAMEROUN = {
    "universités": [
        "Université de Yaoundé I — Faculté des Sciences",
        "Université de Yaoundé II — Soa — Sciences sociales et économiques",
        "Université de Douala — ESSEC (École Supérieure des Sciences Économiques)",
        "Université de Dschang — Institut Universitaire de Technologie",
        "Université de Buéa — Faculty of Engineering and Technology",
        "Institut Universitaire de Technologie (IUT) — Douala",
    ],
    "grandes_ecoles": [
        "École Nationale Supérieure Polytechnique (ENSP) — Yaoundé",
        "École Nationale d'Administration et de Magistrature (ENAM)",
        "Institut National de la Statistique (INS) — Formation continue",
        "Institut Supérieur de Management et d'Informatique (ISMI)",
        "Sup'Management — Yaoundé et Douala",
    ],
    "en_ligne": [
        "Coursera (certificats reconnus, cours en FR disponibles)",
        "LinkedIn Learning (formation continue certifiante)",
        "MOOC de l'Agence Universitaire de la Francophonie (AUF)",
        "Google Digital Academy — Ateliers Numériques",
        "OpenClassrooms (parcours diplômants reconnus en Afrique)",
        "edX — MicroMasters et certificats professionnels",
    ],
    "pro": [
        "FECAPAFS — Fédération Camerounaise de Formation Professionnelle",
        "Centre de Formation Professionnelle Industrielle (CFPI) — Douala",
        "Centre d'Appui aux PME et à l'Innovation (CAPI)",
        "Chambre de Commerce, d'Industrie, des Mines et de l'Artisanat (CCIMA)",
    ],
}

DELAIS = {
    "court":  "1 à 3 mois — auto-formation ou MOOC intensif",
    "moyen":  "3 à 12 mois — certification professionnelle",
    "long":   "12 à 36 mois — diplôme ou formation qualifiante",
}

def _match_ncf_domaine(competence_label: str) -> dict | None:
    """Cherche le domaine NCF le plus proche d'une compétence."""
    ncf = _get_ncf()
    label_low = competence_label.lower()
    for _, row in ncf.iterrows():
        intitule_low = str(row.get("intitule", "")).lower()
        if any(word in intitule_low for word in label_low.split()[:3]):
            return {
                "code": str(row["code"]),
                "intitule": row["intitule"],
                "explication": str(row.get("explication", ""))[:200],
            }
    return None

def _select_formation(competence_label: str, secteur: str = "") -> dict:
    """Sélectionne la ressource de formation la plus adaptée."""
    label_low = competence_label.lower()
    # Heuristiques simples de sélection
    if any(k in label_low for k in ["informatique","logiciel","programmation","data","numérique","excel","sql"]):
        source = FORMATIONS_CAMEROUN["en_ligne"][0]
        type_f = "e-learning"
    elif any(k in label_low for k in ["gestion","comptabilité","finance","management","audit"]):
        source = FORMATIONS_CAMEROUN["grandes_ecoles"][1]
        type_f = "formation_continue"
    elif any(k in label_low for k in ["statistique","analyse","modélisation","économétrie"]):
        source = FORMATIONS_CAMEROUN["grandes_ecoles"][0]
        type_f = "université"
    elif any(k in label_low for k in ["communication","marketing","commerce","vente"]):
        source = FORMATIONS_CAMEROUN["universités"][1]
        type_f = "université"
    else:
        source = FORMATIONS_CAMEROUN["en_ligne"][2]  # AUF par défaut
        type_f = "e-learning"
    return {"etablissement": source, "type": type_f}

def generate_roadmap(
    candidat: dict,
    top_offre: dict,
    competences_manquantes: list[dict],
    score_actuel: float,
) -> dict:
    """
    Génère une roadmap structurée JSON à partir du skill gap.

    Args:
        candidat     : dict profil candidat (module ETL)
        top_offre    : dict offre recommandée (top-1)
        competences_manquantes : liste [{label, importance, type_comp}]
        score_actuel : score hybride actuel (avant amélioration)

    Returns:
        dict roadmap structuré (stocké en JSONB dans la table recommandations)
    """
    n_manq = len(competences_manquantes)
    score_projete = min(0.95, score_actuel + n_manq * 0.07 + 0.05)

    formations_recommandees = []
    for i, comp in enumerate(competences_manquantes[:6]):
        label     = comp.get("label", comp.get("skill_label", "compétence inconnue"))
        importance = comp.get("importance", "optional")
        type_comp  = comp.get("type_comp", comp.get("skillType", "skill/competence"))

        ncf_match = _match_ncf_domaine(label)
        formation = _select_formation(label, top_offre.get("secteur", ""))
        delai_key = "court" if importance == "essential" else "moyen"

        formations_recommandees.append({
            "priorite":           i + 1,
            "competence_cible":   label,
            "importance":         importance,
            "type_competence":    type_comp,
            "formation_ncf":      ncf_match["intitule"] if ncf_match else "Formation générale",
            "code_ncf":           ncf_match["code"] if ncf_match else None,
            "etablissement":      formation["etablissement"],
            "type_formation":     formation["type"],
            "ressource_online":   FORMATIONS_CAMEROUN["en_ligne"][0],
            "delai_acquisition":  DELAIS[delai_key],
        })

    # Étapes court terme : compétences essentielles manquantes
    ess_manq = [c for c in competences_manquantes if c.get("importance") == "essential"]
    etapes_ct = [f"Acquérir '{c.get('label','?')}' dans les 3 prochains mois"
                 for c in ess_manq[:2]]
    if not etapes_ct:
        etapes_ct = ["Renforcer les compétences transversales (communication, organisation)"]

    etapes_mt = [f"Valider '{c.get('label','?')}' par une certification reconnue"
                 for c in (competences_manquantes[2:5] if len(competences_manquantes) > 2 else [])]
    if not etapes_mt:
        etapes_mt = ["Préparer un portfolio de réalisations pour valoriser les compétences acquises"]

    metier_vise  = candidat.get("metier_vise", "poste visé")
    offre_titre  = top_offre.get("titre", "offre cible")
    ncf_candidat = candidat.get("ncf_niveau_final")
    ncf_offre    = top_offre.get("ncf_code")

    conseil = (
        f"Votre profil présente {n_manq} compétence(s) à développer pour le poste "
        f"'{offre_titre}'. En suivant ce plan sur {6 if n_manq > 3 else 3} mois, "
        f"votre score de correspondance peut atteindre {score_projete:.0%}. "
        f"Commencez par les compétences essentielles manquantes qui représentent "
        f"l'écart le plus critique avec les exigences du recruteur."
    )

    return {
        "poste_cible":              offre_titre,
        "metier_vise_candidat":     metier_vise,
        "score_matching_actuel":    round(score_actuel, 3),
        "score_matching_projete":   round(score_projete, 3),
        "nb_competences_manquantes":n_manq,
        "competences_prioritaires": [c.get("label","?") for c in ess_manq[:3]],
        "formations_recommandees":  formations_recommandees,
        "etapes_court_terme":       etapes_ct,
        "etapes_moyen_terme":       etapes_mt,
        "ressources_cameroun":      {
            "universites":    FORMATIONS_CAMEROUN["universités"][:3],
            "en_ligne":       FORMATIONS_CAMEROUN["en_ligne"][:3],
            "professionnel":  FORMATIONS_CAMEROUN["pro"][:2],
        },
        "niveau_ncf_candidat":  ncf_candidat,
        "niveau_ncf_requis":    ncf_offre,
        "gap_ncf":              (int(ncf_offre) - int(ncf_candidat))
                                 if ncf_offre and ncf_candidat else None,
        "conseil_candidature":  conseil,
    }
