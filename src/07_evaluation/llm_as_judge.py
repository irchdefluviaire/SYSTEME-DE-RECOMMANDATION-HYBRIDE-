"""
llm_as_judge.py
Module 07 — Juge LLM pour l'évaluation des recommandations

Évalue la qualité des sorties du système de recommandation hybride selon
5 critères définis dans la méthodologie :

  - pertinence     : le match candidat-offre est-il justifié par les compétences ?
  - coherence      : skill gap et roadmap sont-ils logiques et cohérents ?
  - fidelite       : citations réglementaires (NCF, MEPC) correctes ?
  - completude     : tous les champs requis de la sortie sont-ils présents ?
  - actionnabilite : la roadmap propose-t-elle des étapes concrètes ?

Modes supportés :
  - simulation   : scores heuristiques déterministes (toujours disponible)
  - ollama       : LLM local via OpenAI-compatible API (LLM_BASE_URL dans .env)
  - openai       : API OpenAI (OPENAI_API_KEY dans .env)
"""
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
Tu es un expert en recrutement au Cameroun et en intelligence artificielle appliquée aux RH.
Évalue la qualité d'une recommandation emploi-compétences générée par un système d'IA.

Critères (0.0 à 1.0 chacun) :
1. pertinence     : le match candidat-offre est justifié par les compétences et le profil
2. coherence      : skill gap et roadmap sont logiques et cohérents entre eux
3. fidelite       : références réglementaires (NCF, MEPC) citées sont correctes
4. completude     : tous les champs requis (offres, skill gap, roadmap, verdict) sont présents
5. actionnabilite : la roadmap propose des étapes concrètes et réalistes

Réponds UNIQUEMENT en JSON valide :
{"pertinence":<float>,"coherence":<float>,"fidelite":<float>,"completude":<float>,"actionnabilite":<float>,"justification":"<1-2 phrases>"}"""

_JUDGE_USER = """\
PROFIL CANDIDAT : {candidat}
OFFRE RECOMMANDÉE (top-1) : {offre}
SKILL GAP : {skill_gap}
ROADMAP : {roadmap}
VERDICT : {verdict}

Évalue la qualité de cette recommandation."""


# ─────────────────────────────────────────────────────────────────────────
# Résultat
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeScore:
    """Score LLM-as-judge pour une recommandation."""
    pertinence:     float = 0.0
    coherence:      float = 0.0
    fidelite:       float = 0.0
    completude:     float = 0.0
    actionnabilite: float = 0.0
    global_score:   float = 0.0
    mode:           str   = "simulation"
    justification:  str   = ""

    _WEIGHTS = [0.30, 0.25, 0.15, 0.15, 0.15]

    @classmethod
    def from_scores(
        cls,
        p: float, c: float, f: float, d: float, a: float,
        mode: str, justification: str = "",
    ) -> "JudgeScore":
        g = sum(w * v for w, v in zip(cls._WEIGHTS, [p, c, f, d, a]))
        return cls(
            pertinence=round(p, 3),
            coherence=round(c, 3),
            fidelite=round(f, 3),
            completude=round(d, 3),
            actionnabilite=round(a, 3),
            global_score=round(g, 4),
            mode=mode,
            justification=justification,
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────
# Heuristique de simulation
# ─────────────────────────────────────────────────────────────────────────

def _heuristic_score(
    offre:     dict,
    skill_gap: dict,
    roadmap:   dict,
) -> JudgeScore:
    """Score heuristique déterministe basé sur les métadonnées structurées."""
    from eval_faithfulness import score_roadmap_quality

    score_hybride = float(offre.get("score_hybride", 0.5))
    taux_match    = float(skill_gap.get("taux_matching", 0.5))
    score_proj    = float(roadmap.get("score_matching_projete", taux_match + 0.2))

    # Pertinence : score hybride de l'offre top-1
    pertinence = min(1.0, score_hybride * 1.1)

    # Cohérence : écart taux_match vs score_hybride + progression cohérente
    coherence = 1.0 - abs(taux_match - score_hybride) * 0.5
    if score_proj > taux_match:
        coherence = min(1.0, coherence + 0.1)

    # Fidélité : présence de références réglementaires dans la roadmap
    roadmap_str = json.dumps(roadmap, ensure_ascii=False).lower()
    has_ncf     = any(kw in roadmap_str for kw in ("ncf", "niveau", "formation"))
    has_mepc    = any(kw in roadmap_str for kw in ("mepc", "métier", "groupe"))
    fidelite    = 0.5 + 0.25 * int(has_ncf) + 0.25 * int(has_mepc)

    # Complétude : champs requis de la roadmap
    completude = score_roadmap_quality(roadmap)

    # Actionnabilité : nombre d'étapes dans la roadmap
    etapes         = roadmap.get("etapes", [])
    n_etapes       = len(etapes) if isinstance(etapes, list) else 0
    actionnabilite = min(1.0, 0.4 + 0.12 * n_etapes)

    return JudgeScore.from_scores(
        p=pertinence,
        c=coherence,
        f=fidelite,
        d=completude,
        a=actionnabilite,
        mode="simulation",
        justification="Score heuristique basé sur les métadonnées structurées.",
    )


# ─────────────────────────────────────────────────────────────────────────
# Juge principal
# ─────────────────────────────────────────────────────────────────────────

class LLMJudge:
    """
    Juge LLM évaluant la qualité des recommandations du système hybride.

    En mode simulation (défaut), utilise une heuristique déterministe sans LLM.
    En mode ollama/openai, appelle le LLM configuré dans .env.
    """

    def __init__(self, mode: str = "simulation", model: str | None = None) -> None:
        """
        Args:
            mode  : "simulation" | "ollama" | "openai"
            model : identifiant du modèle LLM (ex. "llama3.1:latest")
        """
        self.mode  = mode
        self.model = model

    @staticmethod
    def _truncate(data: dict, max_chars: int = 400) -> str:
        s = json.dumps(data, ensure_ascii=False)
        return s[:max_chars] + "…" if len(s) > max_chars else s

    def _call_llm(self, user_prompt: str) -> str:
        """Appelle le LLM configuré et retourne le texte brut."""
        import os
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")

        if self.mode == "ollama":
            import openai
            client = openai.OpenAI(
                base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("LLM_API_KEY", "ollama"),
            )
            model = self.model or os.getenv("LLM_CHOICE", "llama3.1:latest")
        elif self.mode == "openai":
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model  = self.model or "gpt-4o-mini"
        else:
            raise ValueError(f"Mode LLM inconnu : {self.mode}")

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _parse_json(raw: str) -> dict:
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def score_recommendation(
        self,
        candidat:  dict,
        offre:     dict,
        skill_gap: dict,
        roadmap:   dict,
        verdict:   str = "",
    ) -> JudgeScore:
        """
        Évalue une recommandation complète (candidat + offre + skill gap + roadmap).

        Returns:
            JudgeScore avec les 5 critères, le score global et la justification.
        """
        if self.mode == "simulation":
            return _heuristic_score(offre, skill_gap, roadmap)

        user_prompt = _JUDGE_USER.format(
            candidat=self._truncate(candidat),
            offre=self._truncate(offre),
            skill_gap=self._truncate(skill_gap),
            roadmap=self._truncate(roadmap),
            verdict=verdict,
        )
        try:
            raw  = self._call_llm(user_prompt)
            data = self._parse_json(raw)

            def _clamp(v) -> float:
                return max(0.0, min(1.0, float(v or 0.5)))

            return JudgeScore.from_scores(
                p=_clamp(data.get("pertinence")),
                c=_clamp(data.get("coherence")),
                f=_clamp(data.get("fidelite")),
                d=_clamp(data.get("completude")),
                a=_clamp(data.get("actionnabilite")),
                mode=self.mode,
                justification=data.get("justification", ""),
            )

        except Exception as exc:
            log.warning(f"Erreur juge LLM ({exc}) — fallback heuristique")
            res      = _heuristic_score(offre, skill_gap, roadmap)
            res.mode = f"simulation_fallback ({self.mode})"
            return res

    def score_batch(self, samples: list[dict]) -> dict[str, float]:
        """
        Évalue un lot de recommandations.

        Chaque sample doit contenir : candidat, offre, skill_gap, roadmap, verdict (opt.)

        Returns:
            Métriques moyennées sur le lot.
        """
        scores = [
            self.score_recommendation(
                s.get("candidat",  {}),
                s.get("offre",     {}),
                s.get("skill_gap", {}),
                s.get("roadmap",   {}),
                s.get("verdict",   ""),
            )
            for s in samples
        ]
        n = len(scores)
        if n == 0:
            return {"n_samples": 0}

        return {
            "n_samples":      n,
            "pertinence":     round(float(np.mean([s.pertinence     for s in scores])), 4),
            "coherence":      round(float(np.mean([s.coherence       for s in scores])), 4),
            "fidelite":       round(float(np.mean([s.fidelite        for s in scores])), 4),
            "completude":     round(float(np.mean([s.completude      for s in scores])), 4),
            "actionnabilite": round(float(np.mean([s.actionnabilite  for s in scores])), 4),
            "global_score":   round(float(np.mean([s.global_score    for s in scores])), 4),
        }
