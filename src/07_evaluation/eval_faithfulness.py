"""
eval_faithfulness.py
Module 07 — Évaluation de la fidélité des sorties LLM au contexte fourni

Méthodes disponibles :
  - token_overlap_faithfulness  : recouvrement lexical contexte → sortie
  - bigram_overlap_faithfulness : recouvrement de bigrammes
  - keyword_faithfulness        : présence de mots-clés critiques
  - score_roadmap_quality       : complétude des champs JSON d'une roadmap
  - FaithfulnessEvaluator       : classe principale agrégeant toutes les méthodes

Référence : définition des champs requis dans la méthodologie (Chapitre III).
"""
import re
from dataclasses import dataclass

import numpy as np

_TOKEN_RE   = re.compile(r"\b\w+\b", re.UNICODE)
_STOP_FR    = frozenset({
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en",
    "à", "au", "aux", "est", "sont", "se", "sur", "par", "pour",
    "que", "qui", "avec", "dans", "il", "elle", "ils", "elles",
    "nous", "vous", "on", "ce", "cet", "cette", "ces", "mon",
    "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "ou",
    "ne", "pas", "plus", "très", "aussi", "mais", "car", "donc",
    "si", "lui", "leur", "leurs", "tout", "tous", "toute", "toutes",
})

# Champs requis de la roadmap (Chapitre III)
_CHAMPS_REQUIS = [
    "poste_cible",
    "score_matching_actuel",
    "score_matching_projete",
    "etapes",
    "duree_totale_estimee",
    "ressources_gratuites",
    "conseil_candidature_immediate",
    "message_motivation",
]
_CHAMPS_BONUS = ["budget_estime", "mentor_suggere", "certifications"]


# ─────────────────────────────────────────────────────────────────────────
# Utilitaires texte
# ─────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Tokenise un texte en filtrant les stopwords français."""
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOP_FR and len(t) > 2
    ]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


# ─────────────────────────────────────────────────────────────────────────
# Fonctions de fidélité
# ─────────────────────────────────────────────────────────────────────────

def token_overlap_faithfulness(output: str, context: str) -> float:
    """
    Fidélité par recouvrement de tokens contexte → sortie.

    Mesure la fraction des tokens du contexte présents dans la sortie.
    Score ∈ [0, 1].
    """
    ctx_tokens = _tokenize(context)
    out_tokens = _tokenize(output)

    if not ctx_tokens:
        return 1.0

    ctx_set = set(ctx_tokens)
    out_set = set(out_tokens)
    return len(ctx_set & out_set) / len(ctx_set)


def bigram_overlap_faithfulness(output: str, context: str) -> float:
    """
    Fidélité par recouvrement de bigrammes contexte → sortie.
    Capture mieux les colocations techniques (ex. "compétences ESCO").
    Score ∈ [0, 1].
    """
    ctx_tokens = _tokenize(context)
    out_tokens = _tokenize(output)

    ctx_bigrams = _ngrams(ctx_tokens, 2)
    out_bigrams = _ngrams(out_tokens, 2)

    if not ctx_bigrams:
        return 1.0
    return len(ctx_bigrams & out_bigrams) / len(ctx_bigrams)


def keyword_faithfulness(output: str, keywords: list[str]) -> float:
    """
    Vérifie la présence de mots-clés critiques dans la sortie.

    Args:
        keywords : termes extraits du contexte (codes NCF, compétences ESCO, etc.)

    Returns:
        fraction de mots-clés trouvés ∈ [0, 1]
    """
    if not keywords:
        return 1.0
    out_lower = output.lower()
    return sum(kw.lower() in out_lower for kw in keywords) / len(keywords)


def score_roadmap_quality(roadmap: dict) -> float:
    """
    Complétude des champs d'une roadmap de montée en compétences.

    Score ∈ [0, 1] où 1.0 = tous les champs requis présents + bonus optionnels.
    """
    n_requis = sum(1 for c in _CHAMPS_REQUIS if roadmap.get(c))
    n_bonus  = sum(1 for c in _CHAMPS_BONUS  if roadmap.get(c))

    score_base  = n_requis / len(_CHAMPS_REQUIS)
    score_bonus = min(n_bonus / max(len(_CHAMPS_BONUS), 1), 1.0) * 0.05
    return round(min(score_base + score_bonus, 1.0), 4)


def composite_faithfulness(output: str, context: str) -> float:
    """
    Score de fidélité composite (token 1-gram × 0.6 + bigrammes × 0.4).
    Équilibre couverture globale et cohérence locale.
    """
    s1 = token_overlap_faithfulness(output, context)
    s2 = bigram_overlap_faithfulness(output, context)
    return round(0.6 * s1 + 0.4 * s2, 4)


# ─────────────────────────────────────────────────────────────────────────
# Évaluateur principal
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class FaithfulnessResult:
    """Résultats d'une évaluation de fidélité pour une seule sortie."""
    token_overlap:   float = 0.0
    bigram_overlap:  float = 0.0
    keyword_recall:  float = 1.0
    roadmap_quality: float = 0.0
    composite:       float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "token_overlap":   round(self.token_overlap,   4),
            "bigram_overlap":  round(self.bigram_overlap,  4),
            "keyword_recall":  round(self.keyword_recall,  4),
            "roadmap_quality": round(self.roadmap_quality, 4),
            "composite":       round(self.composite,       4),
        }


class FaithfulnessEvaluator:
    """
    Évalue la fidélité des sorties générées au contexte GraphRAG fourni.

    Méthodes : token_overlap, bigram_overlap, keyword_recall,
               roadmap_completeness, composite.
    Extension future : NLI-based si un CrossEncoder est disponible.
    """

    def evaluate(
        self,
        output:   str,
        context:  str,
        keywords: list[str] | None = None,
        roadmap:  dict | None = None,
    ) -> FaithfulnessResult:
        """
        Évalue la fidélité d'une sortie LLM au contexte fourni.

        Args:
            output   : texte généré par le LLM
            context  : contexte GraphRAG fourni au LLM
            keywords : mots-clés critiques extraits du contexte (optionnel)
            roadmap  : dict de la roadmap générée (optionnel)
        """
        tok1 = token_overlap_faithfulness(output, context)
        tok2 = bigram_overlap_faithfulness(output, context)
        kw   = keyword_faithfulness(output, keywords or [])
        rq   = score_roadmap_quality(roadmap or {})
        comp = round(0.40 * tok1 + 0.30 * tok2 + 0.20 * kw + 0.10 * rq, 4)

        return FaithfulnessResult(
            token_overlap=round(tok1, 4),
            bigram_overlap=round(tok2, 4),
            keyword_recall=round(kw, 4),
            roadmap_quality=round(rq, 4),
            composite=comp,
        )

    def evaluate_batch(
        self,
        outputs:       list[str],
        contexts:      list[str],
        keywords_list: list[list[str]] | None = None,
        roadmaps:      list[dict]       | None = None,
    ) -> dict[str, float]:
        """
        Évalue un lot de sorties et retourne les métriques moyennées.
        """
        n = len(outputs)
        kw_list = keywords_list or [[] for _ in range(n)]
        rm_list = roadmaps      or [{} for _ in range(n)]

        all_res = [
            self.evaluate(out, ctx, kw, rm)
            for out, ctx, kw, rm in zip(outputs, contexts, kw_list, rm_list)
        ]

        return {
            "n_samples":       n,
            "token_overlap":   round(float(np.mean([r.token_overlap   for r in all_res])), 4),
            "bigram_overlap":  round(float(np.mean([r.bigram_overlap   for r in all_res])), 4),
            "keyword_recall":  round(float(np.mean([r.keyword_recall   for r in all_res])), 4),
            "roadmap_quality": round(float(np.mean([r.roadmap_quality  for r in all_res])), 4),
            "composite":       round(float(np.mean([r.composite        for r in all_res])), 4),
        }
