"""
eval_embedding.py
Module 07 — Wrappers de modèles d'embedding pour le benchmark

Interface unifiée pour :
  - SentenceTransformerModel  (modèles HuggingFace ou locaux)
  - TFIDFModel                (baseline lexicale TF-IDF, sklearn)
  - BM25Model                 (baseline IR classique, implémentation maison)

Interface commune :
    model.rank_all(queries, corpus, top_k)   → {qid: [doc_id, …]}
    model.measure_latency(texts)              → ms/phrase
    model.get_size_mb()                       → float
    model.get_dim()                           → int
"""
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


# ─────────────────────────────────────────────────────────────────────────
# Interface commune
# ─────────────────────────────────────────────────────────────────────────

class EmbeddingModel(ABC):
    """Interface commune pour tous les modèles d'embedding évalués."""

    name:   str
    label:  str
    family: str

    @abstractmethod
    def rank_all(
        self,
        queries: dict[str, str],
        corpus:  dict[str, str],
        top_k:   int = 100,
    ) -> dict[str, list[str]]:
        """
        Classe les documents du corpus pour chaque requête.

        Args:
            queries : {query_id: query_text}
            corpus  : {doc_id:   doc_text}
            top_k   : nombre maximal de résultats par requête

        Returns:
            {query_id: [doc_id1, doc_id2, …]} trié par pertinence décroissante
        """
        ...

    @abstractmethod
    def measure_latency(self, texts: list[str], batch_size: int = 64) -> float:
        """Latence moyenne d'encodage en millisecondes par phrase."""
        ...

    @abstractmethod
    def get_size_mb(self) -> float:
        """Taille estimée du modèle en mégaoctets."""
        ...

    @abstractmethod
    def get_dim(self) -> int:
        """Dimension des embeddings (0 pour les modèles sans vecteur dense)."""
        ...


# ─────────────────────────────────────────────────────────────────────────
# SentenceTransformer
# ─────────────────────────────────────────────────────────────────────────

class SentenceTransformerModel(EmbeddingModel):
    """Wrapper pour un modèle sentence-transformers (HuggingFace ou local)."""

    def __init__(
        self,
        name:   str,
        hf_id:  str,
        label:  str,
        family: str,
        local:  bool = False,
        prefix: str  = "",
        online: bool = False,
    ) -> None:
        self.name    = name
        self.label   = label
        self.family  = family
        self._hf_id  = hf_id
        self._local  = local
        self._prefix = prefix
        self._online = online
        self._model  = None

    # ── Chargement paresseux ────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        path = str(ROOT / self._hf_id) if self._local else self._hf_id
        self._model = SentenceTransformer(path, local_files_only=not self._online)
        log.info(f"  Modèle chargé : {self.name}")

    # ── Encodage ───────────────────────────────────────────────────────

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        self._load()
        if self._prefix:
            texts = [self._prefix + t for t in texts]
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    # ── Interface EmbeddingModel ────────────────────────────────────────

    def rank_all(
        self,
        queries: dict[str, str],
        corpus:  dict[str, str],
        top_k:   int = 100,
    ) -> dict[str, list[str]]:
        corpus_ids   = list(corpus.keys())
        corpus_texts = [corpus[d]  for d in corpus_ids]
        query_ids    = list(queries.keys())
        query_texts  = [queries[q] for q in query_ids]

        corpus_vecs = self._encode(corpus_texts)    # (n_docs,   dim)
        query_vecs  = self._encode(query_texts)     # (n_queries, dim)

        # Cosine similarity = dot product sur vecteurs L2-normalisés
        sims = query_vecs @ corpus_vecs.T            # (n_queries, n_docs)

        results: dict[str, list[str]] = {}
        for i, qid in enumerate(query_ids):
            ranked_idx  = np.argsort(-sims[i])[:top_k]
            results[qid] = [corpus_ids[j] for j in ranked_idx]
        return results

    def measure_latency(self, texts: list[str], batch_size: int = 64) -> float:
        self._load()
        # Warmup sur un petit lot
        self._encode(texts[:min(8, len(texts))], batch_size=batch_size)
        t0 = time.perf_counter()
        self._encode(texts, batch_size=batch_size)
        return round((time.perf_counter() - t0) * 1000.0 / max(len(texts), 1), 3)

    def get_size_mb(self) -> float:
        path = ROOT / self._hf_id if self._local else None
        if path and path.exists():
            total = sum(
                f.stat().st_size
                for f in path.rglob("*")
                if f.is_file() and f.suffix in (".bin", ".pt", ".safetensors")
            )
            if total > 0:
                return round(total / 1024 / 1024, 1)
        # Estimation empirique par dimension
        return float({384: 90, 512: 120, 768: 270, 1024: 440}.get(self.get_dim(), 150))

    def get_dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()


# ─────────────────────────────────────────────────────────────────────────
# TF-IDF
# ─────────────────────────────────────────────────────────────────────────

class TFIDFModel(EmbeddingModel):
    """Baseline lexicale TF-IDF avec cosine similarity (sklearn)."""

    name   = "tfidf"
    label  = "TF-IDF (sparse)"
    family = "lexical"

    def __init__(self, max_features: int = 50_000) -> None:
        self._max_features   = max_features
        self._vectorizer     = None
        self._corpus_matrix: np.ndarray | None = None   # (n_docs, vocab), float32
        self._corpus_ids:    list[str]          = []

    def _corpus_changed(self, ids: list[str]) -> bool:
        if len(ids) != len(self._corpus_ids):
            return True
        if ids and ids[0] != self._corpus_ids[0]:
            return True
        return False

    def _fit(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self._vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        X = self._vectorizer.fit_transform(corpus_texts)
        # Conserver la matrice sparse normalisée (économise ~600 Mo pour 3k×50k)
        self._corpus_matrix = normalize(X, norm="l2")
        self._corpus_ids    = corpus_ids
        log.info(f"  TF-IDF ajusté sur {len(corpus_ids)} documents")

    def rank_all(
        self,
        queries: dict[str, str],
        corpus:  dict[str, str],
        top_k:   int = 100,
    ) -> dict[str, list[str]]:
        from sklearn.metrics.pairwise import cosine_similarity

        corpus_ids   = list(corpus.keys())
        corpus_texts = [corpus[d] for d in corpus_ids]

        if self._corpus_changed(corpus_ids):
            self._fit(corpus_ids, corpus_texts)

        query_ids   = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        Q_sparse = self._vectorizer.transform(query_texts)
        # cosine_similarity accepte les matrices sparse et retourne un dense array
        sims = cosine_similarity(Q_sparse, self._corpus_matrix)   # (n_queries, n_docs)

        results: dict[str, list[str]] = {}
        for i, qid in enumerate(query_ids):
            ranked_idx   = np.argsort(-sims[i])[:top_k]
            results[qid] = [corpus_ids[j] for j in ranked_idx]
        return results

    def measure_latency(self, texts: list[str], batch_size: int = 64) -> float:
        if self._vectorizer is None:
            return 1.0
        t0 = time.perf_counter()
        self._vectorizer.transform(texts)
        return round((time.perf_counter() - t0) * 1000.0 / max(len(texts), 1), 3)

    def get_size_mb(self) -> float:
        return 20.0

    def get_dim(self) -> int:
        if self._vectorizer is None:
            return self._max_features
        return len(self._vectorizer.vocabulary_)


# ─────────────────────────────────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────────────────────────────────

class BM25Model(EmbeddingModel):
    """
    Baseline BM25 (Okapi BM25) implémentée en Python pur.
    Paramètres TREC standards : k1 = 1.5, b = 0.75.
    """

    name   = "bm25"
    label  = "BM25 (sparse)"
    family = "lexical"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b  = b
        self._corpus_tokens: list[list[str]] = []
        self._idf: dict[str, float]          = {}
        self._avgdl: float                   = 1.0
        self._n: int                         = 0
        self._corpus_ids: list[str]          = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _corpus_changed(self, ids: list[str]) -> bool:
        if len(ids) != len(self._corpus_ids):
            return True
        if ids and ids[0] != self._corpus_ids[0]:
            return True
        return False

    def _fit(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        self._corpus_ids    = corpus_ids
        self._corpus_tokens = [self._tokenize(t) for t in corpus_texts]
        self._n             = len(self._corpus_tokens)
        self._avgdl         = (
            sum(len(d) for d in self._corpus_tokens) / self._n
            if self._n > 0 else 1.0
        )
        df: Counter[str] = Counter()
        for doc in self._corpus_tokens:
            for term in set(doc):
                df[term] += 1
        self._idf = {
            term: math.log((self._n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        log.info(f"  BM25 ajusté sur {self._n} documents (avgdl={self._avgdl:.1f})")

    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        doc  = self._corpus_tokens[doc_idx]
        tf   = Counter(doc)
        dl   = len(doc)
        k1   = self.k1
        b    = self.b
        avgdl = self._avgdl
        score = 0.0
        for term in query_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            f = tf[term]
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        return score

    def rank_all(
        self,
        queries: dict[str, str],
        corpus:  dict[str, str],
        top_k:   int = 100,
    ) -> dict[str, list[str]]:
        corpus_ids   = list(corpus.keys())
        corpus_texts = [corpus[d] for d in corpus_ids]

        if self._corpus_changed(corpus_ids):
            self._fit(corpus_ids, corpus_texts)

        results: dict[str, list[str]] = {}
        for qid, qtext in queries.items():
            q_tokens = self._tokenize(qtext)
            scores   = [self._score_doc(q_tokens, i) for i in range(self._n)]
            ranked   = sorted(range(self._n), key=lambda i: -scores[i])[:top_k]
            results[qid] = [corpus_ids[j] for j in ranked]
        return results

    def measure_latency(self, texts: list[str], batch_size: int = 64) -> float:
        # Mesure le temps de tokenisation (goulot d'étranglement pour BM25)
        t0 = time.perf_counter()
        for text in texts:
            self._tokenize(text)
        return round((time.perf_counter() - t0) * 1000.0 / max(len(texts), 1), 3)

    def get_size_mb(self) -> float:
        return 5.0

    def get_dim(self) -> int:
        return 0


# ─────────────────────────────────────────────────────────────────────────
# Fabrique depuis la configuration JSON
# ─────────────────────────────────────────────────────────────────────────

def load_model_from_config(cfg: dict, online: bool = False) -> SentenceTransformerModel:
    """Instancie un SentenceTransformerModel depuis un dict de configuration."""
    return SentenceTransformerModel(
        name=cfg["name"],
        hf_id=cfg["hf_id"],
        label=cfg["label"],
        family=cfg["family"],
        local=cfg.get("local", False),
        prefix=cfg.get("prefix", ""),
        online=online,
    )
