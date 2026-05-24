"""
benchmark_embeddings.py
===========================================================================
Module 07 — Benchmark multi-modèles d'embeddings (Hypothèse H3)

Compare le SentenceTransformer fine-tuné à un panel de modèles sur le corpus
d'offres camerounais (jeu de test holdout issu du fine-tuning).

Hypothèse H3 : le fine-tuning sur données locales surpasse les modèles
génériques (y compris multilingues et français) sur ce corpus spécialisé.

Usage :
    python benchmark_embeddings.py                     # tous les modèles
    python benchmark_embeddings.py --include-lexical   # + TF-IDF et BM25
    python benchmark_embeddings.py --models st_finetuned,multilingual-e5-base
    python benchmark_embeddings.py --bootstrap 1000    # IC 95%
    python benchmark_embeddings.py --offline           # cache local seul

Sorties :
    outputs/evaluation/embedding_benchmark.csv
    outputs/evaluation/embedding_benchmark.json
    outputs/evaluation/embedding_benchmark_plot.png
===========================================================================
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent.parent
CFG_DIR = Path(__file__).parent
sys.path.insert(0, str(CFG_DIR))

OUT_DIR = ROOT / "outputs" / "evaluation"
DATA_FT = ROOT / "data" / "finetune"

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────
# Chargement du jeu de test
# ─────────────────────────────────────────────────────────────────────────

def load_test_pairs() -> list[dict]:
    """Charge les paires de test depuis data/finetune/pairs_test.jsonl."""
    path = DATA_FT / "pairs_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Paires de test introuvables : {path}\n"
            "Exécuter d'abord : poetry run python scripts/run_etl.py"
        )
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log.info(f"Paires de test chargées : {len(pairs)}")
    return pairs


def build_ir_dataset(
    pairs: list[dict],
) -> tuple[dict[str, str], dict[str, str], dict[str, set]]:
    """
    Construit le dataset IR depuis les paires de fine-tuning.

    Protocole identique à InformationRetrievalEvaluator (evaluate_st.py) :
      - queries  = sentence1  (métadonnées structurées de l'offre)
      - corpus   = sentence2  (description textuelle + compétences)
      - relevant = self-retrieval {offre_id → offre_id}

    Returns:
        (queries, corpus, relevant)
    """
    queries  = {p["offre_id"]: p["sentence1"] for p in pairs}
    corpus   = {p["offre_id"]: p["sentence2"] for p in pairs}
    relevant = {qid: {qid} for qid in queries}
    return queries, corpus, relevant


# ─────────────────────────────────────────────────────────────────────────
# Évaluation d'un modèle
# ─────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    queries:     dict[str, str],
    corpus:      dict[str, str],
    relevant:    dict[str, set],
    k_values:    list[int] = None,
    n_latency:   int = 200,
    n_bootstrap: int = 0,
    top_k:       int = 100,
) -> dict:
    """
    Évalue un modèle sur le protocole IR.

    Args:
        model       : instance de EmbeddingModel
        queries     : {qid: query_text}
        corpus      : {doc_id: doc_text}
        relevant    : {qid: {doc_id}}
        k_values    : valeurs de K pour P/R/NDCG
        n_latency   : nombre de phrases pour la mesure de latence
        n_bootstrap : ré-échantillonnages bootstrap (0 = désactivé)
        top_k       : taille du pool de candidats par requête

    Returns:
        Dictionnaire de métriques.
    """
    from eval_retrieval import compute_metrics, bootstrap_ci, ndcg_at_k

    if k_values is None:
        k_values = [1, 5, 10]

    log.info(f"\n{'─' * 60}")
    log.info(f"Évaluation : {model.label}")
    log.info(f"{'─' * 60}")

    # ── Ranking ────────────────────────────────────────────────────────
    t_rank = time.perf_counter()
    rankings = model.rank_all(queries, corpus, top_k=top_k)
    rank_ms  = round((time.perf_counter() - t_rank) * 1000.0, 1)

    # ── Métriques IR ───────────────────────────────────────────────────
    ir_input = {
        qid: (relevant[qid], rankings[qid])
        for qid in queries
        if qid in rankings
    }
    metrics = compute_metrics(ir_input, k_values=k_values, mrr_k=10)

    log.info(f"  NDCG@10  : {metrics.get('ndcg@10',  0):.4f}")
    log.info(f"  MRR@10   : {metrics.get('mrr@10',   0):.4f}")
    log.info(f"  Recall@5 : {metrics.get('recall@5', 0):.4f}")
    log.info(f"  Ranking total : {rank_ms}ms")

    # ── Bootstrap IC 95% ───────────────────────────────────────────────
    ci_info: dict = {}
    if n_bootstrap > 0:
        log.info(f"  Bootstrap CI ({n_bootstrap} ré-échantillonnages)…")
        mean_n, lo_n, hi_n = bootstrap_ci(
            ir_input, ndcg_at_k, k=10, n_bootstrap=n_bootstrap
        )
        ci_info["ndcg@10_ci95"] = [lo_n, hi_n]
        log.info(f"  NDCG@10 IC95% : [{lo_n:.4f}, {hi_n:.4f}]")

    # ── Latence d'encodage ─────────────────────────────────────────────
    latency_texts = list(corpus.values())[:n_latency]
    latency_ms    = model.measure_latency(latency_texts)
    log.info(f"  Latence  : {latency_ms:.2f} ms/phrase")

    return {
        "name":                    model.name,
        "label":                   model.label,
        "family":                  model.family,
        "dim":                     model.get_dim(),
        "size_mb":                 model.get_size_mb(),
        "latency_ms_per_sentence": latency_ms,
        "rank_total_ms":           rank_ms,
        "n_queries":               len(ir_input),
        **metrics,
        **ci_info,
    }


# ─────────────────────────────────────────────────────────────────────────
# Chargement des modèles
# ─────────────────────────────────────────────────────────────────────────

def load_models(
    names_filter:    list[str] | None,
    include_lexical: bool,
    online:          bool,
) -> list:
    """
    Instancie les modèles depuis models_to_benchmark.json et, si demandé,
    les baselines lexicales TF-IDF et BM25.
    """
    from eval_embedding import (
        TFIDFModel, BM25Model, load_model_from_config,
    )

    cfg_path = CFG_DIR / "models_to_benchmark.json"
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    models = []
    for model_cfg in cfg["models"]:
        if names_filter and model_cfg["name"] not in names_filter:
            continue
        models.append(load_model_from_config(model_cfg, online=online))

    if include_lexical:
        if names_filter is None or "tfidf" in names_filter:
            models.append(TFIDFModel())
        if names_filter is None or "bm25" in names_filter:
            models.append(BM25Model())

    return models


# ─────────────────────────────────────────────────────────────────────────
# Sorties
# ─────────────────────────────────────────────────────────────────────────

_CSV_COLS = [
    "label", "family", "dim", "size_mb",
    "ndcg@1", "ndcg@5", "ndcg@10",
    "mrr@10",
    "recall@1", "recall@5", "recall@10",
    "precision@1", "precision@5",
    "latency_ms_per_sentence",
]


def save_csv(rows: list[dict], path: Path) -> None:
    df      = pd.DataFrame(rows)
    avail   = [c for c in _CSV_COLS if c in df.columns]
    df[avail].to_csv(path, index=False, float_format="%.4f")
    log.info(f"CSV sauvegardé : {path}")


def save_json(rows: list[dict], path: Path) -> None:
    payload = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_models":  len(rows),
        "models":    rows,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    log.info(f"JSON sauvegardé : {path}")


def save_plot(rows: list[dict], path: Path) -> None:
    """Graphique NDCG@10 × Latence (axe X log-scale)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        cfg_path = CFG_DIR / "models_to_benchmark.json"
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)

        fam_colors = {
            fam: spec["color"]
            for fam, spec in cfg.get("families", {}).items()
        }
        fallback = {
            "generaliste_en": "#4e79a7", "multilingue": "#f28e2b",
            "francais": "#e15759",       "domaine": "#59a14f",
            "lexical": "#9c755f",
        }

        fig, ax = plt.subplots(figsize=(11, 6))
        seen_families: set[str] = set()

        for row in rows:
            lat   = row.get("latency_ms_per_sentence", 0.1)
            ndcg  = row.get("ndcg@10", 0)
            fam   = row.get("family", "unknown")
            color = fam_colors.get(fam, fallback.get(fam, "#aaa"))
            label = row.get("label", row.get("name", "?"))
            ci    = row.get("ndcg@10_ci95")

            lat = max(lat, 0.01)  # éviter log(0)

            if ci:
                yerr = [[ndcg - ci[0]], [ci[1] - ndcg]]
                ax.errorbar(lat, ndcg, yerr=yerr,
                            fmt="o", color=color, capsize=4, ms=8, alpha=0.85)
            else:
                ax.scatter(lat, ndcg, color=color, s=130, zorder=5, alpha=0.9)

            ax.annotate(
                label, (lat, ndcg),
                textcoords="offset points", xytext=(7, 4),
                fontsize=7.5, color=color, fontweight="medium",
            )
            seen_families.add(fam)

        patches = [
            mpatches.Patch(
                color=fam_colors.get(f, fallback.get(f, "#aaa")),
                label=f.replace("_", " ").title(),
            )
            for f in sorted(seen_families)
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=9)

        ax.set_xscale("log")
        ax.set_xlabel("Latence d'encodage (ms / phrase, échelle log)", fontsize=11)
        ax.set_ylabel("NDCG@10", fontsize=11)
        ax.set_title(
            "Benchmark multi-modèles d'embeddings — Hypothèse H3\n"
            "Corpus : offres emploi Cameroun (jeu de test holdout, self-retrieval)",
            fontsize=12,
        )
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_ylim(bottom=0, top=min(1.05, max(r.get("ndcg@10", 0) for r in rows) + 0.1))

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Graphique sauvegardé : {path}")

    except Exception as exc:
        log.warning(f"Impossible de générer le graphique : {exc}")


# ─────────────────────────────────────────────────────────────────────────
# Orchestrateur
# ─────────────────────────────────────────────────────────────────────────

def run(
    names_filter:    list[str] | None = None,
    include_lexical: bool = False,
    n_bootstrap:     int  = 0,
    online:          bool = True,
    k_values:        list[int] | None = None,
    top_k:           int  = 100,
) -> list[dict]:
    """Lance le benchmark et retourne la liste des résultats par modèle."""
    log.info("=" * 65)
    log.info("MODULE 07 — BENCHMARK MULTI-MODÈLES D'EMBEDDINGS")
    log.info("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Données de test ────────────────────────────────────────────────
    pairs = load_test_pairs()
    queries, corpus, relevant = build_ir_dataset(pairs)
    log.info(f"Corpus : {len(corpus)} docs | Requêtes : {len(queries)}")

    # ── Modèles ────────────────────────────────────────────────────────
    models = load_models(names_filter, include_lexical, online)
    log.info(f"Modèles à évaluer : {len(models)}")
    if not models:
        log.warning("Aucun modèle à évaluer.")
        return []

    # ── Évaluation ─────────────────────────────────────────────────────
    all_results: list[dict] = []
    t_start = time.perf_counter()

    for model in models:
        try:
            result = evaluate_model(
                model,
                queries=queries,
                corpus=corpus,
                relevant=relevant,
                k_values=k_values or [1, 5, 10],
                n_bootstrap=n_bootstrap,
                top_k=top_k,
            )
            all_results.append(result)
        except Exception as exc:
            log.error(f"Erreur [{model.name}] : {exc}")

    elapsed = time.perf_counter() - t_start
    log.info(f"\nTemps total benchmark : {elapsed:.1f}s")

    if not all_results:
        log.warning("Aucun résultat — vérifier la disponibilité des modèles.")
        return []

    # ── Classement par NDCG@10 ─────────────────────────────────────────
    all_results.sort(key=lambda r: r.get("ndcg@10", 0), reverse=True)

    log.info("\n" + "=" * 65)
    log.info("CLASSEMENT FINAL — NDCG@10")
    log.info("=" * 65)
    for i, r in enumerate(all_results, 1):
        ci = r.get("ndcg@10_ci95")
        ci_str = f"  IC95%=[{ci[0]:.4f},{ci[1]:.4f}]" if ci else ""
        log.info(
            f"  {i:2}. {r['label']:<44}"
            f"NDCG@10={r.get('ndcg@10', 0):.4f}"
            f"  lat={r.get('latency_ms_per_sentence', 0):.1f}ms/s"
            f"{ci_str}"
        )

    # ── Sauvegarde ─────────────────────────────────────────────────────
    save_csv( all_results, OUT_DIR / "embedding_benchmark.csv")
    save_json(all_results, OUT_DIR / "embedding_benchmark.json")
    save_plot(all_results, OUT_DIR / "embedding_benchmark_plot.png")

    return all_results


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark multi-modèles d'embeddings — Module 07"
    )
    parser.add_argument(
        "--include-lexical", action="store_true",
        help="Inclure les baselines lexicales TF-IDF et BM25",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Modèles à évaluer, noms séparés par virgule "
             "(ex. st_finetuned,multilingual-e5-base)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=0,
        metavar="N",
        help="Ré-échantillonnages bootstrap pour IC 95%% (0 = désactivé)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Mode hors-ligne : utiliser uniquement le cache HuggingFace local",
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Taille du pool de candidats par requête (défaut : 100)",
    )
    args = parser.parse_args()

    names_filter = (
        [n.strip() for n in args.models.split(",")]
        if args.models else None
    )

    run(
        names_filter=names_filter,
        include_lexical=args.include_lexical,
        n_bootstrap=args.bootstrap,
        online=not args.offline,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
