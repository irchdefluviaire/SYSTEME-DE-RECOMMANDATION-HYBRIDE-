"""
Ablation pgvector vs pgvector + Neo4j graph reranking.

This module measures the marginal effect of graph enrichment on the candidate
to job-offer ranking. It intentionally avoids LLM generation: the comparison is
focused on retrieval/reranking, not on answer style.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "evaluation" / "pgvector_vs_graph"
FIG_DIR = ROOT / "rapport" / "figures" / "generated" / "evaluation"

for module_path in (
    ROOT / "src" / "05_graphrag",
    ROOT / "src" / "08_agentic_graphrag",
):
    module_path_str = str(module_path)
    if module_path_str not in sys.path:
        sys.path.insert(0, module_path_str)

load_dotenv(ROOT / ".env")

from context_builder import GraphRAGContextBuilder  # noqa: E402
from tools import get_neo4j_driver, get_pg_conn, get_st_model  # noqa: E402


@dataclass(frozen=True)
class EvalConfig:
    n_candidats: int = 80
    candidate_pool_k: int = 30
    final_k: int = 10
    random_seed: int = 42


def _ascii(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", str(text or ""))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )


def tokenize(text: str) -> set[str]:
    stop = {
        "avec", "dans", "pour", "plus", "poste", "profil", "emploi", "offre",
        "candidat", "competence", "competences", "niveau", "secteur", "type",
        "contrat", "experience", "formation", "avoir", "vous", "nous", "les",
        "des", "une", "sur", "par", "aux", "est", "sont", "and", "the",
    }
    cleaned = re.sub(r"[^a-z0-9]+", " ", _ascii(text))
    return {tok for tok in cleaned.split() if len(tok) >= 4 and tok not in stop}


def safe_int(value: Any) -> int | None:
    try:
        if value is None or str(value) in {"", "nan", "NaN", "<NA>", "None"}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def metric_values(relevant: set[str], retrieved: list[str], k: int) -> dict[str, float]:
    top = retrieved[:k]
    hits = [1 if item in relevant else 0 for item in top]
    precision = sum(hits) / k if k else 0.0
    recall = sum(hits) / len(relevant) if relevant else 0.0
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal_hits = [1] * min(k, len(relevant))
    idcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    mrr = 0.0
    for i, hit in enumerate(hits, 1):
        if hit:
            mrr = 1.0 / i
            break
    hit_rate = 1.0 if sum(hits) else 0.0
    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"ndcg@{k}": ndcg,
        f"mrr@{k}": mrr,
        f"hit_rate@{k}": hit_rate,
    }


def aggregate_metrics(per_query: list[dict[str, Any]], system: str) -> dict[str, Any]:
    rows = [row for row in per_query if row["system"] == system]
    numeric_cols = [
        col for col in rows[0].keys()
        if col not in {"query_id", "candidat_id", "system", "error"} and isinstance(rows[0][col], (int, float))
    ] if rows else []
    out = {"system": system, "n_queries": len(rows)}
    for col in numeric_cols:
        vals = [float(row[col]) for row in rows]
        out[col] = round(float(np.mean(vals)), 4)
    return out


def _candidate_profile(row: pd.Series) -> dict[str, Any]:
    return {
        "candidat_id": str(row["candidat_id"]),
        "metier_vise": str(row.get("metier_vise", "") or ""),
        "secteur_metier": str(row.get("secteur_metier", "") or ""),
        "ncf_niveau_final": safe_int(row.get("ncf_niveau_final")),
        "filiere_specialite": str(row.get("filiere_specialite", "") or ""),
        "objectif": str(row.get("objectif", "") or "")[:200],
        "diplome_raw": str(row.get("diplome_raw", "") or ""),
        "secteur_demande": str(row.get("secteur_demande", "") or ""),
        "mobilite_geo_bool": row.get("mobilite_geo_bool"),
    }


def _merge_offer_fields(offre: dict[str, Any], offres_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw = offres_by_id.get(str(offre.get("offre_id")), {})
    return {
        **offre,
        "titre": offre.get("titre") or raw.get("titre_poste", ""),
        "secteur": offre.get("secteur") or raw.get("secteur_principal", ""),
        "ville": offre.get("ville") or raw.get("ville_principale", ""),
        "type_contrat": offre.get("type_contrat") or raw.get("type_contrat_norm", ""),
        "ncf_code": offre.get("ncf_code") or raw.get("ncf_niveau_code"),
        "skills": offre.get("skills") or raw.get("skills_raw", ""),
        "details": offre.get("details") or str(raw.get("details_clean", ""))[:300],
    }


def relevance_score(candidat: dict[str, Any], offre: dict[str, Any]) -> float:
    """Independent proxy label from normalized metadata, not from LLM output."""

    cand_tokens = tokenize(
        " ".join(
            [
                candidat.get("metier_vise", ""),
                candidat.get("secteur_metier", ""),
                candidat.get("filiere_specialite", ""),
                candidat.get("secteur_demande", ""),
                candidat.get("objectif", ""),
            ]
        )
    )
    offer_tokens = tokenize(
        " ".join(
            [
                offre.get("titre", ""),
                offre.get("secteur", ""),
                offre.get("skills", ""),
                offre.get("details", ""),
            ]
        )
    )
    overlap = len(cand_tokens & offer_tokens)
    lexical_fit = overlap / max(min(len(cand_tokens), len(offer_tokens)), 1)

    cand_sector = tokenize(candidat.get("secteur_metier", ""))
    offer_sector = tokenize(offre.get("secteur", ""))
    sector_fit = len(cand_sector & offer_sector) / max(len(cand_sector), 1) if cand_sector else 0.0

    cand_ncf = safe_int(candidat.get("ncf_niveau_final"))
    offer_ncf = safe_int(offre.get("ncf_code"))
    if cand_ncf is None or offer_ncf is None:
        ncf_fit = 0.45
    elif cand_ncf >= offer_ncf:
        ncf_fit = 1.0
    elif offer_ncf - cand_ncf == 1:
        ncf_fit = 0.70
    elif offer_ncf - cand_ncf == 2:
        ncf_fit = 0.40
    else:
        ncf_fit = 0.10

    return round(float(0.45 * lexical_fit + 0.30 * sector_fit + 0.25 * ncf_fit), 4)


def relevant_set_for_pool(
    candidat: dict[str, Any],
    pool: list[dict[str, Any]],
    quantile: float = 0.70,
) -> tuple[set[str], list[dict[str, Any]]]:
    scored = []
    for offre in pool:
        rel = relevance_score(candidat, offre)
        scored.append({**offre, "proxy_relevance": rel})
    scores = [row["proxy_relevance"] for row in scored]
    threshold = max(0.35, float(np.quantile(scores, quantile))) if scores else 1.0
    relevant = {str(row["offre_id"]) for row in scored if row["proxy_relevance"] >= threshold}
    if not relevant and scored:
        best = max(scored, key=lambda row: row["proxy_relevance"])
        relevant = {str(best["offre_id"])}
    return relevant, scored


def _plot_metric_comparison(metrics_df: pd.DataFrame) -> None:
    selected = ["precision@10", "recall@10", "ndcg@10", "mrr@10", "hit_rate@10"]
    plot_df = metrics_df.set_index("system")[selected].T
    ax = plot_df.plot(kind="bar", figsize=(9.5, 5), color=["#2f6fdd", "#f29700"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score moyen")
    ax.set_title("Ablation retrieval: pgvector seul vs pgvector + graphe")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    for path in (OUT_DIR / "ablation_metrics_comparison.png", FIG_DIR / "ablation_metrics_comparison.png"):
        ax.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)


def _plot_rank_shift(details_df: pd.DataFrame) -> None:
    if details_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(details_df["rank_shift"].dropna(), bins=15, color="#455a64", edgecolor="white")
    ax.axvline(0, color="#b00020", linewidth=1.2)
    ax.set_title("Déplacement des offres après reranking graphe")
    ax.set_xlabel("Rang vectoriel - rang graphe")
    ax.set_ylabel("Nombre d'offres")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    for path in (OUT_DIR / "graph_rerank_rank_shift.png", FIG_DIR / "graph_rerank_rank_shift.png"):
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_ablation(config: EvalConfig | None = None) -> dict[str, Any]:
    config = config or EvalConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df_c = pd.read_parquet(PROC / "candidats_normalized.parquet")
    df_o = pd.read_parquet(PROC / "offres_normalized.parquet")
    offres_by_id = {
        str(row["offre_id"]): row
        for row in df_o.to_dict(orient="records")
    }

    sample = df_c.sample(min(config.n_candidats, len(df_c)), random_state=config.random_seed)

    pg_conn = get_pg_conn()
    st_model = get_st_model()
    neo4j_driver = get_neo4j_driver()
    builder = GraphRAGContextBuilder(
        neo4j_driver=neo4j_driver,
        pg_conn=pg_conn,
        st_model=st_model,
        top_k_pgvector=config.candidate_pool_k,
        top_k_final=config.final_k,
    )

    per_query: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    t0 = time.time()

    for _, cand_row in sample.iterrows():
        candidat = _candidate_profile(cand_row)
        cid = candidat["candidat_id"]
        try:
            pool_raw = builder._ann_search(cid)
            pool = [_merge_offer_fields(row, offres_by_id) for row in pool_raw]
            relevant, pool_scored = relevant_set_for_pool(candidat, pool)

            vector_ranked = [str(row["offre_id"]) for row in pool_scored]

            enriched = []
            for row in pool_scored:
                enriched_row = builder._enrich_with_neo4j(cid, row)
                enriched_row = _merge_offer_fields(enriched_row, offres_by_id)
                enriched_row["score_hybride"] = builder._compute_hybrid_score(enriched_row)
                enriched.append(enriched_row)
            enriched.sort(key=lambda row: row["score_hybride"], reverse=True)
            graph_ranked = [str(row["offre_id"]) for row in enriched]

            for system, ranked in (
                ("pgvector_only", vector_ranked),
                ("pgvector_plus_graph", graph_ranked),
            ):
                row = {
                    "query_id": cid,
                    "candidat_id": cid,
                    "system": system,
                    "n_relevant_in_pool": len(relevant),
                }
                for k in (1, 3, 5, 10):
                    row.update(metric_values(relevant, ranked, k))
                per_query.append(row)

            vector_pos = {oid: i + 1 for i, oid in enumerate(vector_ranked)}
            for i, row in enumerate(enriched[: config.final_k], 1):
                oid = str(row["offre_id"])
                detail_rows.append(
                    {
                        "candidat_id": cid,
                        "offre_id": oid,
                        "titre": row.get("titre", ""),
                        "proxy_relevance": row.get("proxy_relevance"),
                        "score_sem": row.get("score_sem"),
                        "score_hybride": row.get("score_hybride"),
                        "taux_match": row.get("taux_match"),
                        "verdict_recrutement": row.get("verdict_recrutement"),
                        "rank_pgvector": vector_pos.get(oid),
                        "rank_graph": i,
                        "rank_shift": (vector_pos.get(oid) or i) - i,
                    }
                )
        except Exception as exc:
            errors.append({"candidat_id": cid, "error": str(exc)})
            try:
                pg_conn.rollback()
            except Exception:
                pass

    metrics = [
        aggregate_metrics(per_query, "pgvector_only"),
        aggregate_metrics(per_query, "pgvector_plus_graph"),
    ]
    metrics_df = pd.DataFrame(metrics)
    per_query_df = pd.DataFrame(per_query)
    detail_df = pd.DataFrame(detail_rows)

    delta = {}
    if len(metrics_df) == 2:
        base = metrics_df.set_index("system").loc["pgvector_only"]
        graph = metrics_df.set_index("system").loc["pgvector_plus_graph"]
        for col in metrics_df.columns:
            if col in {"system", "n_queries"}:
                continue
            delta[col] = round(float(graph[col] - base[col]), 4)

    metrics_df.to_csv(OUT_DIR / "ablation_metrics.csv", index=False, encoding="utf-8")
    per_query_df.to_csv(OUT_DIR / "ablation_per_query.csv", index=False, encoding="utf-8")
    detail_df.to_csv(OUT_DIR / "ablation_graph_rerank_details.csv", index=False, encoding="utf-8")
    if errors:
        pd.DataFrame(errors).to_csv(OUT_DIR / "ablation_errors.csv", index=False, encoding="utf-8")

    _plot_metric_comparison(metrics_df)
    _plot_rank_shift(detail_df)

    summary = {
        "config": config.__dict__,
        "elapsed_s": round(time.time() - t0, 2),
        "n_queries_requested": int(len(sample)),
        "n_queries_evaluated": int(metrics_df["n_queries"].max()) if not metrics_df.empty else 0,
        "n_errors": len(errors),
        "metrics": metrics_df.to_dict(orient="records"),
        "delta_pgvector_plus_graph_minus_pgvector": delta,
        "outputs_dir": str(OUT_DIR.relative_to(ROOT)),
        "figures_dir": str(FIG_DIR.relative_to(ROOT)),
        "methodological_warning": (
            "La pertinence est un proxy construit a partir des metadonnees normalisees "
            "(chevauchement lexical, secteur, niveau NCF). Ce n'est pas une annotation humaine."
        ),
    }
    with open(OUT_DIR / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    result = run_ablation()
    print(json.dumps(result, ensure_ascii=False, indent=2))
