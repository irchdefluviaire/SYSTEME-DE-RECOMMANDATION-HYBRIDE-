"""
calibrate_critic_threshold.py
===========================================================================
Calibrage statistique du seuil du critic GraphRAG.

Le critic local (`answer_critic.py`) produit un score lexical de fidelite au
contexte. Ce script calibre son seuil de decision en utilisant :
  1. RAGAS Faithfulness comme juge de reference si un LLM est disponible ;
  2. sinon une colonne d'annotation humaine (`human_label`) deja preparee.

Format d'entree CSV/JSONL attendu :
  - query_id ou id              : identifiant optionnel
  - user_input ou question      : question utilisateur
  - response ou answer          : reponse generee
  - retrieved_contexts/context  : liste JSON de contextes, ou texte brut
  - human_label                 : optionnel, accept|revise|1|0

Sorties :
  - critic_calibration_scores.csv      : scores par exemple
  - critic_threshold_grid.csv          : metriques par seuil
  - critic_threshold_summary.json      : seuil recommande et parametres
  - critic_threshold_curves.png        : precision/rappel/F1 selon seuil
  - critic_score_distributions.png     : distribution des scores par reference
  - critic_confusion_matrix.png        : matrice de confusion au seuil retenu
===========================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "05_graphrag"))

from answer_critic import critique_answer  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "evaluation" / "critic_calibration"
DEFAULT_REPORT_FIGURES_DIR = ROOT / "rapport" / "figures" / "generated" / "evaluation"
Z_95 = 1.959963984540054


def _parse_contexts(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
        if isinstance(parsed, dict):
            return [json.dumps(parsed, ensure_ascii=False)]
    except json.JSONDecodeError:
        pass

    if "\n---\n" in text:
        return [part.strip() for part in text.split("\n---\n") if part.strip()]
    return [text]


def _read_cases(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)

    rename = {}
    if "id" in df.columns and "query_id" not in df.columns:
        rename["id"] = "query_id"
    if "question" in df.columns and "user_input" not in df.columns:
        rename["question"] = "user_input"
    if "answer" in df.columns and "response" not in df.columns:
        rename["answer"] = "response"
    if "context" in df.columns and "retrieved_contexts" not in df.columns:
        rename["context"] = "retrieved_contexts"
    df = df.rename(columns=rename)

    required = {"user_input", "response", "retrieved_contexts"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Colonnes manquantes dans le fichier de calibration : " + ", ".join(missing))

    if "query_id" not in df.columns:
        df["query_id"] = [f"case_{i:04d}" for i in range(len(df))]

    df["retrieved_contexts"] = df["retrieved_contexts"].apply(_parse_contexts)
    df = df[df["response"].fillna("").astype(str).str.strip().ne("")]
    df = df[df["retrieved_contexts"].apply(bool)]
    return df.reset_index(drop=True)


def _label_to_binary(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"accept", "accepted", "oui", "yes", "true", "1", "bon", "correct"}:
        return 1
    if text in {"revise", "revision", "reject", "rejet", "non", "no", "false", "0", "mauvais", "incorrect"}:
        return 0
    return None


def _score_local_critic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        context = row["retrieved_contexts"]
        critic = critique_answer(row["response"], context, min_faithfulness=0.0)
        rows.append(
            {
                "query_id": row["query_id"],
                "critic_faithfulness": float(critic["faithfulness"]),
                "critic_evidence_coverage": float(critic["evidence_coverage"]),
                "critic_unsupported_terms": json.dumps(critic["unsupported_terms"], ensure_ascii=False),
            }
        )
    return df.merge(pd.DataFrame(rows), on="query_id", how="left")


def _build_ragas_llm(provider: str, model: str):
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv(ROOT / ".env")

    provider = provider.lower()
    if provider == "openrouter":
        api_key = os.getenv("API_KEY_OPEN_ROUTEUR") or os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = None
    else:
        raise ValueError("Provider RAGAS non supporte : utiliser openrouter ou openai.")

    if not api_key:
        raise RuntimeError(f"Cle API absente pour {provider}.")

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": 0.0,
        "timeout": 90,
        "max_retries": 2,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _score_ragas(df: pd.DataFrame, *, provider: str, model: str) -> pd.DataFrame:
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.metrics import Faithfulness

    samples = [
        {
            "user_input": str(row["user_input"]),
            "response": str(row["response"]),
            "retrieved_contexts": list(row["retrieved_contexts"]),
        }
        for _, row in df.iterrows()
    ]

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[Faithfulness()],
        llm=_build_ragas_llm(provider, model),
        raise_exceptions=False,
    )
    ragas_df = result.to_pandas()
    df = df.copy()
    df["ragas_faithfulness"] = pd.to_numeric(ragas_df["faithfulness"], errors="coerce")
    return df


def _wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Intervalle de Wilson pour une proportion binomiale."""
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    margin = z * np.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2)))
    low = (centre - margin) / denom
    high = (centre + margin) / denom
    return max(0.0, float(low)), min(1.0, float(high))


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    regular_grid = np.round(np.arange(0.0, 1.001, 0.01), 4)
    observed = np.unique(np.round(scores.astype(float), 4))
    around_observed = np.concatenate(
        [
            observed,
            np.clip(observed - 0.0001, 0.0, 1.0),
            np.clip(observed + 0.0001, 0.0, 1.0),
        ]
    )
    return np.unique(np.round(np.concatenate([regular_grid, around_observed]), 4))


def _metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    fbeta_beta: float = 0.5,
) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    beta2 = fbeta_beta**2
    fbeta = (
        (1 + beta2) * precision * recall / ((beta2 * precision) + recall)
        if ((beta2 * precision) + recall)
        else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2
    false_accept_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_revise_rate = fn / (fn + tp) if (fn + tp) else 0.0
    precision_low, precision_high = _wilson_interval(tp, tp + fp)
    false_accept_low, false_accept_high = _wilson_interval(fp, fp + tn)

    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_accept": round(precision, 4),
        "recall_accept": round(recall, 4),
        "specificity_revise": round(specificity, 4),
        "f1_accept": round(f1, 4),
        "fbeta_accept": round(fbeta, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "false_accept_rate": round(false_accept_rate, 4),
        "false_revise_rate": round(false_revise_rate, 4),
        "predicted_accept_rate": round(float(y_pred.mean()), 4),
        "n_predicted_accept": int(tp + fp),
        "precision_accept_wilson_low": round(precision_low, 4),
        "precision_accept_wilson_high": round(precision_high, 4),
        "false_accept_rate_wilson_low": round(false_accept_low, 4),
        "false_accept_rate_wilson_high": round(false_accept_high, 4),
    }


def _bootstrap_threshold_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    if n_bootstrap <= 0:
        return {}

    rng = np.random.default_rng(seed)
    stats = []
    n = len(scores)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats.append(_metrics_at_threshold(y_true[idx], scores[idx], threshold))

    out = {}
    for key in ["precision_accept", "recall_accept", "f1_accept", "false_accept_rate"]:
        vals = np.array([s[key] for s in stats], dtype=float)
        out[f"{key}_ci_low"] = round(float(np.quantile(vals, 0.025)), 4)
        out[f"{key}_ci_high"] = round(float(np.quantile(vals, 0.975)), 4)
    return out


def _plot_threshold_curves(grid: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(grid["threshold"], grid["precision_accept"], label="Precision accept", linewidth=2.2)
    ax.plot(grid["threshold"], grid["recall_accept"], label="Recall accept", linewidth=2.2)
    ax.plot(grid["threshold"], grid["f1_accept"], label="F1 accept", linewidth=2.2)
    ax.plot(
        grid["threshold"],
        1 - grid["false_accept_rate"],
        label="1 - taux de fausse acceptation",
        linewidth=2.0,
        linestyle="--",
    )
    best_threshold = float(summary["threshold"])
    ax.axvline(best_threshold, color="#C62828", linestyle=":", linewidth=2.4, label=f"Seuil retenu = {best_threshold:.2f}")
    ax.set_xlabel("Seuil du critic lexical")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.03)
    ax.set_title("Calibration du seuil du critic : compromis precision-rappel")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_score_distributions(df: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accepted = df.loc[df["y_true_accept"] == 1, "critic_faithfulness"].astype(float)
    revised = df.loc[df["y_true_accept"] == 0, "critic_faithfulness"].astype(float)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bins = np.linspace(0, max(0.5, float(df["critic_faithfulness"].max()) + 0.02), 24)
    ax.hist(revised, bins=bins, alpha=0.65, label="Reference: revise", color="#C62828")
    ax.hist(accepted, bins=bins, alpha=0.65, label="Reference: accept", color="#1565C0")
    best_threshold = float(summary["threshold"])
    ax.axvline(best_threshold, color="#111111", linestyle=":", linewidth=2.4, label=f"Seuil retenu = {best_threshold:.2f}")
    ax.set_xlabel("Score lexical du critic")
    ax.set_ylabel("Nombre de cas")
    ax.set_title("Distribution des scores du critic selon la decision de reference")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_confusion_matrix(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array([[summary["tn"], summary["fp"]], [summary["fn"], summary["tp"]]], dtype=float)
    labels = np.array([["TN\nrevise -> revise", "FP\nrevise -> accept"], ["FN\naccept -> revise", "TP\naccept -> accept"]])

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Prediction revise", "Prediction accept"])
    ax.set_yticks([0, 1], labels=["Reference revise", "Reference accept"])
    ax.set_title(f"Matrice de confusion au seuil {float(summary['threshold']):.2f}")

    max_value = matrix.max() if matrix.size else 0
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > max_value / 2 else "black"
            ax.text(j, i, f"{labels[i, j]}\n{int(matrix[i, j])}", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _copy_to_report(src: Path, report_dir: Path) -> Path:
    src = src.resolve()
    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    dst = report_dir / src.name
    dst.write_bytes(src.read_bytes())
    return dst


def _write_plots(
    df: pd.DataFrame,
    grid: pd.DataFrame,
    summary: dict[str, Any],
    *,
    output_dir: Path,
    report_figures_dir: Path | None,
) -> dict[str, str]:
    output_dir = output_dir.resolve()
    if report_figures_dir is not None:
        report_figures_dir = report_figures_dir.resolve()

    plot_paths = {
        "threshold_curves": output_dir / "critic_threshold_curves.png",
        "score_distributions": output_dir / "critic_score_distributions.png",
        "confusion_matrix": output_dir / "critic_confusion_matrix.png",
    }

    _plot_threshold_curves(grid, summary, plot_paths["threshold_curves"])
    _plot_score_distributions(df, summary, plot_paths["score_distributions"])
    _plot_confusion_matrix(summary, plot_paths["confusion_matrix"])

    report_paths = {}
    if report_figures_dir is not None:
        for key, path in plot_paths.items():
            report_paths[key] = str(_copy_to_report(path, report_figures_dir).relative_to(ROOT))
    return {key: str(path.relative_to(ROOT)) for key, path in plot_paths.items()} | {
        f"report_{key}": value for key, value in report_paths.items()
    }


def _calibrate(
    df: pd.DataFrame,
    *,
    label_source: str,
    ragas_accept_threshold: float,
    target_precision: float,
    max_false_accept_rate: float,
    min_predicted_accept: int,
    fbeta_beta: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()

    if label_source == "ragas":
        if "ragas_faithfulness" not in df.columns:
            raise ValueError("Scores RAGAS absents.")
        df["y_true_accept"] = (df["ragas_faithfulness"] >= ragas_accept_threshold).astype(int)
    elif label_source == "human":
        if "human_label" not in df.columns:
            raise ValueError("Colonne human_label absente.")
        labels = df["human_label"].apply(_label_to_binary)
        df = df[labels.notna()].copy()
        df["y_true_accept"] = labels[labels.notna()].astype(int).to_numpy()
    else:
        raise ValueError("label_source doit etre ragas ou human.")

    y_true = df["y_true_accept"].to_numpy(dtype=int)
    scores = df["critic_faithfulness"].to_numpy(dtype=float)

    thresholds = _threshold_candidates(scores)
    grid = pd.DataFrame([_metrics_at_threshold(y_true, scores, t, fbeta_beta=fbeta_beta) for t in thresholds])

    statistically_feasible = grid[
        (grid["precision_accept_wilson_low"] >= target_precision)
        & (grid["false_accept_rate_wilson_high"] <= max_false_accept_rate)
        & (grid["tp"] > 0)
        & (grid["n_predicted_accept"] >= min_predicted_accept)
    ].copy()
    point_feasible = grid[
        (grid["precision_accept"] >= target_precision)
        & (grid["false_accept_rate"] <= max_false_accept_rate)
        & (grid["tp"] > 0)
        & (grid["n_predicted_accept"] >= min_predicted_accept)
    ].copy()

    if not statistically_feasible.empty:
        feasible = statistically_feasible
        selection_rule = "wilson_constrained_precision_and_false_accept_rate"
    elif not point_feasible.empty:
        feasible = point_feasible
        selection_rule = "point_estimate_constraints_wilson_inconclusive"
    else:
        feasible = grid[(grid["tp"] > 0) & (grid["n_predicted_accept"] >= min_predicted_accept)].copy()
        if feasible.empty:
            feasible = grid[grid["tp"] > 0].copy()
        selection_rule = "fallback_max_fbeta_no_constraint"

    feasible = feasible.sort_values(
        [
            "fbeta_accept",
            "precision_accept_wilson_low",
            "false_accept_rate_wilson_high",
            "recall_accept",
            "threshold",
        ],
        ascending=[False, False, True, False, True],
    )
    best = feasible.iloc[0].to_dict()
    best.update(
        {
            "selection_rule": selection_rule,
            "label_source": label_source,
            "ragas_accept_threshold": ragas_accept_threshold if label_source == "ragas" else None,
            "target_precision": target_precision,
            "max_false_accept_rate": max_false_accept_rate,
            "min_predicted_accept": min_predicted_accept,
            "fbeta_beta": fbeta_beta,
            "n_cases": int(len(df)),
            "positive_rate_reference": round(float(y_true.mean()), 4),
            "n_thresholds_evaluated": int(len(grid)),
            "n_thresholds_wilson_feasible": int(len(statistically_feasible)),
            "n_thresholds_point_feasible": int(len(point_feasible)),
        }
    )
    best.update(
        _bootstrap_threshold_ci(
            y_true,
            scores,
            threshold=float(best["threshold"]),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    )
    return df, grid, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="CSV ou JSONL de cas GraphRAG.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-figures-dir", type=Path, default=DEFAULT_REPORT_FIGURES_DIR)
    parser.add_argument("--no-report-figures", action="store_true")
    parser.add_argument("--label-source", choices=["ragas", "human"], default="ragas")
    parser.add_argument("--skip-ragas", action="store_true", help="Utilise uniquement human_label.")
    parser.add_argument("--ragas-provider", choices=["openrouter", "openai"], default="openrouter")
    parser.add_argument("--ragas-model", default=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"))
    parser.add_argument("--ragas-accept-threshold", type=float, default=0.80)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--max-false-accept-rate", type=float, default=0.10)
    parser.add_argument("--min-predicted-accept", type=int, default=5)
    parser.add_argument("--fbeta-beta", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.report_figures_dir = args.report_figures_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _read_cases(args.input)
    df = _score_local_critic(df)

    label_source = args.label_source
    if args.skip_ragas:
        label_source = "human"
    elif label_source == "ragas":
        df = _score_ragas(df, provider=args.ragas_provider, model=args.ragas_model)

    df_calibrated, grid, summary = _calibrate(
        df,
        label_source=label_source,
        ragas_accept_threshold=args.ragas_accept_threshold,
        target_precision=args.target_precision,
        max_false_accept_rate=args.max_false_accept_rate,
        min_predicted_accept=args.min_predicted_accept,
        fbeta_beta=args.fbeta_beta,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    scores_path = args.output_dir / "critic_calibration_scores.csv"
    grid_path = args.output_dir / "critic_threshold_grid.csv"
    summary_path = args.output_dir / "critic_threshold_summary.json"

    df.to_csv(scores_path, index=False)
    grid.to_csv(grid_path, index=False)
    plot_paths = _write_plots(
        df_calibrated,
        grid,
        summary,
        output_dir=args.output_dir,
        report_figures_dir=None if args.no_report_figures else args.report_figures_dir,
    )
    summary["plots"] = plot_paths
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Scores : {scores_path}")
    print(f"Grille : {grid_path}")
    print(f"Resume : {summary_path}")
    print("Figures :")
    for value in plot_paths.values():
        print(f"  - {value}")


if __name__ == "__main__":
    main()
