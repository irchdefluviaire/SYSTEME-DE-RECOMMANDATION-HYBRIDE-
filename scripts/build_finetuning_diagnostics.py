from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[1]
DATA_FT = ROOT / "data" / "finetune"
MODEL_PATH = ROOT / "models" / "st_finetuned" / "final"
METRICS_PATH = ROOT / "models" / "st_finetuned" / "evaluation_metrics.json"
OUT = ROOT / "outputs" / "memoire_stats" / "implementation_deep_current"
FIG = ROOT / "rapport" / "figures" / "generated" / "implementation_deep_current"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_loss(
    model: SentenceTransformer,
    pairs: list[dict],
    batch_size: int,
    scale: float,
    sample_size: int = 64,
) -> dict[str, float]:
    if len(pairs) > sample_size:
        rng = random.Random(42)
        pairs = rng.sample(pairs, sample_size)
    anchors_all = [p["sentence1"] for p in pairs]
    positives_all = [p["sentence2"] for p in pairs]
    emb_anchors = model.encode(
        anchors_all,
        batch_size=128,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    emb_positives = model.encode(
        positives_all,
        batch_size=128,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    total_loss = 0.0
    total_examples = 0
    batch_losses: list[float] = []

    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        if end - start < 2:
            continue
        emb_a = emb_anchors[start:end]
        emb_p = emb_positives[start:end]
        scores = util.cos_sim(emb_a, emb_p) * scale
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = F.cross_entropy(scores, labels)
        n = scores.size(0)
        total_loss += float(loss.detach().cpu()) * n
        total_examples += n
        batch_losses.append(float(loss.detach().cpu()))

    return {
        "loss_mnrl": total_loss / total_examples if total_examples else float("nan"),
        "n_examples": total_examples,
        "n_batches": len(batch_losses),
        "sampled": len(pairs) >= sample_size,
    }


def build_figure(losses: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    logs = pd.DataFrame(metrics["training_logs"])
    train_logs = logs[logs["loss"].notna()]
    ndcg_col = [c for c in logs.columns if c.endswith("ndcg@10")][0]
    mrr_col = [c for c in logs.columns if c.endswith("mrr@10")][0]
    eval_logs = logs[logs[ndcg_col].notna()]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))
    ax_loss, ax_split, ax_metric, ax_lr = axes.ravel()

    ax_loss.plot(train_logs["epoch"], train_logs["loss"], color="#2457A6", marker="o", linewidth=1.8)
    ax_loss.set_ylabel("Loss train")
    ax_loss.set_xlabel("Epoque")
    ax_loss.set_title("1. Apprentissage : chute rapide puis stabilisation")
    if not train_logs.empty:
        start_loss = float(train_logs["loss"].iloc[0])
        end_loss = float(train_logs["loss"].iloc[-1])
        ax_loss.annotate(
            f"{start_loss:.3f} -> {end_loss:.3f}",
            xy=(train_logs["epoch"].iloc[-1], end_loss),
            xytext=(-85, 25),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#374151"},
            fontsize=9,
        )

    colors = ["#2457A6", "#E68619", "#3A7D44"]
    ax_split.bar(losses["split"], losses["loss_mnrl"], color=colors[: len(losses)])
    ax_split.set_ylabel("Loss finale")
    ax_split.set_title("2. Generalisation : splits non vus plus difficiles")
    for i, row in enumerate(losses.itertuples()):
        ax_split.text(i, row.loss_mnrl, f"{row.loss_mnrl:.3f}", ha="center", va="bottom", fontsize=9)
    ax_split.text(
        0.02,
        0.93,
        "64 paires par split pour limiter le cout de reevaluation",
        transform=ax_split.transAxes,
        fontsize=8.5,
        color="#374151",
        va="top",
    )

    ax_metric.plot(eval_logs["epoch"], eval_logs[ndcg_col], color="#3A7D44", marker="o", linewidth=2.0, label="NDCG@10")
    ax_metric.plot(eval_logs["epoch"], eval_logs[mrr_col], color="#E68619", marker="s", linewidth=1.8, label="MRR@10")
    ax_metric.set_ylim(0.54, 0.67)
    ax_metric.set_ylabel("Score")
    ax_metric.set_xlabel("Epoque")
    ax_metric.set_title("3. Validation : le classement progresse")
    ax_metric.legend(loc="lower right")

    ax_lr.plot(train_logs["epoch"], train_logs["learning_rate"], color="#B13E3E", linewidth=2.0)
    ax_lr.fill_between(train_logs["epoch"], train_logs["learning_rate"], color="#B13E3E", alpha=0.12)
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_xlabel("Epoque")
    ax_lr.set_title("4. Calendrier leger : warmup puis decroissance")
    ax_lr.text(
        0.02,
        0.93,
        "LR max 2e-5, batch 32, 5 epoques : choix sobre pour ressources limitees",
        transform=ax_lr.transAxes,
        fontsize=8.5,
        color="#374151",
        va="top",
    )

    fig.suptitle("Fine-tuning SentenceTransformer : cout maitrise et gain de classement", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "04_finetuning_courbes.png", bbox_inches="tight")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    batch_size = int(metrics.get("batch_size", 32))
    scale = 20.0
    model = SentenceTransformer(str(MODEL_PATH), local_files_only=True)

    rows = []
    for split in ["train", "val", "test"]:
        pairs = load_jsonl(DATA_FT / f"pairs_{split}.jsonl")
        row = split_loss(model, pairs, batch_size=batch_size, scale=scale)
        row["split"] = split
        rows.append(row)

    losses = pd.DataFrame(rows)[["split", "n_examples", "n_batches", "sampled", "loss_mnrl"]]
    losses.to_csv(OUT / "04_finetuning_loss_par_split.csv", index=False, encoding="utf-8-sig")
    build_figure(losses)
    print(losses.to_string(index=False))


if __name__ == "__main__":
    main()
