from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "memoire_stats" / "implementation_deep_current"
FIG = ROOT / "rapport" / "figures" / "generated" / "implementation_deep_current"

COLORS = {
    "blue": "#2457A6",
    "teal": "#008B8B",
    "green": "#3A7D44",
    "orange": "#E68619",
    "red": "#B13E3E",
    "purple": "#6F4DA8",
    "gray": "#5F6B7A",
    "light": "#EEF3F8",
    "dark": "#1F2A44",
}

MATCHING_RELATIONS = {
    "POSSEDE",
    "REQUIERT",
    "NECESSITE",
    "CORRESPOND_MEPC",
    "A_NIVEAU",
    "REQUIERT_NIVEAU",
    "REQUIERT_NIVEAU_NCF",
    "DANS_SECTEUR",
    "LOCALISEE_A",
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def setup_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

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

    def fmt_int(x, _pos=None):
        if pd.isna(x):
            return ""
        return f"{int(x):,}".replace(",", " ")

    return plt, FuncFormatter(fmt_int)


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_json(obj: dict[str, Any], name: str) -> Path:
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_fig(fig, name: str) -> Path:
    path = FIG / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return path


def node_label(labels: list[str] | tuple[str, ...] | None) -> str:
    if not labels:
        return "NoLabel"
    return str(labels[0])


def node_name(row: Any) -> str:
    for key in ("name", "preferredLabel", "label", "nom", "titre_poste", "id", "code", "ville"):
        val = row.get(key) if hasattr(row, "get") else None
        if val:
            return str(val)
    return str(row.get("node_key", "")) if hasattr(row, "get") else ""


def query_exact_neo4j(session) -> dict[str, Any]:
    labels = pd.DataFrame(
        [dict(r) for r in session.run(
            """
            CALL db.labels() YIELD label
            CALL {
              WITH label
              MATCH (n)
              WHERE label IN labels(n)
              RETURN count(n) AS n
            }
            RETURN label, n
            ORDER BY n DESC
            """
        )]
    )
    rels = pd.DataFrame(
        [dict(r) for r in session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS relation, count(r) AS n
            ORDER BY n DESC
            """
        )]
    )
    degree = pd.DataFrame(
        [dict(r) for r in session.run(
            """
            MATCH (n)
            WITH n, labels(n)[0] AS label
            OPTIONAL MATCH (n)--()
            WITH label, n, count(*) AS degree
            RETURN label,
                   count(*) AS n_nodes,
                   avg(degree) AS avg_degree,
                   percentileCont(degree, 0.5) AS p50_degree,
                   percentileCont(degree, 0.9) AS p90_degree,
                   min(degree) AS min_degree,
                   max(degree) AS max_degree,
                   stDev(degree) AS sd_degree
            ORDER BY n_nodes DESC
            """
        )]
    )
    global_degree = pd.DataFrame(
        [dict(r) for r in session.run(
            """
            MATCH (n)
            OPTIONAL MATCH (n)--()
            WITH n, count(*) AS degree
            RETURN count(*) AS n_nodes,
                   avg(degree) AS avg_degree,
                   percentileCont(degree, 0.5) AS p50_degree,
                   percentileCont(degree, 0.9) AS p90_degree,
                   min(degree) AS min_degree,
                   max(degree) AS max_degree,
                   stDev(degree) AS sd_degree
            """
        )]
    )
    patterns = pd.DataFrame(
        [dict(r) for r in session.run(
            """
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] AS source, type(r) AS relation, labels(b)[0] AS cible, count(*) AS n
            ORDER BY n DESC
            LIMIT 80
            """
        )]
    )
    totals = dict(
        session.run(
            """
            MATCH (n)
            WITH count(n) AS nodes
            MATCH ()-[r]->()
            RETURN nodes, count(r) AS relationships
            """
        ).single()
    )
    return {"labels": labels, "relations": rels, "degree": degree, "global_degree": global_degree, "patterns": patterns, "totals": totals}


def query_edges(session) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows = []
    edge_rows = []
    query = """
    MATCH (a)-[r]->(b)
    WHERE type(r) IN $relations
    RETURN elementId(a) AS src, labels(a) AS src_labels,
           coalesce(a.preferredLabel, a.label, a.nom, a.titre_poste, a.id, a.code, a.ville) AS src_name,
           type(r) AS relation,
           elementId(b) AS dst, labels(b) AS dst_labels,
           coalesce(b.preferredLabel, b.label, b.nom, b.titre_poste, b.id, b.code, b.ville) AS dst_name
    """
    for rec in session.run(query, relations=sorted(MATCHING_RELATIONS)):
        src = rec["src"]
        dst = rec["dst"]
        src_label = node_label(rec["src_labels"])
        dst_label = node_label(rec["dst_labels"])
        node_rows.append({"node_id": src, "label": src_label, "nom": rec["src_name"] or src})
        node_rows.append({"node_id": dst, "label": dst_label, "nom": rec["dst_name"] or dst})
        edge_rows.append({"src": src, "dst": dst, "relation": rec["relation"], "src_label": src_label, "dst_label": dst_label})
    nodes = pd.DataFrame(node_rows).drop_duplicates("node_id")
    edges = pd.DataFrame(edge_rows)
    return nodes, edges


def build_networkx(nodes: pd.DataFrame, edges: pd.DataFrame):
    import networkx as nx

    g = nx.Graph()
    for row in nodes.itertuples(index=False):
        g.add_node(row.node_id, label=row.label, nom=row.nom)
    for row in edges.itertuples(index=False):
        if row.src != row.dst:
            g.add_edge(row.src, row.dst, relation=row.relation)
    return g


def graph_descriptive(nx_graph, exact: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    import networkx as nx

    n = nx_graph.number_of_nodes()
    m = nx_graph.number_of_edges()
    comps = sorted(nx.connected_components(nx_graph), key=len, reverse=True)
    giant = len(comps[0]) if comps else 0
    density_undirected = nx.density(nx_graph) if n > 1 else math.nan
    directed_density = (
        exact["totals"]["relationships"] / (exact["totals"]["nodes"] * (exact["totals"]["nodes"] - 1))
        if exact["totals"]["nodes"] > 1
        else math.nan
    )
    row = {
        "indicateur": "Graphe complet Neo4j - noeuds",
        "valeur": exact["totals"]["nodes"],
        "note": "Toutes relations et tous labels",
    }
    summary = [
        row,
        {"indicateur": "Graphe complet Neo4j - relations", "valeur": exact["totals"]["relationships"], "note": "Toutes relations"},
        {"indicateur": "Nombre de labels", "valeur": len(exact["labels"]), "note": "CALL db.labels"},
        {"indicateur": "Nombre de types de relations", "valeur": len(exact["relations"]), "note": "MATCH ()-[r]->()"},
        {"indicateur": "Densite dirigee complete", "valeur": directed_density, "note": "relations / n(n-1)"},
        {"indicateur": "Projection matching - noeuds", "valeur": n, "note": "Relations utiles au matching"},
        {"indicateur": "Projection matching - aretes", "valeur": m, "note": "Graphe non dirige"},
        {"indicateur": "Projection matching - densite", "valeur": density_undirected, "note": "2m / n(n-1)"},
        {"indicateur": "Composantes connexes", "valeur": len(comps), "note": "Projection matching"},
        {"indicateur": "Taille composante geante", "valeur": giant, "note": "Projection matching"},
        {"indicateur": "Part composante geante", "valeur": giant / n if n else math.nan, "note": "Projection matching"},
    ]
    meta = {
        "projection_nodes": n,
        "projection_edges": m,
        "n_components": len(comps),
        "giant_size": giant,
        "giant_share": giant / n if n else None,
        "directed_density_complete": directed_density,
        "density_projection": density_undirected,
    }
    return pd.DataFrame(summary), meta


def degree_distribution(nx_graph, nodes: pd.DataFrame, plt, fmt_int) -> tuple[pd.DataFrame, pd.DataFrame]:
    deg = dict(nx_graph.degree())
    rows = []
    for node_id, d in deg.items():
        data = nx_graph.nodes[node_id]
        rows.append({"node_id": node_id, "label": data.get("label"), "nom": data.get("nom"), "degree": d})
    degrees = pd.DataFrame(rows)
    by_label = (
        degrees.groupby("label")["degree"]
        .agg(n_nodes="count", avg_degree="mean", median_degree="median", min_degree="min", max_degree="max", variance_degree="var")
        .reset_index()
        .sort_values("n_nodes", ascending=False)
    )
    save_table(degrees.sort_values("degree", ascending=False), "14_neo4j_projection_degrees_detail")
    save_table(by_label, "14_neo4j_projection_degree_stats")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    d = degrees["degree"].to_numpy()
    bins = np.unique(np.logspace(0, np.log10(max(d.max(), 2)), 28).astype(int))
    axes[0].hist(d, bins=bins, color=COLORS["blue"], alpha=0.86)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Degré k (échelle log)")
    axes[0].set_ylabel("Nombre de noeuds (échelle log)")
    axes[0].set_title("Distribution du degré")
    plot = by_label.sort_values("max_degree").tail(12)
    axes[1].barh(plot["label"], plot["max_degree"], color=COLORS["orange"])
    axes[1].xaxis.set_major_formatter(fmt_int)
    axes[1].set_xlabel("Degré maximal")
    axes[1].set_title("Asymétrie par type de noeud")
    save_fig(fig, "14_neo4j_distribution_degres")
    plt.close(fig)
    return degrees, by_label


def estimate_power_law(degrees: pd.DataFrame) -> pd.DataFrame:
    positive = degrees.loc[degrees["degree"] > 0, "degree"].astype(float)
    rows = []
    for xmin in [1, 2, 5, 10]:
        tail = positive[positive >= xmin]
        if len(tail) < 50:
            continue
        alpha = 1 + len(tail) / np.log(tail / (xmin - 0.5)).sum()
        rows.append({"xmin": xmin, "n_tail": int(len(tail)), "gamma_estime": float(alpha)})
    return pd.DataFrame(rows)


def centralities(nx_graph, degrees: pd.DataFrame, plt, fmt_int) -> dict[str, pd.DataFrame]:
    import networkx as nx

    n = nx_graph.number_of_nodes()
    degree_cent = {node: deg / (n - 1) if n > 1 else 0.0 for node, deg in nx_graph.degree()}
    pagerank = nx.pagerank(nx_graph, alpha=0.85, max_iter=120, tol=1e-06)

    # Betweenness is approximated on a high-degree induced subgraph. Exact
    # betweenness on the full graph would be computationally disproportionate.
    top_nodes = degrees.sort_values("degree", ascending=False).head(min(3500, len(degrees)))["node_id"].tolist()
    sub = nx_graph.subgraph(top_nodes).copy()
    k = min(250, sub.number_of_nodes())
    bet = nx.betweenness_centrality(sub, k=k, seed=42, normalized=True) if sub.number_of_nodes() > 2 else {}

    cent_rows = []
    for node_id in nx_graph.nodes:
        data = nx_graph.nodes[node_id]
        cent_rows.append(
            {
                "node_id": node_id,
                "label": data.get("label"),
                "nom": data.get("nom"),
                "degree": nx_graph.degree(node_id),
                "degree_centrality": degree_cent.get(node_id, 0.0),
                "pagerank": pagerank.get(node_id, 0.0),
                "betweenness_approx_top_degree": bet.get(node_id, np.nan),
            }
        )
    central = pd.DataFrame(cent_rows)
    save_table(central.sort_values("pagerank", ascending=False), "15_neo4j_centralites_noeuds")

    skill = central[central["label"].eq("Compétence")].copy()
    top_degree = skill.sort_values("degree_centrality", ascending=False).head(15)
    top_pagerank = skill.sort_values("pagerank", ascending=False).head(15)
    top_between = skill.dropna(subset=["betweenness_approx_top_degree"]).sort_values("betweenness_approx_top_degree", ascending=False).head(15)
    save_table(top_degree, "15_neo4j_top_competences_degree")
    save_table(top_pagerank, "15_neo4j_top_competences_pagerank")
    save_table(top_between, "15_neo4j_top_competences_betweenness")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6))
    for ax, df, metric, title, color in [
        (axes[0], top_degree.sort_values("degree_centrality"), "degree", "Compétences les plus connectées", COLORS["blue"]),
        (axes[1], top_pagerank.sort_values("pagerank"), "pagerank", "Compétences influentes PageRank", COLORS["green"]),
        (axes[2], top_between.sort_values("betweenness_approx_top_degree"), "betweenness_approx_top_degree", "Compétences passerelles", COLORS["purple"]),
    ]:
        if df.empty:
            ax.axis("off")
            continue
        labels = df["nom"].astype(str).str.slice(0, 31)
        ax.barh(labels, df[metric], color=color)
        ax.set_title(title)
        if metric == "degree":
            ax.xaxis.set_major_formatter(fmt_int)
    save_fig(fig, "15_neo4j_centralites_competences")
    plt.close(fig)
    return {"central": central, "top_degree": top_degree, "top_pagerank": top_pagerank, "top_betweenness": top_between}


def communities_and_paths(nx_graph, degrees: pd.DataFrame, plt, fmt_int) -> dict[str, Any]:
    import networkx as nx

    giant_nodes = max(nx.connected_components(nx_graph), key=len) if nx_graph.number_of_nodes() else set()
    giant = nx_graph.subgraph(giant_nodes).copy()
    rng = random.Random(42)

    sample_nodes = list(giant.nodes)
    if len(sample_nodes) > 4500:
        weighted = degrees.set_index("node_id").reindex(sample_nodes)["degree"].fillna(1).to_numpy()
        probs = weighted / weighted.sum()
        sampled = set(np.random.default_rng(42).choice(sample_nodes, size=4500, replace=False, p=probs))
        # Keep high-degree articulation-like nodes to preserve the skeleton.
        sampled.update(degrees.sort_values("degree", ascending=False).head(600)["node_id"].tolist())
        sampled = sampled.intersection(giant_nodes)
    else:
        sampled = set(sample_nodes)
    sub = giant.subgraph(sampled).copy()

    try:
        comms = nx.algorithms.community.louvain_communities(sub, seed=42, resolution=1.0)
        method = "louvain_networkx"
    except Exception:
        comms = nx.algorithms.community.greedy_modularity_communities(sub)
        method = "greedy_modularity_networkx"
    membership = []
    for cid, members in enumerate(sorted(comms, key=len, reverse=True)):
        labels = Counter(nx_graph.nodes[n].get("label") for n in members)
        skills = [
            nx_graph.nodes[n].get("nom")
            for n in members
            if nx_graph.nodes[n].get("label") == "Compétence"
        ][:8]
        membership.append(
            {
                "community_id": cid,
                "n_nodes": len(members),
                "dominant_label": labels.most_common(1)[0][0] if labels else "",
                "labels_resume": "; ".join(f"{k}:{v}" for k, v in labels.most_common(5)),
                "competences_exemples": "; ".join(str(s) for s in skills if s),
            }
        )
    comm_df = pd.DataFrame(membership)
    save_table(comm_df, "16_neo4j_communautes_louvain")

    path_lengths = []
    source_nodes = rng.sample(list(sampled), min(120, len(sampled))) if sampled else []
    for node in source_nodes:
        lengths = nx.single_source_shortest_path_length(sub, node, cutoff=8)
        for target, length in lengths.items():
            if target != node:
                path_lengths.append(length)
    path_df = pd.DataFrame({"distance": path_lengths})
    if not path_df.empty:
        save_table(path_df.describe(percentiles=[0.5, 0.75, 0.9]).reset_index(), "16_neo4j_distances_resume")

    comp_sizes = sorted([len(c) for c in nx.connected_components(nx_graph)], reverse=True)
    comp_df = pd.DataFrame({"rang": range(1, len(comp_sizes) + 1), "taille": comp_sizes})
    save_table(comp_df, "16_neo4j_composantes_connexes")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    cplot = comm_df.head(12).sort_values("n_nodes")
    axes[0].barh(cplot["community_id"].astype(str), cplot["n_nodes"], color=COLORS["green"])
    axes[0].xaxis.set_major_formatter(fmt_int)
    axes[0].set_title("Principales communautés")
    axes[0].set_xlabel("Noeuds")
    axes[0].set_ylabel("Communauté")
    if not path_df.empty:
        axes[1].hist(path_df["distance"], bins=range(1, int(path_df["distance"].max()) + 2), color=COLORS["purple"], alpha=0.86)
        axes[1].set_title("Distances dans la composante géante")
        axes[1].set_xlabel("Nombre de sauts")
    axes[2].plot(comp_df.head(40)["rang"], comp_df.head(40)["taille"], color=COLORS["red"], marker="o", linewidth=1.7)
    axes[2].set_yscale("log")
    axes[2].set_title("Taille des composantes")
    axes[2].set_xlabel("Rang")
    axes[2].set_ylabel("Taille log")
    save_fig(fig, "16_neo4j_communautes_connectivite")
    plt.close(fig)

    return {
        "community_method": method,
        "n_communities_sample": len(comm_df),
        "distance_mean_sample": float(np.mean(path_lengths)) if path_lengths else None,
        "distance_median_sample": float(np.median(path_lengths)) if path_lengths else None,
    }


def domain_specific(session, plt, fmt_int) -> dict[str, Any]:
    tables: dict[str, pd.DataFrame] = {}
    queries = {
        "17_neo4j_competences_par_offre": """
            MATCH (o:OffreEmploi)
            OPTIONAL MATCH (o)-[:REQUIERT]->(c:Compétence)
            WITH o, count(DISTINCT c) AS n_competences
            RETURN n_competences
        """,
        "17_neo4j_competences_par_candidat": """
            MATCH (p:Candidat)
            OPTIONAL MATCH (p)-[:POSSEDE]->(c:Compétence)
            WITH p, count(DISTINCT c) AS n_competences
            RETURN n_competences
        """,
        "17_neo4j_competences_par_metier": """
            MATCH (m:Métier)
            OPTIONAL MATCH (m)-[:NECESSITE]->(c:Compétence)
            WITH m, count(DISTINCT c) AS n_competences
            RETURN n_competences
        """,
    }
    summaries = []
    for name, query in queries.items():
        df = pd.DataFrame([dict(r) for r in session.run(query)])
        tables[name] = df
        save_table(df.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).reset_index(), f"{name}_resume")
        label = name.replace("17_neo4j_competences_par_", "")
        summaries.append(
            {
                "objet": label,
                "n": int(len(df)),
                "moyenne": float(df["n_competences"].mean()) if len(df) else None,
                "mediane": float(df["n_competences"].median()) if len(df) else None,
                "p90": float(df["n_competences"].quantile(0.9)) if len(df) else None,
                "max": int(df["n_competences"].max()) if len(df) else None,
            }
        )
    summary_df = pd.DataFrame(summaries)
    save_table(summary_df, "17_neo4j_competences_moyennes")

    skill_gap = pd.DataFrame(
        [dict(r) for r in session.run(
            """
            MATCH (cand:Candidat)
            WITH cand
            ORDER BY rand()
            LIMIT 350
            MATCH (offre:OffreEmploi)
            WITH cand, offre
            ORDER BY rand()
            LIMIT 7000
            MATCH (offre)-[:REQUIERT]->(req:Compétence)
            WITH cand, offre, collect(DISTINCT req) AS required
            OPTIONAL MATCH (cand)-[:POSSEDE]->(got:Compétence)
            WITH cand, offre, required, collect(DISTINCT got) AS possessed
            WITH cand, offre,
                 size(required) AS n_required,
                 size([s IN required WHERE s IN possessed]) AS n_acquired
            WHERE n_required > 0
            RETURN n_required,
                   n_acquired,
                   n_required - n_acquired AS skill_gap,
                   toFloat(n_acquired) / n_required AS taux_match
            """
        )]
    )
    if not skill_gap.empty:
        save_table(skill_gap, "17_neo4j_skill_gap_sample")
        save_table(skill_gap.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).reset_index(), "17_neo4j_skill_gap_resume")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    for ax, key, title, color in [
        (axes[0], "17_neo4j_competences_par_offre", "Compétences exigées par offre", COLORS["orange"]),
        (axes[1], "17_neo4j_competences_par_candidat", "Compétences possédées par candidat", COLORS["blue"]),
        (axes[2], "17_neo4j_competences_par_metier", "Compétences par métier", COLORS["green"]),
    ]:
        df = tables[key]
        ax.hist(df["n_competences"], bins=25, color=color, alpha=0.86)
        ax.set_title(title)
        ax.set_xlabel("Nombre de compétences")
        ax.yaxis.set_major_formatter(fmt_int)
    save_fig(fig, "17_neo4j_competences_metier_offre_candidat")
    plt.close(fig)

    if not skill_gap.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
        axes[0].hist(skill_gap["skill_gap"], bins=range(0, int(skill_gap["skill_gap"].max()) + 2), color=COLORS["red"], alpha=0.86)
        axes[0].set_title("Distribution du skill gap")
        axes[0].set_xlabel("Compétences requises non possédées")
        axes[1].hist(skill_gap["taux_match"], bins=np.linspace(0, 1, 22), color=COLORS["teal"], alpha=0.86)
        axes[1].set_title("Taux de couverture des compétences")
        axes[1].set_xlabel("n acquises / n requises")
        save_fig(fig, "17_neo4j_skill_gap_distribution")
        plt.close(fig)

    return {
        "skill_gap_mean": float(skill_gap["skill_gap"].mean()) if not skill_gap.empty else None,
        "skill_gap_median": float(skill_gap["skill_gap"].median()) if not skill_gap.empty else None,
        "skill_gap_under_3_share": float((skill_gap["skill_gap"] <= 3).mean()) if not skill_gap.empty else None,
    }


def robustness(nx_graph, degrees: pd.DataFrame, plt, fmt_int) -> pd.DataFrame:
    import networkx as nx

    base_giant = max((len(c) for c in nx.connected_components(nx_graph)), default=0)
    rows = [{"scenario": "base", "removed_share": 0.0, "removed_nodes": 0, "giant_size": base_giant, "giant_share_of_base": 1.0}]
    sorted_nodes = degrees.sort_values("degree", ascending=False)["node_id"].tolist()
    n = nx_graph.number_of_nodes()
    for share in [0.01, 0.05, 0.10]:
        k = max(1, int(round(n * share)))
        g2 = nx_graph.copy()
        g2.remove_nodes_from(sorted_nodes[:k])
        giant = max((len(c) for c in nx.connected_components(g2)), default=0)
        rows.append(
            {
                "scenario": f"retrait_top_{int(share * 100)}pct_degre",
                "removed_share": share,
                "removed_nodes": k,
                "giant_size": giant,
                "giant_share_of_base": giant / base_giant if base_giant else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    save_table(df, "18_neo4j_robustesse")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(df["removed_share"] * 100, df["giant_share_of_base"] * 100, marker="o", color=COLORS["red"], linewidth=2.2)
    ax.set_xlabel("Noeuds les plus centraux retirés (%)")
    ax.set_ylabel("Composante géante restante (%)")
    ax.set_title("Robustesse de la projection matching")
    ax.set_ylim(0, 105)
    save_fig(fig, "18_neo4j_robustesse")
    plt.close(fig)
    return df


def main() -> None:
    ensure_dirs()
    plt, fmt_int = setup_plot()
    sys.path.insert(0, str(ROOT / "src" / "03_knowledge_graph"))

    try:
        from neo4j import GraphDatabase
        from config_neo4j import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DATABASE) as session:
            exact = query_exact_neo4j(session)
            nodes, edges = query_edges(session)
            domain = domain_specific(session, plt, fmt_int)
        driver.close()
    except Exception as exc:
        save_json({"available": False, "error": str(exc)}, "14_neo4j_network_error")
        raise

    for name, df in [
        ("08_neo4j_node_counts", exact["labels"]),
        ("08_neo4j_relation_counts", exact["relations"]),
        ("08_neo4j_patterns_top", exact["patterns"]),
        ("09_neo4j_degree_stats", exact["degree"]),
        ("14_neo4j_global_degree_stats", exact["global_degree"]),
        ("14_neo4j_projection_nodes", nodes),
        ("14_neo4j_projection_edges", edges),
    ]:
        save_table(df, name)

    g = build_networkx(nodes, edges)
    summary_df, meta = graph_descriptive(g, exact)
    save_table(summary_df, "14_neo4j_statistiques_globales")

    degrees, degree_by_label = degree_distribution(g, nodes, plt, fmt_int)
    power = estimate_power_law(degrees)
    save_table(power, "14_neo4j_power_law_estimation")
    cent = centralities(g, degrees, plt, fmt_int)
    comm = communities_and_paths(g, degrees, plt, fmt_int)
    rob = robustness(g, degrees, plt, fmt_int)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    label_plot = exact["labels"].head(12).sort_values("n")
    axes[0, 0].barh(label_plot["label"], label_plot["n"], color=COLORS["orange"])
    axes[0, 0].xaxis.set_major_formatter(fmt_int)
    axes[0, 0].set_title("Noeuds par label")
    rel_plot = exact["relations"].head(12).sort_values("n")
    axes[0, 1].barh(rel_plot["relation"], rel_plot["n"], color=COLORS["teal"])
    axes[0, 1].xaxis.set_major_formatter(fmt_int)
    axes[0, 1].set_title("Relations par type")
    dplot = degree_by_label.sort_values("max_degree").tail(12)
    axes[1, 0].barh(dplot["label"], dplot["max_degree"], color=COLORS["purple"])
    axes[1, 0].xaxis.set_major_formatter(fmt_int)
    axes[1, 0].set_title("Degrés maximaux")
    axes[1, 1].plot(rob["removed_share"] * 100, rob["giant_share_of_base"] * 100, marker="o", color=COLORS["red"])
    axes[1, 1].set_title("Robustesse")
    axes[1, 1].set_xlabel("Retrait des hubs (%)")
    axes[1, 1].set_ylabel("Composante géante restante (%)")
    save_fig(fig, "14_neo4j_dashboard_reseau")
    plt.close(fig)

    summary = {
        "available": True,
        "exact": exact["totals"],
        "projection": meta,
        "power_law": power.to_dict("records"),
        "centrality_rows": len(cent["central"]),
        "communities": comm,
        "domain": domain,
    }
    save_json(summary, "14_neo4j_network_summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
