import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities, modularity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    COMBINED_CLEAN,
    COMBINED_EDGES,
    COMBINED_WITH_NETWORK,
    FIGURES_DIR,
    GRAPHS_DIR,
    TABLES_DIR,
    ensure_dirs,
)


def read_is_top_level(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def build_graph(edges):
    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=float(row.weight))
    return graph


def weighted_degree(graph):
    degree = {}
    for node in graph.nodes:
        in_weight = graph.in_degree(node, weight="weight")
        out_weight = graph.out_degree(node, weight="weight")
        degree[node] = in_weight + out_weight
    return degree


def detect_communities(undirected):
    if undirected.number_of_nodes() == 0:
        return {}, [], 0.0
    if undirected.number_of_edges() == 0:
        communities = [frozenset([node]) for node in undirected.nodes]
    else:
        communities = list(greedy_modularity_communities(undirected, weight="weight"))
    partition = {}
    for idx, community in enumerate(communities):
        for node in community:
            partition[node] = idx
    score = modularity(undirected, communities, weight="weight") if len(communities) > 1 else 0.0
    return partition, communities, score


def graph_summary(game, graph, undirected, communities, modularity_score):
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    components = list(nx.connected_components(undirected)) if nodes else []
    largest = max((len(c) for c in components), default=0)
    avg_degree = (sum(dict(undirected.degree()).values()) / nodes) if nodes else 0
    clustering = nx.average_clustering(undirected, weight="weight") if nodes else 0
    reciprocity = nx.reciprocity(graph) if edges else 0
    if reciprocity is None or math.isnan(reciprocity):
        reciprocity = 0
    return {
        "game": game,
        "nodes": nodes,
        "edges": edges,
        "density": round(nx.density(graph), 6) if nodes > 1 else 0,
        "components": len(components),
        "largest_component_size": largest,
        "average_degree": round(avg_degree, 4),
        "clustering": round(clustering, 6),
        "reciprocity": round(reciprocity, 6),
        "communities": len(communities),
        "modularity": round(modularity_score, 6),
    }


def centrality_table(game, graph, undirected, partition):
    nodes = list(graph.nodes)
    if not nodes:
        return pd.DataFrame()

    weighted = weighted_degree(graph)
    in_degree = dict(graph.in_degree(weight="weight"))
    out_degree = dict(graph.out_degree(weight="weight"))
    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {n: 0 for n in nodes}

    if undirected.number_of_nodes() > 2 and undirected.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(undirected, weight="weight", k=min(200, len(nodes)), seed=42)
    else:
        betweenness = {n: 0 for n in nodes}

    rows = []
    for node in nodes:
        rows.append(
            {
                "game": game,
                "authorChannelId": node,
                "community_id": partition.get(node),
                "weighted_degree": weighted.get(node, 0),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "pagerank": pagerank.get(node, 0),
                "betweenness": betweenness.get(node, 0),
            }
        )
    return pd.DataFrame(rows).sort_values(["game", "pagerank"], ascending=[True, False])


def save_network_plot(game, undirected, centrality, partition):
    if undirected.number_of_nodes() == 0:
        return
    graph = undirected.copy()
    if not nx.is_connected(graph):
        largest = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest).copy()
    if graph.number_of_nodes() > 250:
        top_nodes = (
            centrality[centrality["authorChannelId"].isin(graph.nodes)]
            .nlargest(250, "weighted_degree")["authorChannelId"]
            .tolist()
        )
        graph = graph.subgraph(top_nodes).copy()

    pos = nx.spring_layout(graph, seed=42, weight="weight", k=0.25)
    node_sizes = [40 + 8 * math.sqrt(graph.degree(n, weight="weight")) for n in graph.nodes]
    node_colors = [partition.get(n, -1) for n in graph.nodes]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(graph, pos, alpha=0.15, width=0.5)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="tab20",
        linewidths=0,
        alpha=0.9,
    )
    plt.title(f"{game} Reply Network")
    plt.axis("off")
    plt.tight_layout()
    filename = game.lower().replace(" ", "_").replace("!", "")
    plt.savefig(FIGURES_DIR / f"{filename}_reply_network.png", dpi=200)
    plt.close()


def main():
    ensure_dirs()
    comments = pd.read_csv(COMBINED_CLEAN)
    edges = pd.read_csv(COMBINED_EDGES)

    summaries = []
    centralities = []
    community_rows = []
    comments["community_id"] = pd.NA

    for game, game_edges in edges.groupby("gameLabel"):
        graph = build_graph(game_edges)
        undirected = graph.to_undirected()
        partition, communities, modularity_score = detect_communities(undirected)

        summaries.append(graph_summary(game, graph, undirected, communities, modularity_score))
        centrality = centrality_table(game, graph, undirected, partition)
        centralities.append(centrality)

        for idx, community in enumerate(communities):
            community_rows.append(
                {
                    "game": game,
                    "community_id": idx,
                    "users": len(community),
                    "weighted_degree_sum": sum(dict(undirected.degree(community, weight="weight")).values()),
                }
            )

        mask = comments["gameLabel"].eq(game)
        comments.loc[mask, "community_id"] = comments.loc[mask, "authorChannelId"].map(partition)
        nx.write_gexf(graph, GRAPHS_DIR / f"{game.lower().replace(' ', '_')}_reply_network.gexf")
        save_network_plot(game, undirected, centrality, partition)

    summary_df = pd.DataFrame(summaries)
    centrality_df = pd.concat(centralities, ignore_index=True) if centralities else pd.DataFrame()
    community_df = pd.DataFrame(community_rows)

    if not centrality_df.empty:
        centrality_df["centrality_group"] = "Peripheral"
        for game, group in centrality_df.groupby("game"):
            threshold = group["weighted_degree"].quantile(0.90)
            centrality_df.loc[
                centrality_df["game"].eq(game) & centrality_df["weighted_degree"].ge(threshold),
                "centrality_group",
            ] = "High centrality"

        comments = comments.merge(
            centrality_df[
                [
                    "game",
                    "authorChannelId",
                    "weighted_degree",
                    "in_degree",
                    "out_degree",
                    "pagerank",
                    "betweenness",
                    "centrality_group",
                ]
            ],
            left_on=["gameLabel", "authorChannelId"],
            right_on=["game", "authorChannelId"],
            how="left",
        ).drop(columns=["game"])

    summary_df.to_csv(TABLES_DIR / "network_summary.csv", index=False)
    centrality_df.to_csv(TABLES_DIR / "centrality_by_user.csv", index=False)
    community_df.to_csv(TABLES_DIR / "network_community_sizes.csv", index=False)
    comments.to_csv(COMBINED_WITH_NETWORK, index=False, encoding="utf-8-sig")

    if not community_df.empty:
        top_communities = community_df.sort_values(["game", "users"], ascending=[True, False])
        top_communities.to_csv(TABLES_DIR / "top_network_communities.csv", index=False)
        plot_df = top_communities.groupby("game").head(10)
        ax = plot_df.assign(label=lambda x: x["game"] + " C" + x["community_id"].astype(str)).plot(
            x="label", y="users", kind="bar", legend=False, figsize=(10, 5), color="#4263eb"
        )
        ax.set_ylabel("Users")
        ax.set_xlabel("")
        ax.set_title("Top Reply Network Communities by User Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "community_size_distribution.png", dpi=200)
        plt.close()

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
