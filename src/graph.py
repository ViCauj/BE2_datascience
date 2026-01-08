import networkx as nx
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt


def build_citation_graph(corpus: Dict[str, Dict]) -> nx.DiGraph:
    """
    Construit le graphe de citations à partir du corpus.
    Gère les références sortantes (references) ET entrantes (cited_by).
    """
    print("Construction du graphe de citations...")
    G = nx.DiGraph()
    G.add_nodes_from(corpus.keys())

    for doc_id, doc in corpus.items():
        meta = doc.get("metadata", {})

        refs_out = doc.get("references", [])

        for ref_id in refs_out:
            if ref_id in corpus:
                G.add_edge(doc_id, ref_id)

        cited_by_list = meta.get("cited_by", [])

        for citing_id in cited_by_list:
            if citing_id in corpus:
                if not G.has_edge(citing_id, doc_id):
                    G.add_edge(citing_id, doc_id)

    print(
        f"Graphe construit : {G.number_of_nodes()} noeuds, {G.number_of_edges()} arcs."
    )
    return G


def get_graph_statistics(G: nx.DiGraph) -> Dict[str, Any]:
    stats = {}
    stats["num_nodes"] = G.number_of_nodes()
    stats["num_edges"] = G.number_of_edges()
    stats["density"] = nx.density(G)

    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]

    stats["avg_in_degree"] = np.mean(in_degrees) if in_degrees else 0
    stats["avg_out_degree"] = np.mean(out_degrees) if out_degrees else 0
    stats["max_in_degree"] = np.max(in_degrees) if in_degrees else 0

    return stats


def get_top_centrality(G: nx.DiGraph, top_k: int = 5):
    print(f"\nCalcul du PageRank (Centralité)...")
    if G.number_of_edges() == 0:
        print("Attention : Graphe vide, impossible de calculer la centralité.")
        return []

    try:
        pagerank = nx.pagerank(G, alpha=0.85)
        sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]
        print(f"--- Top {top_k} Articles influents (PageRank) ---")
        return sorted_pr
    except Exception as e:
        print(f"Erreur calcul centralité : {e}")
        return []


def visualize_backbone(G: nx.DiGraph, top_k: int = 300):
    """
    Affiche le 'squelette' du graphe : les top_k articles les plus influents (PageRank)
    et les liens qui les relient entre eux.
    """
    print(f"\nExtraction du squelette (Top {top_k} PageRank)...")

    if G.number_of_nodes() == 0:
        print("Graphe vide.")
        return

    try:
        pagerank = nx.pagerank(G, alpha=0.85)
    except:
        pagerank = dict(G.degree())

    top_nodes = [
        n
        for n, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
    ]

    backbone = G.subgraph(top_nodes)
    print(
        f"Squelette extrait : {backbone.number_of_nodes()} noeuds, {backbone.number_of_edges()} arcs."
    )

    # 4. Dessin
    plt.figure(figsize=(14, 12))

    # Layout : Kamada-Kawai est souvent meilleur que spring pour les structures déconnectées/clusters
    print("Calcul du layout (Kamada-Kawai)...")
    try:
        pos = nx.kamada_kawai_layout(backbone)
    except:
        pos = nx.spring_layout(backbone, k=0.3, iterations=50)

    node_sizes = [v * 10 for v in dict(backbone.degree()).values()]
    node_colors = [dict(backbone.degree())[n] for n in backbone.nodes()]

    nx.draw_networkx_nodes(
        backbone,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.coolwarm,
        alpha=0.8,
    )
    nx.draw_networkx_edges(backbone, pos, alpha=0.2, edge_color="gray", arrowsize=10)

    plt.title(f"Squelette du graphe de citations (Top {top_k} articles influents)")
    plt.axis("off")
    print("Affichage...")
    plt.show()
