import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


class CitationGraph:
    """
    Classe responsable de la construction, de l'analyse et de la visualisation
    du graphe de citations.
    """

    def __init__(self, corpus: Dict[str, Dict]):
        self.corpus = corpus
        self.G = nx.DiGraph()
        self.stats = {}

    def build(self):
        """
        Construit le graphe à partir du corpus.
        """

        print("Construction du graphe...")
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.corpus.keys())

        for doc_id, doc in self.corpus.items():
            meta = doc.get("metadata", {})

            # Liens sortants
            refs_out = doc.get("references", [])
            for ref_id in refs_out:
                if ref_id in self.corpus:
                    self.G.add_edge(doc_id, ref_id)

            # Liens entrants
            cited_by_list = meta.get("cited_by", [])
            for citing_id in cited_by_list:
                if citing_id in self.corpus:
                    if not self.G.has_edge(citing_id, doc_id):
                        self.G.add_edge(citing_id, doc_id)

        print(
            f"Graphe construit : {self.G.number_of_nodes()} nœuds, {self.G.number_of_edges()} arcs."
        )

    def compute_smoothed_embeddings(
        self, doc_ids: List[str], base_matrix: np.ndarray, alpha: float = 0.7
    ) -> np.ndarray:
        """
        Utilise la structure du graphe pour lisser les embeddings fournis.
        """

        id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        G_undir = self.G.to_undirected()

        new_matrix = np.zeros_like(base_matrix)
        isolated_docs = 0

        for i, doc_id in enumerate(doc_ids):
            vec_self = base_matrix[i]

            if doc_id not in G_undir or G_undir.degree(doc_id) == 0:
                new_matrix[i] = vec_self
                isolated_docs += 1
                continue

            neighbors = list(G_undir.neighbors(doc_id))
            valid_indices = [id_to_idx[n] for n in neighbors if n in id_to_idx]

            if not valid_indices:
                new_matrix[i] = vec_self
                isolated_docs += 1
                continue

            vec_neighbors = np.mean(base_matrix[valid_indices], axis=0)
            new_matrix[i] = alpha * vec_self + (1 - alpha) * vec_neighbors
        return new_matrix

    def analyze(self, top_k_centrality: int = 5):
        """
        Calcule et affiche les statistiques et la centralité.
        """

        if self.G.number_of_nodes() == 0:
            print("Graphe vide. Appelez .build() d'abord.")
            return

        # Calculs Stats
        self.stats["num_nodes"] = self.G.number_of_nodes()
        self.stats["num_edges"] = self.G.number_of_edges()
        self.stats["density"] = nx.density(self.G)

        in_degrees = [d for n, d in self.G.in_degree()]
        self.stats["avg_in_degree"] = np.mean(in_degrees) if in_degrees else 0
        self.stats["max_in_degree"] = np.max(in_degrees) if in_degrees else 0

        # Affichage Stats
        print("\n--- Statistiques du Graphe ---")
        print(f"Nœuds (Articles) : {self.stats['num_nodes']}")
        print(f"Arcs (Citations) : {self.stats['num_edges']}")
        print(f"Densité          : {self.stats['density']:.6f}")
        print(f"Citations reçues (moyenne) : {self.stats['avg_in_degree']:.2f}")
        print(f"Article le plus cité       : {self.stats['max_in_degree']} fois")

        # Centralité (si arcs)
        if self.stats["num_edges"] > 0:
            print(f"\nCalcul du PageRank (Top {top_k_centrality})...")
            try:
                pagerank = nx.pagerank(self.G, alpha=0.85)
                sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[
                    :top_k_centrality
                ]

                print(f"--- Top {top_k_centrality} Articles influents ---")
                for doc_id, score in sorted_pr:
                    title = self.corpus[doc_id].get("title", "Sans titre")
                    print(f"[{score:.4f}] {doc_id} - {title[:80]}...")
            except Exception as e:
                print(f"Erreur calcul centralité : {e}")
        else:
            print("Pas d'arcs, impossible de calculer la centralité.")

    def visualize(
        self,
        top_k: int = 500,
        output_path: str = "outputs/Squelette_graphe_top_500.png",
    ):
        """
        Génère une visualisation interprétable du squelette du graphe.
        Améliorations :
        - Clustering : Couleurs basées sur la détection de communautés (domaines).
        - Taille : Basée sur le PageRank (importance).
        - Layout : Force-directed plus espacé pour voir les grappes.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        from networkx.algorithms import community

        print(f"Génération de la visualisation du graphe (Top {top_k})...")

        # 1. Extraction du sous-graphe (les articles les plus connectés)
        # CORRECTION : self.G au lieu de self.graph
        degrees = dict(self.G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_k]

        # CORRECTION : self.G au lieu de self.graph
        subgraph = self.G.subgraph(top_nodes)

        # 2. Détection de communautés (pour colorer par "Domaine Scientifique")
        print("   - Détection des communautés...")
        try:
            communities_generator = community.greedy_modularity_communities(subgraph)
            # Création d'une map {noeud -> id_communauté}
            node_community = {}
            for i, comm in enumerate(communities_generator):
                for node in comm:
                    node_community[node] = i

            nb_comms = len(communities_generator)
            print(f"   - {nb_comms} communautés détectées.")
            node_colors = [node_community.get(n, 0) for n in subgraph.nodes()]
            cmap = plt.cm.tab20  # Palette de couleurs distinctes
        except Exception as e:
            print(
                f"   Warning: Détection de communautés échouée ({e}), utilisation couleur unique."
            )
            node_colors = "skyblue"
            cmap = None

        # 3. Calcul de l'importance (PageRank) pour la taille des points
        try:
            pr = nx.pagerank(subgraph)
            # On multiplie le PageRank par un facteur pour que les points soient visibles
            node_sizes = [pr[n] * 50000 for n in subgraph.nodes()]
        except:
            node_sizes = 50  # Taille par défaut si échec

        # 4. Calcul du Layout (Positionnement)
        print("   - Calcul du layout (Spring)...")
        pos = nx.spring_layout(subgraph, k=0.20, iterations=50, seed=42)

        # 5. Dessin
        plt.figure(figsize=(15, 15))

        # Dessin des arêtes
        nx.draw_networkx_edges(subgraph, pos, alpha=0.1, edge_color="gray")

        # Dessin des noeuds
        sc = nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            alpha=0.9,
        )

        plt.title(
            f"Squelette du Graphe de Citations (Top {top_k} articles)\nCouleur = Communauté (Domaine), Taille = PageRank (Influence)",
            fontsize=16,
        )
        plt.axis("off")

        # Sauvegarde
        try:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✅ Visualisation sauvegardée : {output_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'image : {e}")
