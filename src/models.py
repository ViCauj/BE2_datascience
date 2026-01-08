from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from sentence_transformers import SentenceTransformer
from .graph import CitationGraph
import pandas as pd


class BaseSearchEngine:
    """
    Classe mère regroupant la logique commune aux moteurs de recherche
    """

    def __init__(
        self, corpus: Dict[str, Dict], queries: Dict[str, Dict], **vectorizer_params
    ):
        self.corpus = corpus
        self.queries = queries
        self.doc_ids = list(corpus.keys())
        self.id_to_index = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        self.vectorizer_params = vectorizer_params
        self.vectorizer = None
        self.matrix = None

    def fit(self):
        """
        Construit la matrice Documents x Termes en utilisant le vectorizer défini.
        """
        if self.vectorizer is None:
            raise NotImplementedError(
                "Le vectorizer doit être défini dans la classe fille."
            )

        print(f"Entraînement du modèle {self.__class__.__name__}...")

        corpus_texts = []
        for doc_id in self.doc_ids:
            doc = self.corpus[doc_id]
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            corpus_texts.append(text)

        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def search(
        self, query_id: str, candidate_ids: Optional[List[str]] = None, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Recherche les documents les plus similaires à une requête donnée par son ID.
        """
        if self.matrix is None:
            raise ValueError("Le modèle n'est pas entraîné. Appelez .fit() d'abord.")

        if query_id not in self.queries:
            print(f"Requête {query_id} introuvable.")
            return []

        query_data = self.queries[query_id]
        query_text = query_data.get("text", "")
        query_vec = self.vectorizer.transform([query_text])

        if candidate_ids:
            valid_candidates = [cid for cid in candidate_ids if cid in self.id_to_index]
            if not valid_candidates:
                return []

            indices = [self.id_to_index[cid] for cid in valid_candidates]
            candidate_matrix = self.matrix[indices]
            similarities = cosine_similarity(query_vec, candidate_matrix).flatten()
            sorted_indices_local = np.argsort(similarities)[::-1]  # décroissant

            results = []
            for idx in sorted_indices_local[:top_k]:
                doc_id = valid_candidates[idx]
                score = similarities[idx]
                results.append((doc_id, score))
            return results
        else:
            similarities = cosine_similarity(query_vec, self.matrix).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = [(self.doc_ids[idx], similarities[idx]) for idx in top_indices]
            return results

    def print_results(
        self,
        results: List[Tuple[str, float]],
        query_id: str,
        qrels: Optional[Dict] = None,
    ):
        """
        Affiche les résultats formatés avec vérification de la pertinence.
        """
        query_title = self.queries[query_id].get("title")
        if not query_title:
            query_title = self.queries[query_id].get("text", "")[:50] + "..."

        print(f"\n--- Résultats {self.__class__.__name__} pour : {query_title} ---")

        for doc_id, score in results:
            title = self.corpus[doc_id].get("title", "Sans titre")

            relevance_msg = ""
            if qrels and query_id in qrels:
                is_relevant = qrels[query_id].get(doc_id, 0)
                status = "✅" if is_relevant == 1 else "❌"
                relevance_msg = f" [{status}]"

            print(f"[{score:.4f}]{relevance_msg} {doc_id} - {title[:80]}...")

    def create_submission(self, file_path: str, output_path: str) -> None:
        """
        Génère un fichier de soumission Kaggle (CSV) à partir du fichier de test (TSV).
        """

        print(f"Chargement du fichier de test : {file_path}")
        try:
            test_df = pd.read_csv(file_path, sep="\t")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier : {e}")
            return

        # Vérification rapide des colonnes
        if "query-id" not in test_df.columns or "corpus-id" not in test_df.columns:
            print(
                "Erreur : Le fichier test doit contenir les colonnes 'query-id' et 'corpus-id'."
            )
            return

        scores_map = {}
        unique_queries = test_df["query-id"].unique()

        for qid in unique_queries:
            subset = test_df[test_df["query-id"] == qid]
            candidate_ids = subset["corpus-id"].tolist()

            results = self.search(
                qid, candidate_ids=candidate_ids, top_k=len(candidate_ids)
            )

            for doc_id, score in results:
                scores_map[(qid, doc_id)] = score

        test_df["score"] = test_df.apply(
            lambda row: scores_map.get((row["query-id"], row["corpus-id"]), 0.0), axis=1
        )

        submission = pd.DataFrame(
            {
                "RowId": range(1, len(test_df) + 1),
                "query-id": test_df["query-id"],
                "corpus-id": test_df["corpus-id"],
                "score": test_df["score"],
            }
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission.to_csv(output_path, index=False)
        print(f"Soumission générée avec succès : {output_path}")


class BagOfWordsCountSearchEngine(BaseSearchEngine):
    def __init__(self, corpus: Dict, queries: Dict, **kwargs):
        super().__init__(corpus, queries, **kwargs)
        default = {"stop_words": "english", "max_features": 50000}
        self.vectorizer = CountVectorizer(**{**default, **kwargs})


class BagOfWordsTfidfSearchEngine(BaseSearchEngine):
    def __init__(self, corpus: Dict, queries: Dict, **kwargs):
        super().__init__(corpus, queries, **kwargs)
        default = {"stop_words": "english", "max_features": 50000}
        self.vectorizer = TfidfVectorizer(**{**default, **kwargs})


class DenseSearchEngine(BaseSearchEngine):
    def __init__(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, Dict],
        model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__(corpus, queries)
        self.model_name = model_name
        self.dump_path = "outputs/dense_embeddings_" + model_name + ".npy"

        print(f"Chargement du modèle SentenceTransformer : {model_name}...")
        self.model = SentenceTransformer(model_name)

    def fit(self):
        """
        Calcule ou charge les embeddings du corpus.
        """
        if os.path.exists(self.dump_path):
            print(f"Chargement des embeddings existants depuis {self.dump_path}...")
            self.matrix = np.load(self.dump_path)
            return

        print("Calcul des embeddings...")
        corpus_texts = []
        for doc_id in self.doc_ids:
            doc = self.corpus[doc_id]
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            corpus_texts.append(text)

        self.matrix = self.model.encode(corpus_texts, show_progress_bar=True)
        os.makedirs(os.path.dirname(self.dump_path), exist_ok=True)
        np.save(self.dump_path, self.matrix)
        print(f"Embeddings sauvegardés dans {self.dump_path}.")

    def search(
        self, query_id: str, candidate_ids: Optional[List[str]] = None, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Surcharge de la méthode search car on utilise self.model.encode() et non vectorizer.transform().
        """
        if self.matrix is None:
            raise ValueError("Le modèle n'est pas entraîné.")

        if query_id not in self.queries:
            return []

        query_data = self.queries[query_id]
        query_text = query_data.get("text", "")
        query_vec = self.model.encode([query_text])

        if candidate_ids:
            valid_candidates = [cid for cid in candidate_ids if cid in self.id_to_index]
            if not valid_candidates:
                return []

            indices = [self.id_to_index[cid] for cid in valid_candidates]
            candidate_matrix = self.matrix[indices]
            similarities = cosine_similarity(query_vec, candidate_matrix).flatten()
            sorted_indices = np.argsort(similarities)[::-1]

            return [
                (valid_candidates[i], similarities[i]) for i in sorted_indices[:top_k]
            ]
        else:
            similarities = cosine_similarity(query_vec, self.matrix).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [(self.doc_ids[i], similarities[i]) for i in top_indices]


class HybridSearchEngine(DenseSearchEngine):
    """
    Moteur hybride combinant embeddings denses (contenu) et lissage par graphe (structure).
    Hérite de DenseSearchEngine pour réutiliser la logique BERT.
    """

    def __init__(
        self,
        corpus: Dict,
        queries: Dict,
        model_name: str = "all-MiniLM-L6-v2",
        alpha: float = 0.7,
    ):
        super().__init__(corpus, queries, model_name)
        self.alpha = alpha

    def fit(self):
        """
        1. Embeddings Denses (BERT).
        2. Construction du Graphe.
        3. Lissage par le Graphe.
        """

        super().fit()

        graph_analyzer = CitationGraph(self.corpus)
        graph_analyzer.build()

        self.matrix = graph_analyzer.compute_smoothed_embeddings(
            doc_ids=self.doc_ids, base_matrix=self.matrix, alpha=self.alpha
        )


class EnsembleSearchEngine(BaseSearchEngine):
    """
    Combine les scores du meilleur modèle et du TF-IDF.
    """

    def __init__(self, corpus: Dict, queries: Dict, alpha: float = 0.85):
        super().__init__(corpus, queries)
        self.alpha = alpha

        self.strong_model = GCNSearchEngine(
            corpus, queries, model_name="all-MiniLM-L6-v2", k_hops=3, alpha=0.65
        )
        self.weak_model = BagOfWordsTfidfSearchEngine(corpus, queries)

    def fit(self):
        self.strong_model.fit()
        self.weak_model.fit()

    def search(
        self, query_id: str, candidate_ids: Optional[List[str]] = None, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        candidates_to_score = candidate_ids

        # Si aucun candidat n'est imposé, on laisse le modèle fort proposer son top 50
        if candidates_to_score is None:
            strong_results = self.strong_model.search(query_id, top_k=50)
            candidates_to_score = [doc_id for doc_id, _ in strong_results]

        # On force top_k=len(candidates) pour récupérer tous les scores sans filtrage
        res_strong = dict(
            self.strong_model.search(
                query_id,
                candidate_ids=candidates_to_score,
                top_k=len(candidates_to_score),
            )
        )
        res_weak = dict(
            self.weak_model.search(
                query_id,
                candidate_ids=candidates_to_score,
                top_k=len(candidates_to_score),
            )
        )

        final_results = []
        for doc_id in candidates_to_score:
            s_strong = res_strong.get(doc_id, 0.0)
            s_weak = res_weak.get(doc_id, 0.0)

            # Moyenne pondérée
            final_score = self.alpha * s_strong + (1 - self.alpha) * s_weak
            final_results.append((doc_id, final_score))

        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:top_k]


class GCNSearchEngine(DenseSearchEngine):
    """
    Implémente une approche GCN simplifiée.
    Capture les relations à longue distance (voisins des voisins).
    """

    def __init__(
        self,
        corpus: Dict,
        queries: Dict,
        model_name: str = "all-MiniLM-L6-v2",
        k_hops: int = 3,
        alpha: float = 0.65,
    ):
        super().__init__(corpus, queries, model_name)
        self.k_hops = k_hops
        self.alpha = alpha

    def fit(self):
        super().fit()

        graph_analyzer = CitationGraph(self.corpus)
        graph_analyzer.build()

        current_matrix = self.matrix

        for i in range(self.k_hops):
            print(f"Couche GCN {i + 1}/{self.k_hops}...")
            current_matrix = graph_analyzer.compute_smoothed_embeddings(
                doc_ids=self.doc_ids, base_matrix=current_matrix, alpha=self.alpha
            )

        self.matrix = current_matrix
        print("Propagation GCN terminée.")
