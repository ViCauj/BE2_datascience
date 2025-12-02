from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from sentence_transformers import SentenceTransformer


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


class BagOfWordsSearchEngine(BaseSearchEngine):
    def __init__(self, corpus: Dict[str, Dict], queries: Dict[str, Dict]):
        super().__init__(corpus, queries)
        self.vectorizer = CountVectorizer(stop_words="english", max_features=50000)


class TfidfSearchEngine(BaseSearchEngine):
    def __init__(self, corpus: Dict[str, Dict], queries: Dict[str, Dict]):
        super().__init__(corpus, queries)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)


class DenseSearchEngine(BaseSearchEngine):
    def __init__(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, Dict],
        model_name: str = "all-MiniLM-L6-v2",
        dump_path: str = "outputs/dense_embeddings.npy",
    ):
        super().__init__(corpus, queries)
        self.model_name = model_name
        self.dump_path = dump_path

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

    def search(self, query_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
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

        similarities = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = similarities[idx]
            results.append((doc_id, score))

        return results

