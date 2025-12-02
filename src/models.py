from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple, Optional


class BagOfWordsSearchEngine:
    def __init__(self, corpus: Dict[str, Dict], queries: Dict[str, Dict]):
        """
        Moteur de recherche basé sur le comptage simple des mots (Bag-of-Words).
        """
        self.corpus = corpus
        self.queries = queries
        self.doc_ids = list(corpus.keys())
        self.vectorizer = CountVectorizer(stop_words="english", max_features=50000)
        self.matrix = None

    def fit(self):
        """
        Construit la matrice Documents x Termes (comptage).
        """
        print("Entraînement du modèle Bag-of-Words (CountVectorizer)...")

        corpus_texts = []
        for doc_id in self.doc_ids:
            doc = self.corpus[doc_id]
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            corpus_texts.append(text)

        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def search(self, query_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Recherche à partir d'un ID de requête.
        """
        if self.matrix is None:
            raise ValueError("Le modèle n'est pas entraîné.")

        if query_id not in self.queries:
            print(f"Requête {query_id} introuvable.")
            return []

        query_data = self.queries[query_id]
        query_text = query_data.get("text", "")
        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = similarities[idx]
            results.append((doc_id, score))

        return results

    def print_results(
        self,
        results: List[Tuple[str, float]],
        query_id: Optional[str] = None,
        qrels: Optional[Dict] = None,
    ):
        if not results:
            print("Aucun résultat.")
            return
        print("Résultat de la recherche:")
        for doc_id, score in results:
            doc = self.corpus.get(doc_id, {})
            title = doc.get("title", "Sans titre")

            relevance_msg = ""
            if query_id and qrels:
                if query_id in qrels:
                    is_relevant = qrels[query_id].get(doc_id, 0)
                    status = "✅ OUI" if is_relevant == 1 else "❌ NON"
                    relevance_msg = f" [Pertinent: {status}]"
                else:
                    relevance_msg = " [Pertinence: ?]"

            print(f"[{score:.4f}]{relevance_msg} {doc_id} - {title[:80]}...")


class TfidfSearchEngine:
    def __init__(self, corpus: Dict[str, Dict], queries: Dict[str, Dict]):
        """Moteur utilisant TfidfVectorizer (Fréquence pondérée)."""
        self.corpus = corpus
        self.queries = queries
        self.doc_ids = list(corpus.keys())
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        self.matrix = None

    def fit(self):
        print("Entraînement du modèle TF-IDF...")
        corpus_texts = []
        for doc_id in self.doc_ids:
            doc = self.corpus[doc_id]
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            corpus_texts.append(text)

        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def search(self, query_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.matrix is None:
            raise ValueError("Le modèle n'est pas entraîné.")
        if query_id not in self.queries:
            return []
        query_data = self.queries[query_id]
        query_text = query_data.get("text", "")
        if not query_text:
            query_text = (
                f"{query_data.get('title', '')} {query_data.get('abstract', '')}"
            )
        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = similarities[idx]
            results.append((doc_id, score))
        return results

    def print_results(
        self,
        results: List[Tuple[str, float]],
        query_id: str,
        qrels: Optional[Dict] = None,
    ):
        if not results:
            print("Aucun résultat.")
            return
        print("Résultat de la recherche:")
        for doc_id, score in results:
            title = self.corpus[doc_id].get("title", "Sans titre")
            relevance = ""
            if qrels and query_id in qrels:
                status = "✅" if qrels[query_id].get(doc_id, 0) == 1 else "❌"
                relevance = f" [{status}]"
            print(f"[{score:.4f}]{relevance} {doc_id} - {title[:80]}...")
