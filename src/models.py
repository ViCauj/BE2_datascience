from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple, Optional


class BaseSearchEngine:
    """
    Classe mère regroupant la logique commune aux moteurs de recherche vectoriels
    (Construction de matrice, Recherche Cosinus, Affichage).
    """

    def __init__(self, corpus: Dict[str, Dict], queries: Dict[str, Dict]):
        self.corpus = corpus
        self.queries = queries
        self.doc_ids = list(corpus.keys())
        self.vectorizer = None  # Sera défini par les classes filles
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

    def search(self, query_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
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
        print("\n")


class BagOfWordsSearchEngine(BaseSearchEngine):
    def __init__(self, corpus: Dict[str, Dict], queries: Dict[str, Dict]):
        super().__init__(corpus, queries)
        self.vectorizer = CountVectorizer(stop_words="english", max_features=50000)


class TfidfSearchEngine(BaseSearchEngine):
    def __init__(self, corpus: Dict[str, Dict], queries: Dict[str, Dict]):
        super().__init__(corpus, queries)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)

