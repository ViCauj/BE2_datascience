from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, List
import pickle
import os


def run_lda_analysis(corpus: Dict[str, Dict], n_topics: int = 5, n_top_words: int = 10):
    """
    Exécute une analyse thématique (LDA) sur le corpus.
    Affiche les mots les plus représentatifs pour chaque thème.
    """
    model_path = f"outputs/lda_model_{n_topics}.pkl"
    vect_path = "outputs/lda_vectorizer.pkl"

    try:
        print(f"\nTentative de chargement depuis {model_path}...")
        with open(model_path, "rb") as f:
            lda = pickle.load(f)
        with open(vect_path, "rb") as f:
            tf_vectorizer = pickle.load(f)
        print("Modèle chargé avec succès.")

    except (FileNotFoundError, OSError, pickle.PickleError) as e:
        print(f"Pas de modèle valide trouvé ({e}), on lance le calcul.")
        print(f"\n--- Analyse Thématique (LDA) : {n_topics} thèmes ---")
        print("Préparation des textes...")
        texts = []
        for doc in corpus.values():
            content = f"{doc.get('title', '')} {doc.get('text', '')}"
            texts.append(content)

        print("Vectorisation (CountVectorizer)...")
        tf_vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, max_features=1000, stop_words="english"
        )
        tf = tf_vectorizer.fit_transform(texts)

        print("Entraînement du modèle LDA...")
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_method="online",
            random_state=42,
            n_jobs=-1,
        )
        lda.fit(tf)

        print("Sauvegarde des modèles dans outputs/...")
        try:
            os.makedirs("outputs", exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(lda, f)
            with open(vect_path, "wb") as f:
                pickle.dump(tf_vectorizer, f)
        except Exception as save_error:
            print(f"Attention : Impossible de sauvegarder le modèle ({save_error})")

    print(f"\n--- Résultat : Mots-clés par Thème ({n_topics} thèmes) ---")
    if tf_vectorizer and lda:
        feature_names = tf_vectorizer.get_feature_names_out()

        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[: -n_top_words - 1 : -1]
            top_words = [feature_names[i] for i in top_indices]
            print(f"Thème #{topic_idx + 1}: {', '.join(top_words)}")

    return lda, tf_vectorizer
