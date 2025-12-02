from sklearn.feature_extraction.text import TfidfVectorizer
from src import (
    load_corpus,
    load_queries,
    load_qrels,
    BagOfWordsSearchEngine,
    TfidfSearchEngine,
    DenseSearchEngine,
    evaluate_engine,
)
import sys


def main():
    try:
        corpus = load_corpus("data/corpus.jsonl")
        queries = load_queries("data/queries.jsonl")
        qrels = load_qrels("data/valid.tsv")

        print(
            f"\n{len(corpus)} documents, {len(queries)} requêtes, {len(qrels)} jugements chargés."
        )
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        sys.exit(1)

    for model in [BagOfWordsSearchEngine, TfidfSearchEngine, DenseSearchEngine]:
        print("")
        engine = model(corpus, queries)
        engine.fit()
        metrics = evaluate_engine(engine, qrels)
        for metric, value in metrics.items():
            print(f"   {metric:<10}: {value:.4f}")


if __name__ == "__main__":
    main()
