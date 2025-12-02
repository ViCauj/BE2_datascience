from sklearn.feature_extraction.text import TfidfVectorizer
from src import (
    load_corpus,
    load_queries,
    load_qrels,
    BagOfWordsSearchEngine,
    TfidfSearchEngine,
    DenseSearchEngine,
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

    first_query_id = next(iter(queries))

    for model in [BagOfWordsSearchEngine, TfidfSearchEngine, DenseSearchEngine]:
        print("")
        engine = model(corpus, queries)
        engine.fit()
        results = engine.search(first_query_id, top_k=25)
        engine.print_results(results, query_id=first_query_id, qrels=qrels)


if __name__ == "__main__":
    main()
