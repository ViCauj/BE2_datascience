from src import (
    load_corpus,
    load_queries,
    load_qrels,
    BagOfWordsSearchEngine,
    TfidfSearchEngine,
    DenseSearchEngine,
    evaluate_engine,
    run_lda_analysis,
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

    run_lda_analysis(corpus, n_topics=5, n_top_words=12)

    for model in [BagOfWordsSearchEngine, TfidfSearchEngine, DenseSearchEngine]:
        print(f"\n--- {model.__name__} ---")
        engine = model(corpus, queries)
        engine.fit()
        metrics = evaluate_engine(engine, qrels)
        for metric, value in metrics.items():
            print(f"   {metric:<10}: {value:.4f}")


if __name__ == "__main__":
    main()
