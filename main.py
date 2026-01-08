from src import (
    load_corpus,
    load_queries,
    load_qrels,
    BagOfWordsCountSearchEngine,
    BagOfWordsTfidfSearchEngine,
    DenseSearchEngine,
    evaluate_engine,
    run_lda_analysis,
    CitationGraph,
    HybridSearchEngine,
    EnsembleSearchEngine,
    GCNSearchEngine,
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

    # --- Évaluation des Moteurs de Recherche ---

    for model in [
        BagOfWordsCountSearchEngine,
        BagOfWordsTfidfSearchEngine,
        DenseSearchEngine,
        HybridSearchEngine,
        GCNSearchEngine,
        EnsembleSearchEngine,
    ]:
        print(f"\n--- {model.__name__} ---")
        engine = model(corpus, queries)
        engine.fit()
        metrics = evaluate_engine(engine, qrels)
        for metric, value in metrics.items():
            print(f"   {metric:<10}: {value:.4f}")

        engine.create_submission(
            file_path="data/test_final.tsv",
            output_path=f"outputs/submissions/submission_{model.__name__}.csv",
        )

    # --- LDA ---

    run_lda_analysis(corpus, n_topics=5, n_top_words=12)

    # --- Graphe ---

    print("")
    graph_analysis = CitationGraph(corpus)
    graph_analysis.build()
    graph_analysis.analyze(top_k_centrality=5)
    # Visualisation (Uniquement si le graphe n'est pas vide)
    if graph_analysis.stats.get("num_edges", 0) > 0:
        graph_analysis.visualize(
            top_k=1500, output_path="outputs/Squelette_graphe_top_1500.png"
        )


if __name__ == "__main__":
    main()
