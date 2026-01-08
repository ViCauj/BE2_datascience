from src import (
    load_corpus,
    load_queries,
    load_qrels,
    BagOfWordsSearchEngine,
    TfidfSearchEngine,
    DenseSearchEngine,
    evaluate_engine,
    run_lda_analysis,
    build_citation_graph,
    get_graph_statistics,
    get_top_centrality,
    visualize_backbone,
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
        print(f"\n--- {model.__name__} ---")
        engine = model(corpus, queries)
        engine.fit()
        metrics = evaluate_engine(engine, qrels)
        for metric, value in metrics.items():
            print(f"   {metric:<10}: {value:.4f}")

    run_lda_analysis(corpus, n_topics=5, n_top_words=12)

    print("")
    G = build_citation_graph(corpus)
    stats = get_graph_statistics(G)
    print("\n--- Statistiques du Graphe ---")
    print(f"Noeuds (Articles) : {stats['num_nodes']}")
    print(f"Arcs (Citations) : {stats['num_edges']}")
    print(f"Densité          : {stats['density']:.6f}")
    print(f"Citations reçues (moyenne) : {stats['avg_in_degree']:.2f}")
    print(f"Article le plus cité       : {stats['max_in_degree']} fois")

    if stats["num_edges"] > 0:
        top_articles = get_top_centrality(G, top_k=5)

        for doc_id, score in top_articles:
            title = corpus[doc_id].get("title", "Sans titre")
            print(f"[{score:.4f}] {doc_id} - {title[:80]}...")

        print("\nVisualisation du 'Squelette' du graphe...")
        visualize_backbone(G, top_k=500)
    else:
        print("Pas d'arcs dans le graphe, impossible de calculer la centralité.")


if __name__ == "__main__":
    main()
