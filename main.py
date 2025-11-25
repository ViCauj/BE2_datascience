from src import load_corpus, load_queries, load_qrels
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


if __name__ == "__main__":
    main()
