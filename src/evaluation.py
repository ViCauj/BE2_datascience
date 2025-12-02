import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_engine(engine, qrels: dict, top_k: int = 5):
    """
    Évalue un moteur de recherche sur toutes les requêtes présentes dans qrels.
    Retourne la moyenne des métriques.
    """
    precisions = []
    recalls = []
    f1_scores = []
    aucs = []

    print(f"Évaluation en cours ({len(qrels)} requêtes)...")

    for query_id, candidates_dict in qrels.items():
        candidate_ids = list(candidates_dict.keys())
        results = engine.search(
            query_id, candidate_ids=candidate_ids, top_k=len(candidate_ids)
        )

        if not results:
            continue

        y_true = [candidates_dict[doc_id] for doc_id, score in results]
        y_scores = [score for doc_id, score in results]
        y_true_k = y_true[:top_k]

        n_relevant_retrieved = sum(y_true_k)
        n_relevant_total = sum(candidates_dict.values())

        precision = n_relevant_retrieved / top_k
        recall = n_relevant_retrieved / n_relevant_total if n_relevant_total > 0 else 0

        f1 = 0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        try:
            auc = (
                roc_auc_score(y_true, y_scores)
                if (sum(y_true) > 0 and sum(y_true) < len(y_true))
                else 0.5
            )
        except ValueError:
            auc = 0.5

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        aucs.append(auc)

    return {
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1-Score": np.mean(f1_scores),
        "AUC": np.mean(aucs),
    }
