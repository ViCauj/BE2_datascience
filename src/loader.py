# Chargement des données
import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def load_corpus(file_path: str) -> Dict[str, Dict]:
    """
    Charge le corpus depuis un fichier JSONL.
    Retourne un dictionnaire : { 'doc_id': { 'title': ..., 'abstract': ... } }
    """
    corpus = {}
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

    print(f"Chargement du corpus depuis {file_path}...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc
    return corpus


def load_queries(file_path: str) -> Dict[str, Dict]:
    """
    Charge les requêtes depuis un fichier JSONL.
    """
    queries = {}
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

    print(f"Chargement des requêtes depuis {file_path}...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            query = json.loads(line)
            queries[query["_id"]] = query
    return queries


def load_qrels(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Charge les jugements de pertinence (valid.tsv).
    Retourne : { 'query_id': { 'doc_id': score, ... } }
    """
    qrels = {}
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

    print(f"Chargement des qrels depuis {file_path}...")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = int(row["score"])

            # Si on rencontre cette query pour la première fois
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score
    return qrels
