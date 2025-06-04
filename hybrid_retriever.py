from haystack import Document
from haystack_retriever import HaystackRetriever
import subprocess, json, shlex

def normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]

class Retriever:
    """
    Hybrid retriever: combines Haystack FAISS (E5) and LlamaIndex BM25.
    """
    def __init__(self, e5_index_directory: str, bm25_index_directory: str, top_k: int = 5):
        self.top_k = top_k
        self.e5 = HaystackRetriever(e5_index_directory)
        self.bm25_python = "/home/inbe405h/bm25_env/bin/python"   # path to the bm25_env interpreter
        self.bm25_script = "bm25_worker.py"                     
        self.bm25_index_directory = bm25_index_directory

    def retrieve(self, query: str, top_k: int = 5):
        # E5 (semantic) retrieval
        e5_docs = self.e5.retrieve(query, top_k=top_k * 2)
        e5_map = {doc.meta["paper_id"]: doc for doc in e5_docs}
        e5_scores = {doc.meta["paper_id"]: doc.score for doc in e5_docs}

        # BM25 retrieval
        cmd = [
            self.bm25_python,
            self.bm25_script,
            query,
            self.bm25_index_directory,
            str(top_k)
        ]
        bm25_raw = subprocess.check_output(cmd, text=True)   
        print(bm25_raw)# run in bm25_env
        bm25_items = json.loads(bm25_raw)# list of dicts

        bm25_map    = {it["paper_id"]: it for it in bm25_items}
        bm25_scores = {it["paper_id"]: it["score"] for it in bm25_items}

        # Union of paper IDs
        all_ids = set(e5_scores.keys()).union(bm25_scores.keys())

        # Normalize
        e5_norm = normalize([e5_scores.get(pid, 0.0) for pid in all_ids])
        bm25_norm = normalize([bm25_scores.get(pid, 0.0) for pid in all_ids])

        # Combine
        combined = []
        for pid, e5_s, bm25_s in zip(all_ids, e5_norm, bm25_norm):
            final_score = 0.65 * e5_s + 0.35 * bm25_s
            if pid in e5_map:
                doc = e5_map[pid]
                print('E5 node:', doc)
            else:
                node = bm25_map[pid]
                print('BM25 node:', node)
                doc = Document(id=node.get("paper_id",""), content=node.get("text",""))
            combined.append((final_score, doc))

        combined.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "id": d.meta.get("paper_id"),
                "abstract": d.content,
                "semantic_score": getattr(d, 'score', None)
            }
            for _, d in combined[:top_k]
        ]
