from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack import Document
import os, glob, json, shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from config import (
    DATA_DIR,
    DOC_INDEX_DIR,
    EMBEDDING_MODEL,
    MODEL_FORMAT,
    EMBEDDING_DIM,
    USE_GPU
)

def _parse_jsonl_file(file_path):
    """
    Load a JSONL file and return a list of haystack Document objects.
    Executed in parallel.
    """
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    paper = json.loads(line.strip())
                    meta = paper.get("metadata", {}) or {}
                    title = meta.get("title", "")
                    abstract = paper.get("abstract", "")
                    sections = paper.get("sections", {}) or {}
                    sections_text = " ".join(
                        f"{sec}: {obj.get('text','')}" for sec, obj in sections.items() if isinstance(obj, dict)
                    )
                    content = f"{title}. {abstract}"
                    full_text = f"{title}. {abstract}. {sections_text}" 
                    documents.append(Document(
                        content=content,
                        meta={
                            "paper_id": paper.get("paper_id", ""),
                            "title": title,
                            "authors": meta.get("authors", []),
                            "full_text": full_text
                        }
                    ))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
    return documents

class Retriever:
    """
    Encapsulates FAISS index loading/building and semantic retrieval.
    """
    def __init__(self, data_directory: str, index_directory: str, top_k: int = 3):
        self.top_k = top_k
        self.data_directory = data_directory
        self.index_directory = index_directory
        self.document_store = self._build_or_load_index(data_directory, index_directory)
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format=MODEL_FORMAT,
            use_gpu=USE_GPU
        )

    def _build_or_load_index(self, data_directory: str, index_directory: str) -> FAISSDocumentStore:
        print("_build_or_load_index")
        faiss_file = os.path.join(index_directory, "faiss_index")
        config_file = os.path.join(index_directory, "faiss_index.json")
        if os.path.exists(faiss_file) and os.path.exists(config_file):
            print(f"Loading FAISS index from {index_directory}...")
            return FAISSDocumentStore.load(index_path=faiss_file, config_path=config_file)

        print(f"Building FAISS index at {index_directory}...")
        # Clean existing index directory
        if os.path.isdir(index_directory):
            shutil.rmtree(index_directory)
        os.makedirs(index_directory, exist_ok=True)

        # Gather JSONL files
        jsonl_files = glob.glob(os.path.join(data_directory, "**", "*.jsonl"), recursive=True)
        print(f"Found {len(jsonl_files)} JSONL files...")

        # Parse in parallel (safely limit cores)
        docs = []
        max_workers = max(4, multiprocessing.cpu_count())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            print(f"Using {max_workers} cores")
            for result in executor.map(_parse_jsonl_file, jsonl_files):
                docs.extend(result)

        # Build and write index
        db_path = os.path.join(index_directory, "index_store.db")

        document_store = FAISSDocumentStore(
            embedding_dim=EMBEDDING_DIM,
            faiss_index_factory_str="IVF100,Flat",
            sql_url=f"sqlite:///{db_path}"
        )

        BATCH_SIZE = 10000  # or smaller depending on memory
        total_docs = len(docs)
        print(f"Writing {total_docs} documents in batches of {BATCH_SIZE}...")

        for i in range(0, total_docs, BATCH_SIZE):
            batch = docs[i:i + BATCH_SIZE]
            document_store.write_documents(batch)
            print(f" → Written {min(i + BATCH_SIZE, total_docs)}/{total_docs} documents")
        print(f"Writing documents to FAISS index")
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format=MODEL_FORMAT,
            use_gpu=USE_GPU
        )
        print(f"before embedding")
        document_store.update_embeddings(retriever)
        print("after embedding")
        document_store.save(index_path=faiss_file, config_path=config_file)
        return document_store

    def retrieve(self, query: str, top_k: int = 5):
        """
        Perform semantic retrieval and return a list of result dicts.
        """
        docs = self.retriever.retrieve(query, top_k=top_k)
        print(docs)
        return [
            {
                "id": d.meta.get("paper_id"),
                "text": d.meta.get("full_text", d.content),  # Return full text if available, fallback to content
                "semantic_score": getattr(d, 'score', None)
            }
            for d in docs
        ]
