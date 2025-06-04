from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack import Document

from config import EMBEDDING_MODEL, MODEL_FORMAT, EMBEDDING_DIM, USE_GPU

class HaystackRetriever:
    def __init__(self, e5_index_directory: str):
        self.document_store = FAISSDocumentStore.load(
            index_path=f"{e5_index_directory}/faiss_index",
            config_path=f"{e5_index_directory}/faiss_index.json"
        )
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=EMBEDDING_MODEL,
            model_format=MODEL_FORMAT,
            use_gpu=USE_GPU
        )

    def retrieve(self, query: str, top_k: int = 5):
        docs = self.retriever.retrieve(query, top_k=top_k)
        return docs
