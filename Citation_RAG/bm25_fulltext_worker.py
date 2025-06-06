#!/usr/bin/env python
"""
bm25_fulltext_worker.py
Gets full text for a specific document ID from the BM25 index
"""
import sys
import json
from llama_index.retrievers.bm25 import BM25Retriever

def get_full_text_by_id(doc_id, index_dir):
    """Get full text for a specific document ID"""
    try:
        # Load the BM25 retriever
        retriever = BM25Retriever.from_persist_dir(index_dir)
        
        # Access the docstore to get document by ID
        docstore = retriever._docstore
        
        # Try to find the document
        for node_id, doc in docstore.docs.items():
            if doc.metadata.get("paper_id") == doc_id:
                # Return full text if available, otherwise return content
                full_text = doc.metadata.get("full_text", doc.get_content())
                return full_text
        
        return "NOT_FOUND"
    
    except Exception as e:
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bm25_fulltext_worker.py <doc_id> <index_dir>")
        sys.exit(1)
    
    doc_id = sys.argv[1]
    index_dir = sys.argv[2]
    
    result = get_full_text_by_id(doc_id, index_dir)
    print(result)