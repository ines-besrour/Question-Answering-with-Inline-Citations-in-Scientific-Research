#!/usr/bin/env python
"""
bm25_fulltext_worker_batch.py
Fixed version that works with actual BM25Retriever API
"""
import sys
import json
from llama_index.retrievers.bm25 import BM25Retriever

def get_full_texts_batch(doc_ids, index_dir):
    """Get full texts for multiple documents efficiently"""
    try:
        # Load retriever once for all documents
        retriever = BM25Retriever.from_persist_dir(index_dir)
        
        results = {}
        
        # Try different ways to access the documents
        try:
            # Method 1: Try _docstore (newer versions)
            if hasattr(retriever, '_docstore'):
                docstore = retriever._docstore
                for node_id, doc in docstore.docs.items():
                    paper_id = doc.metadata.get("paper_id")
                    if paper_id in doc_ids:
                        full_text = doc.metadata.get("full_text", doc.get_content())
                        if full_text:
                            results[paper_id] = full_text
            
            # Method 2: Try docstore (older versions)
            elif hasattr(retriever, 'docstore'):
                docstore = retriever.docstore
                for node_id, doc in docstore.docs.items():
                    paper_id = doc.metadata.get("paper_id")
                    if paper_id in doc_ids:
                        full_text = doc.metadata.get("full_text", doc.get_content())
                        if full_text:
                            results[paper_id] = full_text
            
            # Method 3: Try accessing nodes directly
            elif hasattr(retriever, '_nodes'):
                nodes = retriever._nodes
                for node in nodes:
                    paper_id = node.metadata.get("paper_id")
                    if paper_id in doc_ids:
                        full_text = node.metadata.get("full_text", node.get_content())
                        if full_text:
                            results[paper_id] = full_text
            
            # Method 4: Fallback - use retrieve method for each doc
            else:
                for doc_id in doc_ids:
                    try:
                        # Use the retriever to find the document
                        search_results = retriever.retrieve(doc_id)
                        if search_results:
                            for result in search_results:
                                if result.node.metadata.get("paper_id") == doc_id:
                                    full_text = result.node.metadata.get("full_text", result.node.get_content())
                                    if full_text:
                                        results[doc_id] = full_text
                                    break
                    except:
                        continue
                        
        except Exception as e:
            return {"error": f"Failed to access BM25 documents: {str(e)}"}
        
        return results
        
    except Exception as e:
        return {"error": f"Failed to load BM25 retriever: {str(e)}"}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python bm25_fulltext_worker_batch.py <doc_ids_json> <index_dir>"}))
        sys.exit(1)
    
    doc_ids_json = sys.argv[1]
    index_dir = sys.argv[2]
    
    try:
        doc_ids = json.loads(doc_ids_json)
        if not isinstance(doc_ids, list):
            raise ValueError("doc_ids must be a list")
            
        results = get_full_texts_batch(doc_ids, index_dir)
        print(json.dumps(results))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))