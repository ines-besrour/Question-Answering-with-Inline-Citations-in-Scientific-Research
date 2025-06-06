#!/usr/bin/env python
"""
bm25_fulltext_worker_optimized.py
Optimized batch worker with multiple strategies and caching
"""
import sys
import json
import time
import pickle
import os
from llama_index.retrievers.bm25 import BM25Retriever

# Global cache to reuse loaded retriever
_cached_retriever = None
_cached_index_dir = None

def load_retriever_cached(index_dir):
    """Load retriever with caching to avoid reloading"""
    global _cached_retriever, _cached_index_dir
    
    if _cached_retriever is None or _cached_index_dir != index_dir:
        print(f"Loading BM25 retriever from {index_dir}...", file=sys.stderr)
        start_time = time.time()
        _cached_retriever = BM25Retriever.from_persist_dir(index_dir)
        _cached_index_dir = index_dir
        elapsed = time.time() - start_time
        print(f"Loaded in {elapsed:.2f}s", file=sys.stderr)
    
    return _cached_retriever

def get_full_texts_optimized(doc_ids, index_dir):
    """Optimized full text retrieval with multiple strategies"""
    try:
        retriever = load_retriever_cached(index_dir)
        results = {}
        
        # Strategy 1: Direct docstore access (fastest)
        docstore = None
        try:
            if hasattr(retriever, '_docstore'):
                docstore = retriever._docstore
            elif hasattr(retriever, 'docstore'):
                docstore = retriever.docstore
                
            if docstore and hasattr(docstore, 'docs'):
                print(f"Using direct docstore access for {len(doc_ids)} docs", file=sys.stderr)
                
                # Build lookup map for faster access
                paper_id_to_doc = {}
                for node_id, doc in docstore.docs.items():
                    paper_id = doc.metadata.get("paper_id")
                    if paper_id:
                        paper_id_to_doc[paper_id] = doc
                
                # Fast lookup
                for doc_id in doc_ids:
                    if doc_id in paper_id_to_doc:
                        doc = paper_id_to_doc[doc_id]
                        full_text = doc.metadata.get("full_text")
                        if not full_text:
                            full_text = doc.get_content()
                        if full_text and len(full_text.strip()) > 50:  # Ensure substantial content
                            results[doc_id] = full_text
                
                if len(results) == len(doc_ids):
                    print(f"✅ Direct access got all {len(results)} texts", file=sys.stderr)
                    return results
                    
        except Exception as e:
            print(f"Direct access failed: {e}", file=sys.stderr)
        
        # Strategy 2: Node access (if docstore method failed)
        missing_ids = [doc_id for doc_id in doc_ids if doc_id not in results]
        if missing_ids and hasattr(retriever, '_nodes'):
            print(f"Using node access for {len(missing_ids)} missing docs", file=sys.stderr)
            try:
                nodes = retriever._nodes
                for node in nodes:
                    paper_id = node.metadata.get("paper_id")
                    if paper_id in missing_ids:
                        full_text = node.metadata.get("full_text", node.get_content())
                        if full_text and len(full_text.strip()) > 50:
                            results[paper_id] = full_text
                            
            except Exception as e:
                print(f"Node access failed: {e}", file=sys.stderr)
        
        # Strategy 3: Search-based retrieval (slowest but most reliable)
        still_missing = [doc_id for doc_id in doc_ids if doc_id not in results]
        if still_missing:
            print(f"Using search-based retrieval for {len(still_missing)} docs", file=sys.stderr)
            
            for doc_id in still_missing:
                try:
                    # Try exact ID match first
                    search_results = retriever.retrieve(doc_id)
                    for result in search_results:
                        if result.node.metadata.get("paper_id") == doc_id:
                            full_text = result.node.metadata.get("full_text", result.node.get_content())
                            if full_text and len(full_text.strip()) > 50:
                                results[doc_id] = full_text
                            break
                    
                    # If not found, try partial match
                    if doc_id not in results:
                        search_results = retriever.retrieve(f"paper_id:{doc_id}")
                        if search_results:
                            best_match = search_results[0]
                            full_text = best_match.node.metadata.get("full_text", best_match.node.get_content())
                            if full_text and len(full_text.strip()) > 50:
                                results[doc_id] = full_text
                                
                except Exception as e:
                    print(f"Search failed for {doc_id}: {e}", file=sys.stderr)
                    continue
        
        print(f"✅ Retrieved {len(results)}/{len(doc_ids)} full texts", file=sys.stderr)
        return results
        
    except Exception as e:
        return {"error": f"Failed to retrieve full texts: {str(e)}"}

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python script.py <doc_ids_json> <index_dir>"}))
        return
    
    try:
        doc_ids_json = sys.argv[1]
        index_dir = sys.argv[2]
        
        # Parse input
        doc_ids = json.loads(doc_ids_json)
        if not isinstance(doc_ids, list):
            raise ValueError("doc_ids must be a list")
        
        # Limit batch size to prevent memory issues
        if len(doc_ids) > 50:
            print(f"Warning: Large batch size {len(doc_ids)}, limiting to 50", file=sys.stderr)
            doc_ids = doc_ids[:50]
        
        start_time = time.time()
        results = get_full_texts_optimized(doc_ids, index_dir)
        elapsed = time.time() - start_time
        
        print(f"Batch retrieval completed in {elapsed:.2f}s", file=sys.stderr)
        print(json.dumps(results))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()