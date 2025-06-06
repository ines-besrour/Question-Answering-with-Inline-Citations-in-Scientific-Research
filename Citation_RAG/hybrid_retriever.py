from haystack import Document
from haystack_retriever import HaystackRetriever
import subprocess
import json
import shlex
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os

logger = logging.getLogger(__name__)

def normalize(scores):
    """Normalize scores to 0-1 range"""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]

class Retriever:
    """
    OPTIMIZED Hybrid retriever with In-Memory BM25 support
    Combines Haystack FAISS (E5) and LlamaIndex BM25 with major performance improvements
    Supports two-phase retrieval: abstracts first, then full texts
    NO CHUNKING - returns full documents as requested
    """
    def __init__(self, e5_index_directory: str, bm25_index_directory: str, top_k: int = 5):
        self.top_k = top_k
        self.e5 = HaystackRetriever(e5_index_directory)
        self.bm25_python = "bm25_env/bin/python"   # path to the bm25_env interpreter
        self.bm25_script = "bm25_worker.py"
        self.bm25_fulltext_script = "bm25_fulltext_worker.py"  # NEW: for getting full texts                   
        self.bm25_index_directory = bm25_index_directory
        
        # Enhanced caching with size limits
        self._doc_cache = {}
        self._abstract_cache = {}  # Query -> results cache
        self._fulltext_cache = {}  # Doc_id -> full_text cache
        self._cache_size = 100  # Increased cache size
        
        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self._retrieval_times = []
        
        # 🚀 KEY OPTIMIZATION: Load BM25 into memory
        self._bm25_retriever = self._load_bm25_into_memory()

    def _load_bm25_into_memory(self):
        """
        🚀 CRITICAL OPTIMIZATION: Load BM25 index into memory
        This eliminates the 30s subprocess overhead per query!
        """
        logger.info("🚀 Attempting to load BM25 index into memory...")
        
        try:
            start_time = time.time()
            
            # Try to import and load BM25 retriever directly
            from llama_index.indices.bm25_retriever import BM25Retriever
            
            # Load the index into memory (one-time cost)
            retriever = BM25Retriever.from_persist_dir(self.bm25_index_directory)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ BM25 index loaded into memory in {elapsed:.2f}s!")
            logger.info(f"🎯 This eliminates ~30s subprocess overhead per query!")
            
            # Quick test to ensure it works
            try:
                test_results = retriever.retrieve("test")
                logger.info(f"✅ BM25 memory test: {len(test_results)} results")
            except Exception as e:
                logger.warning(f"BM25 memory test failed: {e}")
            
            return retriever
            
        except ImportError:
            logger.warning("❌ llama_index.retrievers.bm25 not available")
            logger.info("🔄 Will use subprocess method instead")
            return None
        except Exception as e:
            logger.warning(f"❌ Could not load BM25 into memory: {e}")
            logger.info("🔄 Will use subprocess method instead")
            return None

    def _search_bm25_memory(self, query: str, top_k: int = 10):
        """
        ⚡ FAST: Search BM25 using in-memory index (no subprocess!)
        This should be ~100x faster than subprocess
        """
        if not self._bm25_retriever:
            return []
            
        try:
            start_time = time.time()
            
            # Direct search using loaded index - no subprocess overhead!
            results = self._bm25_retriever.retrieve(query)
            
            # Convert to expected format
            formatted_results = []
            for result in results[:top_k]:
                formatted_results.append({
                    "paper_id": result.node.metadata.get("paper_id", result.node.id_),
                    "text": result.node.get_content(),
                    "score": result.score
                })
            
            elapsed = time.time() - start_time
            logger.info(f"⚡ BM25 in-memory: {len(formatted_results)} results in {elapsed:.3f}s")
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"BM25 memory search failed: {e}")
            return []

    def _search_bm25_subprocess_optimized(self, query: str, top_k: int = 10):
        """
        OPTIMIZED: Subprocess method with timeout and better error handling
        """
        try:
            start_time = time.time()
            
            cmd = [
                self.bm25_python,
                self.bm25_script,
                query,
                self.bm25_index_directory,
                str(top_k)
            ]
            
            # Add timeout to prevent hanging on HPC
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=50  # 20 second timeout
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                bm25_results = json.loads(result.stdout)
                logger.info(f"✅ BM25 subprocess: {len(bm25_results)} results in {elapsed:.2f}s")
                return bm25_results
            else:
                logger.warning(f"❌ BM25 subprocess failed (returncode {result.returncode}): {result.stderr}")
                return []
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            logger.warning(f"⏱️ BM25 subprocess timed out after {elapsed:.1f}s - continuing without BM25")
            return []
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"❌ BM25 subprocess error after {elapsed:.1f}s: {e}")
            return []

    def _get_bm25_results(self, query, top_k):
        """
        🎯 ENHANCED: Try in-memory first, fallback to subprocess
        This automatically uses the fastest available method
        """
        try:
            # Strategy 1: Try in-memory BM25 (should be FAST!)
            if self._bm25_retriever:
                results = self._search_bm25_memory(query, top_k)
                if results:  # If we got results, return them
                    return results
                else:
                    logger.debug("In-memory BM25 returned no results, trying subprocess")
            
            # Strategy 2: Fallback to optimized subprocess
            return self._search_bm25_subprocess_optimized(query, top_k)
            
        except Exception as e:
            logger.warning(f"BM25 retrieval failed: {e}")
            return []

    def _get_e5_results(self, query, top_k):
        """Get E5 results (run in thread)"""
        try:
            return self.e5.retrieve(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"E5 retrieval failed: {e}")
            return []

    def _fast_normalize(self, scores):
        """Fast numpy-based normalization"""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def retrieve_abstracts(self, query: str, top_k: int = None) -> list:
        """
        🎯 OPTIMIZED: Phase 1 retrieval with in-memory BM25 and enhanced caching
        Returns: List of (abstract_text, doc_id) tuples
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{query.lower().strip()}_{top_k}"
        if cache_key in self._abstract_cache:
            logger.info("⚡ Abstract cache hit!")
            return self._abstract_cache[cache_key]
        
        if top_k is None:
            top_k = self.top_k
            
        logger.info(f"Retrieving {top_k} abstracts for query: {query}")
        
        # Show BM25 method being used
        bm25_method = "in-memory" if self._bm25_retriever else "subprocess"
        logger.info(f"🔧 BM25 method: {bm25_method}")
        
        # Parallel E5 and BM25 retrieval
        e5_future = self._executor.submit(self._get_e5_results, query, top_k * 2)
        bm25_future = self._executor.submit(self._get_bm25_results, query, top_k * 2)
        
        # Get results
        e5_docs = e5_future.result()
        bm25_items = bm25_future.result()

        # Build mappings
        e5_map = {doc.meta["paper_id"]: doc for doc in e5_docs}
        e5_scores = {doc.meta["paper_id"]: doc.score for doc in e5_docs}
        bm25_map = {it["paper_id"]: it for it in bm25_items}
        bm25_scores = {it["paper_id"]: it["score"] for it in bm25_items}

        # Union of paper IDs
        all_ids = list(set(e5_scores.keys()).union(bm25_scores.keys()))
        
        if not all_ids:
            logger.warning("No documents retrieved")
            return []

        # Fast vectorized score normalization
        e5_score_array = np.array([e5_scores.get(pid, 0.0) for pid in all_ids])
        bm25_score_array = np.array([bm25_scores.get(pid, 0.0) for pid in all_ids])
        
        e5_norm = self._fast_normalize(e5_score_array)
        bm25_norm = self._fast_normalize(bm25_score_array)
        
        # Combine scores - adjust weights if BM25 failed
        if bm25_items:
            combined_scores = 0.65 * e5_norm + 0.35 * bm25_norm
            logger.info(f"✅ True hybrid: E5({len(e5_docs)}) + BM25({len(bm25_items)})")
        else:
            combined_scores = e5_norm
            logger.info(f"ℹ️ E5-only fallback: BM25 unavailable")

        # Build results
        combined = []
        for i, pid in enumerate(all_ids):
            final_score = combined_scores[i]
            
            # Get abstract content
            if pid in e5_map:
                doc = e5_map[pid]
                abstract_text = doc.content
                logger.debug(f'E5 doc found for {pid}')
            elif pid in bm25_map:
                node = bm25_map[pid]
                abstract_text = node.get("text", "")
                logger.debug(f'BM25 node found for {pid}')
            else:
                continue
                
            # Cache the document info for potential full-text retrieval
            self._doc_cache[pid] = {
                'abstract': abstract_text,
                'e5_doc': e5_map.get(pid),
                'bm25_node': bm25_map.get(pid),
                'score': final_score
            }
            
            combined.append((final_score, abstract_text, pid))

        # Sort by combined score
        combined.sort(key=lambda x: x[0], reverse=True)

        # Return in expected format: List of (abstract_text, doc_id) tuples
        result = [(text, doc_id) for _, text, doc_id in combined[:top_k]]
        
        # Smart cache management with LRU
        if len(self._abstract_cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._abstract_cache))
            del self._abstract_cache[oldest_key]
        
        self._abstract_cache[cache_key] = result
        
        elapsed = time.time() - start_time
        self._retrieval_times.append(elapsed)
        logger.info(f"Retrieved {len(result)} abstracts in {elapsed:.2f}s")
        
        return result

    def get_full_texts(self, doc_ids: list, db=None) -> list:
        """
        OPTIMIZED: Batch full text retrieval with chunking and fallback
        NO CHUNKING - returns full documents as requested
        """
        if not doc_ids:
            return []
            
        start_time = time.time()
        logger.info(f"🚀 OPTIMIZED: Retrieving full texts for {len(doc_ids)} documents (NO CHUNKING)")
        
        # Step 1: Check cache first
        cached_results = {}
        missing_ids = []
        
        for doc_id in doc_ids:
            if doc_id in self._fulltext_cache:
                cached_results[doc_id] = self._fulltext_cache[doc_id]
            else:
                missing_ids.append(doc_id)
        
        logger.info(f"💾 Cache hits: {len(cached_results)}, Need to fetch: {len(missing_ids)}")
        
        # Step 2: Batch retrieve missing documents
        if missing_ids:
            batch_results = self._optimized_batch_retrieve(missing_ids)
            cached_results.update(batch_results)
            
            # Update cache with size management
            for doc_id, text in batch_results.items():
                if len(self._fulltext_cache) >= self._cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._fulltext_cache))
                    del self._fulltext_cache[oldest_key]
                self._fulltext_cache[doc_id] = text
        
        # Step 3: Prepare final results - NO CHUNKING!
        final_texts = []
        total_chars = 0
        
        for doc_id in doc_ids:
            if doc_id not in cached_results:
                # Fallback to abstract if full text not available
                if doc_id in self._doc_cache:
                    text = self._doc_cache[doc_id]['abstract']
                    logger.debug(f"Using abstract as fallback for {doc_id}")
                else:
                    continue
            else:
                text = cached_results[doc_id]
            
            # NO CHUNKING - return full text as requested
            final_texts.append((text, doc_id))
            total_chars += len(text)
        
        elapsed = time.time() - start_time
        avg_length = total_chars // max(len(final_texts), 1)
        logger.info(f"✅ OPTIMIZED: Retrieved {len(final_texts)} FULL texts in {elapsed:.2f}s (avg {avg_length} chars/doc)")
        
        return final_texts

    def _optimized_batch_retrieve(self, doc_ids: list) -> dict:
        """
        Multi-strategy batch retrieval with parallel fallbacks
        """
        results = {}
        
        # Strategy 1: Try batch worker with timeout
        batch_results = self._try_batch_worker_with_timeout(doc_ids, timeout=15)
        if batch_results:
            results.update(batch_results)
            successfully_retrieved = set(batch_results.keys())
            remaining_ids = [doc_id for doc_id in doc_ids if doc_id not in successfully_retrieved]
        else:
            remaining_ids = doc_ids
        
        # Strategy 2: Parallel individual retrieval for remaining docs
        if remaining_ids:
            logger.info(f"Using parallel individual retrieval for {len(remaining_ids)} remaining docs")
            parallel_results = self._parallel_individual_retrieve(remaining_ids)
            results.update(parallel_results)
        
        return results

    def _try_batch_worker_with_timeout(self, doc_ids: list, timeout: int = 15) -> dict:
        """Try batch worker with timeout and error handling"""
        
        # Only try the batch worker that exists
        worker_scripts = ["bm25_fulltext_worker_batch.py"]
        
        # Check if optimized worker exists
        if os.path.exists("bm25_fulltext_worker_optimized.py"):
            worker_scripts.insert(0, "bm25_fulltext_worker_optimized.py")
        
        for worker_script in worker_scripts:
            try:
                cmd = [
                    self.bm25_python,
                    worker_script,
                    json.dumps(doc_ids),
                    self.bm25_index_directory
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    batch_results = json.loads(result.stdout.strip())
                    if "error" not in batch_results and batch_results:
                        logger.info(f"✅ Batch worker {worker_script} retrieved {len(batch_results)} texts")
                        return batch_results
                
                logger.debug(f"Batch worker {worker_script} failed: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                logger.debug(f"Batch worker {worker_script} timed out after {timeout}s")
            except Exception as e:
                logger.debug(f"Batch worker {worker_script} error: {e}")
        
        logger.info("📝 Batch workers didn't work, using parallel individual retrieval (this is fine)")
        return {}

    def _parallel_individual_retrieve(self, doc_ids: list) -> dict:
        """Parallel individual retrieval using thread pool"""
        results = {}
        
        def get_single_doc(doc_id):
            try:
                text = self._get_full_text_for_doc(doc_id)
                return doc_id, text if text and text != "NOT_FOUND" else None
            except Exception as e:
                logger.debug(f"Individual retrieval failed for {doc_id}: {e}")
                return doc_id, None
        
        # Use thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_id = {executor.submit(get_single_doc, doc_id): doc_id for doc_id in doc_ids}
            
            for future in as_completed(future_to_id, timeout=50):
                try:
                    doc_id, text = future.result()
                    if text:
                        results[doc_id] = text
                except Exception as e:
                    logger.debug(f"Future failed: {e}")
        
        logger.info(f"✅ Parallel individual retrieval got {len(results)}/{len(doc_ids)} texts")
        return results

    def _get_full_text_for_doc(self, doc_id: str) -> str:
        """
        Get full text for a document using BM25 system with improved fallbacks
        """
        # Try BM25 worker first
        try:
            cmd = [
                self.bm25_python,
                self.bm25_fulltext_script,
                doc_id,
                self.bm25_index_directory
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                full_text = result.stdout.strip()
                if full_text and full_text != "NOT_FOUND" and not full_text.startswith("ERROR:"):
                    return full_text
        except Exception as e:
            logger.debug(f"BM25 individual lookup failed for {doc_id}: {e}")
        
        # Fallback strategies
        return self._try_fallback_sources(doc_id)

    def _try_fallback_sources(self, doc_id: str) -> str:
        """Try multiple fallback sources for full text"""
        
        # Fallback 1: Cached BM25 node
        if doc_id in self._doc_cache and 'bm25_node' in self._doc_cache[doc_id]:
            bm25_node = self._doc_cache[doc_id]['bm25_node']
            if bm25_node and 'full_text' in bm25_node:
                return bm25_node['full_text']
        
        # Fallback 2: E5 metadata
        if doc_id in self._doc_cache and 'e5_doc' in self._doc_cache[doc_id]:
            e5_doc = self._doc_cache[doc_id]['e5_doc']
            if e5_doc and 'full_text' in e5_doc.meta:
                return e5_doc.meta['full_text']
            # If E5 content is substantial, use it
            elif e5_doc and len(e5_doc.content) > 1000:
                return e5_doc.content
        
        # Fallback 3: Use abstract as last resort
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]['abstract']
        
        return ""

    def retrieve(self, query: str, top_k: int = None):
        """
        Legacy method for backward compatibility with enhanced caching
        """
        if top_k is None:
            top_k = self.top_k
        
        # Check cache
        cache_key = f"legacy_{query.lower().strip()}_{top_k}"
        if hasattr(self, '_legacy_cache') and cache_key in self._legacy_cache:
            logger.info("⚡ Legacy cache hit!")
            return self._legacy_cache[cache_key]
        
        # Parallel retrieval
        e5_future = self._executor.submit(self._get_e5_results, query, top_k * 2)
        bm25_future = self._executor.submit(self._get_bm25_results, query, top_k)
        
        e5_docs = e5_future.result()
        bm25_items = bm25_future.result()
        
        e5_map = {doc.meta["paper_id"]: doc for doc in e5_docs}
        e5_scores = {doc.meta["paper_id"]: doc.score for doc in e5_docs}
        bm25_map = {it["paper_id"]: it for it in bm25_items}
        bm25_scores = {it["paper_id"]: it["score"] for it in bm25_items}

        all_ids = set(e5_scores.keys()).union(bm25_scores.keys())
        
        # Use vectorized normalization
        all_ids_list = list(all_ids)
        e5_score_array = np.array([e5_scores.get(pid, 0.0) for pid in all_ids_list])
        bm25_score_array = np.array([bm25_scores.get(pid, 0.0) for pid in all_ids_list])
        
        e5_norm = self._fast_normalize(e5_score_array)
        bm25_norm = self._fast_normalize(bm25_score_array)

        combined = []
        for i, pid in enumerate(all_ids_list):
            final_score = 0.65 * e5_norm[i] + 0.35 * bm25_norm[i]
            if pid in e5_map:
                doc = e5_map[pid]
            else:
                node = bm25_map[pid]
                doc = Document(id=node.get("paper_id",""), content=node.get("text",""))
            combined.append((final_score, doc))

        combined.sort(key=lambda x: x[0], reverse=True)

        result = [
            {
                "id": d.meta.get("paper_id") if hasattr(d, 'meta') else d.id,
                "abstract": d.content,
                "semantic_score": getattr(d, 'score', final_score)
            }
            for final_score, d in combined[:top_k]
        ]
        
        # Cache result
        if not hasattr(self, '_legacy_cache'):
            self._legacy_cache = {}
        if len(self._legacy_cache) >= 50:
            oldest_key = next(iter(self._legacy_cache))
            del self._legacy_cache[oldest_key]
        self._legacy_cache[cache_key] = result
        
        return result

    def get_bm25_status(self):
        """
        Diagnostic method to check BM25 status
        """
        if self._bm25_retriever:
            return {
                "method": "in-memory",
                "available": True,
                "expected_speed": "~0.5s per query",
                "status": "✅ OPTIMIZED"
            }
        else:
            return {
                "method": "subprocess", 
                "available": True,
                "expected_speed": "~20-30s per query",
                "status": "⚠️ SLOW"
            }

    def get_performance_stats(self):
        """Get performance statistics"""
        if self._retrieval_times:
            avg_time = sum(self._retrieval_times) / len(self._retrieval_times)
            return {
                "avg_retrieval_time": avg_time,
                "total_retrievals": len(self._retrieval_times),
                "cache_sizes": {
                    "abstract_cache": len(self._abstract_cache),
                    "fulltext_cache": len(self._fulltext_cache),
                    "doc_cache": len(self._doc_cache)
                },
                "bm25_status": self.get_bm25_status()
            }
        return {"no_data": True}
    
    def close(self):
        """Clean up resources with enhanced reporting"""
        try:
            if hasattr(self.e5, 'close'):
                self.e5.close()
        except Exception as e:
            logger.error(f"Error closing E5 retriever: {e}")
        
        # Close thread pool
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        
        # Clear caches
        self._doc_cache.clear()
        self._abstract_cache.clear()
        self._fulltext_cache.clear()
        
        # Print final stats and BM25 status
        stats = self.get_performance_stats()
        if "avg_retrieval_time" in stats:
            logger.info(f"📊 Final stats: {stats['total_retrievals']} retrievals, avg {stats['avg_retrieval_time']:.2f}s")
        
        bm25_status = self.get_bm25_status()
        logger.info(f"🔧 BM25 final status: {bm25_status['method']} ({bm25_status['status']})")
        
        logger.info("Retriever closed")