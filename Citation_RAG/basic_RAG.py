#!/usr/bin/env python3
"""
Enhanced Basic RAG with Hybrid Retrieval and Passage Tracking
- Uses hybrid E5+BM25 retrieval for 5 abstracts
- Generates answers with passage attribution
- Tracks which specific passages were used in the answer
"""

import argparse
import json
import time
import datetime
import os
import logging
import numpy as np
from tqdm import tqdm
import re
import random
import string
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Generate a unique ID for log filename
def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/enhanced_basic_rag_{timestamp}_{random_str}.log"

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging with unique filename
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("Enhanced_Basic_RAG")

# Import your Retriever class
from hybrid_retriever import Retriever

class PassageTracker:
    """Tracks which passages from retrieved documents were used in the answer"""
    
    def __init__(self):
        self.passages = {}  # doc_id -> passage_text
        self.passage_to_doc = {}  # passage_id -> doc_id
        self.next_passage_id = 1
    
    def add_document(self, doc_text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split document into passages and assign IDs
        Returns list of passage info for prompt
        """
        # Split document into sentences/passages
        sentences = self._split_into_sentences(doc_text)
        
        # Group sentences into meaningful passages (3-4 sentences each)
        passages = self._group_into_passages(sentences)
        
        passage_info = []
        for passage_text in passages:
            if len(passage_text.strip()) > 50:  # Skip very short passages
                passage_id = f"P{self.next_passage_id}"
                
                self.passages[passage_id] = {
                    'text': passage_text,
                    'doc_id': doc_id,
                    'passage_id': passage_id
                }
                self.passage_to_doc[passage_id] = doc_id
                
                passage_info.append({
                    'passage_id': passage_id,
                    'text': passage_text,
                    'doc_id': doc_id
                })
                
                self.next_passage_id += 1
        
        return passage_info
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        return sentences
    
    def _group_into_passages(self, sentences: List[str], passage_size: int = 3) -> List[str]:
        """Group sentences into passages"""
        passages = []
        
        for i in range(0, len(sentences), passage_size):
            passage_sentences = sentences[i:i+passage_size]
            passage_text = '. '.join(passage_sentences)
            if passage_text and not passage_text.endswith('.'):
                passage_text += '.'
            passages.append(passage_text)
        
        return passages
    
    def extract_used_passages(self, answer: str) -> List[Dict[str, Any]]:
        """
        Extract which passages were referenced in the answer
        Based on passage IDs mentioned in the answer
        """
        used_passages = []
        
        # Find all passage IDs mentioned in the answer
        passage_ids = re.findall(r'P\d+', answer)
        
        for passage_id in set(passage_ids):  # Remove duplicates
            if passage_id in self.passages:
                passage_info = self.passages[passage_id]
                used_passages.append({
                    'passage_id': passage_id,
                    'text': passage_info['text'],
                    'doc_id': passage_info['doc_id'],
                    'usage_count': passage_ids.count(passage_id)
                })
        
        return used_passages
    
    def get_passage_summary(self) -> Dict[str, int]:
        """Get summary of passages by document"""
        doc_passage_count = {}
        for passage_id, doc_id in self.passage_to_doc.items():
            doc_passage_count[doc_id] = doc_passage_count.get(doc_id, 0) + 1
        return doc_passage_count


class EnhancedBasicRAG:
    """Enhanced Basic RAG with passage tracking and attribution"""
    
    def __init__(self, retriever, agent_model=None, top_k=5, falcon_api_key=None):
        """
        Initialize Enhanced Basic RAG
        
        Args:
            retriever: Hybrid Retriever instance
            agent_model: Model name or pre-initialized agent
            top_k: Number of abstracts to retrieve (default: 5)
            falcon_api_key: API key for Falcon if using API
        """
        self.retriever = retriever
        self.top_k = top_k
        
        # Initialize LLM agent
        if isinstance(agent_model, str):
            if "falcon" in agent_model.lower() and falcon_api_key:
                from api_agent import FalconAgent
                self.agent = FalconAgent(api_key=falcon_api_key)
                logger.info("Using Falcon agent with API")
            else:
                from local_agent import LLMAgent
                self.agent = LLMAgent(agent_model)
                logger.info(f"Using local LLM agent with model {agent_model}")
        else:
            self.agent = agent_model
            logger.info("Using pre-initialized agent")
        
        logger.info(f"Enhanced Basic RAG initialized with top_k={top_k}")
    
    def _create_rag_prompt_with_passages(self, query: str, passage_info: List[Dict[str, Any]]) -> str:
        """
        Create RAG prompt with numbered passages for tracking
        """
        # Build passages section with IDs
        passages_text = ""
        for i, passage in enumerate(passage_info):
            passages_text += f"\n{passage['passage_id']} (from {passage['doc_id']}):\n{passage['text']}\n"
        
        return f"""You are an accurate and helpful AI assistant. Answer the question based ONLY on the information provided in the passages below. 

IMPORTANT INSTRUCTIONS:
1. When you use information from a passage, ALWAYS reference it by its ID (like P1, P2, P3, etc.)
2. Put the passage ID immediately after the information you use from that passage
3. If multiple passages support the same point, reference all relevant passage IDs
4. If the passages don't contain enough information to answer the question, state that clearly
5. Do NOT make up information not found in the passages

PASSAGES:
{passages_text}

QUESTION: {query}

Please provide a comprehensive answer using the passages above, and make sure to reference passage IDs (P1, P2, etc.) when using information from specific passages.

ANSWER:"""
    
    def _extract_paper_metadata(self, doc_id: str, doc_text: str) -> Dict[str, str]:
        """Extract basic metadata from document"""
        # Try to extract title and basic info
        lines = doc_text.split('\n')[:10]  # Check first 10 lines
        
        title = "Unknown Title"
        authors = "Unknown Authors"
        
        # Simple heuristics for title extraction
        for line in lines:
            line = line.strip()
            if len(line) > 20 and len(line) < 200 and not line.startswith('{'):
                if any(word in line.lower() for word in ['title:', 'paper:', 'article:']):
                    title = line.replace('title:', '').replace('paper:', '').replace('article:', '').strip()
                    break
                elif len(line) > 30 and '.' not in line[:30]:  # Likely a title
                    title = line
                    break
        
        # Try to extract authors
        for line in lines:
            if any(word in line.lower() for word in ['author', 'by ']):
                authors = line.strip()
                break
        
        return {
            'doc_id': doc_id,
            'title': title[:100] + "..." if len(title) > 100 else title,
            'authors': authors[:100] + "..." if len(authors) > 100 else authors,
            'paper_id': doc_id
        }
    
    def answer_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process query using enhanced basic RAG with passage tracking
        """
        logger.info(f"🔍 Processing query with Enhanced Basic RAG: {query}")
        
        # Initialize passage tracker
        passage_tracker = PassageTracker()
        
        # Step 1: Retrieve abstracts using hybrid retrieval
        logger.info(f"📄 Retrieving top-{self.top_k} abstracts via hybrid E5+BM25...")
        start_time = time.time()
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        retrieval_time = time.time() - start_time
        
        logger.info(f"✅ Retrieved {len(retrieved)} abstracts in {retrieval_time:.2f}s")
        
        if not retrieved:
            logger.warning("No documents retrieved!")
            return "I couldn't find any relevant documents to answer your question.", {
                "retrieved_docs": [],
                "passages_used": [],
                "document_metadata": [],
                "retrieval_time": retrieval_time,
                "total_passages": 0
            }
        
        # Step 2: Process documents into tracked passages
        logger.info("🔄 Processing documents into trackable passages...")
        all_passage_info = []
        document_metadata = []
        
        for item in retrieved:
            doc_id = item["id"]
            doc_text = item["abstract"]
            
            # Extract metadata
            metadata = self._extract_paper_metadata(doc_id, doc_text)
            document_metadata.append(metadata)
            
            # Split into passages with IDs
            passage_info = passage_tracker.add_document(doc_text, doc_id)
            all_passage_info.extend(passage_info)
            
            logger.debug(f"Document {doc_id}: {len(passage_info)} passages created")
        
        logger.info(f"📝 Created {len(all_passage_info)} trackable passages from {len(retrieved)} documents")
        
        # Step 3: Create prompt with passage tracking
        prompt = self._create_rag_prompt_with_passages(query, all_passage_info)
        
        # Step 4: Generate answer
        logger.info("🤖 Generating answer with passage tracking...")
        answer_start_time = time.time()
        raw_answer = self.agent.generate(prompt)
        answer_time = time.time() - answer_start_time
        
        logger.info(f"✅ Answer generated in {answer_time:.2f}s")
        
        # Step 5: Extract which passages were used
        logger.info("🔍 Analyzing which passages were used in the answer...")
        used_passages = passage_tracker.extract_used_passages(raw_answer)
        
        logger.info(f"📊 Found {len(used_passages)} passages referenced in answer")
        
        # Step 6: Clean up answer (remove technical formatting if needed)
        clean_answer = self._clean_answer(raw_answer)
        
        # Build comprehensive debug info
        debug_info = {
            "raw_answer": raw_answer,
            "clean_answer": clean_answer,
            "retrieved_docs": [(item["abstract"], item["id"]) for item in retrieved],
            "passages_used": used_passages,
            "document_metadata": document_metadata,
            "total_passages": len(all_passage_info),
            "passage_summary": passage_tracker.get_passage_summary(),
            "retrieval_time": retrieval_time,
            "answer_generation_time": answer_time,
            "prompt": prompt,
            "retrieval_method": "enhanced_hybrid_e5_bm25"
        }
        
        # Log usage statistics
        docs_with_passages_used = set(p['doc_id'] for p in used_passages)
        logger.info(f"📈 Usage stats: {len(used_passages)} passages from {len(docs_with_passages_used)} documents")
        
        return clean_answer, debug_info
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up the answer for better presentation"""
        # Remove any system prompts that might have leaked through
        cleaned = answer.strip()
        
        # Remove common prompt artifacts
        artifacts = [
            "Based on the passages provided:",
            "According to the passages:",
            "From the information given:",
        ]
        
        for artifact in artifacts:
            if cleaned.startswith(artifact):
                cleaned = cleaned[len(artifact):].strip()
        
        return cleaned
    
    def close(self):
        """Clean up resources"""
        try:
            self.retriever.close()
        except AttributeError:
            pass


def load_questions(file_path: str) -> List[Dict[str, Any]]:
    """Load questions from JSON or JSONL file"""
    is_jsonl = file_path.lower().endswith(".jsonl")
    
    try:
        questions = []
        
        if is_jsonl:
            logger.info(f"Loading questions from JSONL file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        question = json.loads(line)
                        if "id" not in question:
                            question["id"] = line_num
                        questions.append(question)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON at line {line_num}: {e}")
        else:
            logger.info(f"Loading questions from JSON file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    questions = data
                    for i, question in enumerate(questions):
                        if "id" not in question:
                            question["id"] = i + 1
                elif isinstance(data, dict):
                    if "questions" in data:
                        questions = data["questions"]
                    elif "question" in data:
                        questions = [data]
                    else:
                        questions = [data]
                        
        logger.info(f"Loaded {len(questions)} questions")
        return questions
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading questions: {e}")
        return []


def format_enhanced_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format result with enhanced passage information"""
    
    # Format passages used with detailed information
    formatted_passages = []
    for passage in result.get("passages_used", []):
        formatted_passages.append({
            "passage_id": passage["passage_id"],
            "text": passage["text"],
            "doc_id": passage["doc_id"],
            "usage_count": passage.get("usage_count", 1),
            "preview": passage["text"][:200] + "..." if len(passage["text"]) > 200 else passage["text"]
        })
    
    # Format document metadata
    formatted_metadata = []
    for doc_meta in result.get("document_metadata", []):
        formatted_metadata.append({
            "doc_id": doc_meta["doc_id"],
            "title": doc_meta["title"],
            "authors": doc_meta["authors"],
            "paper_id": doc_meta["paper_id"]
        })
    
    # Create comprehensive result
    formatted_result = {
        "id": result.get("id", 0),
        "question": result.get("question", ""),
        "answer": result.get("model_answer", ""),
        "passages_used": formatted_passages,
        "document_metadata": formatted_metadata,
        "retrieval_stats": {
            "documents_retrieved": len(result.get("retrieved_docs", [])),
            "total_passages_created": result.get("total_passages", 0),
            "passages_actually_used": len(result.get("passages_used", [])),
            "documents_with_passages_used": len(set(p["doc_id"] for p in result.get("passages_used", []))),
            "retrieval_method": result.get("retrieval_method", "enhanced_hybrid_e5_bm25")
        },
        "performance": {
            "retrieval_time": result.get("retrieval_time", 0),
            "answer_generation_time": result.get("answer_generation_time", 0),
            "total_processing_time": result.get("process_time", 0)
        }
    }
    
    return formatted_result


def write_results_to_jsonl(results: List[Dict[str, Any]], output_file: str):
    """Write enhanced results to JSONL file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            formatted_result = format_enhanced_result(result)
            f.write(json.dumps(formatted_result, ensure_ascii=False) + "\n")
    logger.info(f"Enhanced results written to {output_file}")


def write_result_to_json(result: Dict[str, Any], output_file: str):
    """Write single enhanced result to JSON file"""
    formatted_result = format_enhanced_result(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Enhanced result written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Basic RAG with Hybrid Retrieval and Passage Tracking")
    parser.add_argument(
        "--model",
        type=str,
        default="tiiuae/Falcon3-10B-Instruct",
        help="Model for LLM agent"
    )
    parser.add_argument(
        "--falcon_api_key",
        type=str,
        default=None,
        help="API key for Falcon API (only needed if using Falcon API)"
    )
    parser.add_argument(
        "--e5_index_dir", type=str, default="e5_faiss_index", 
        help="Directory containing FAISS E5 index"
    )
    parser.add_argument(
        "--bm25_index_dir", type=str, default="bm25_index", 
        help="Directory containing BM25 index"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, 
        help="Number of abstracts to retrieve (default: 5)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="quick_test_questions.jsonl",
        help="File containing questions (JSON or JSONL)"
    )
    parser.add_argument(
        "--single_question",
        type=str,
        default=None,
        help="Process a single question instead of the entire dataset"
    )
    parser.add_argument(
        "--output_format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format: 'json' for single file, 'jsonl' for line-delimited JSON"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", 
        help="Directory to save results"
    )
    args = parser.parse_args()
    
    # Initialize hybrid Retriever
    logger.info(f"🔍 Initializing hybrid Retriever...")
    logger.info(f"   E5 index: {args.e5_index_dir}")
    logger.info(f"   BM25 index: {args.bm25_index_dir}")
    
    retriever = Retriever(
        e5_index_directory=args.e5_index_dir,
        bm25_index_directory=args.bm25_index_dir,
        top_k=args.top_k
    )
    
    # Show BM25 status
    bm25_status = retriever.get_bm25_status()
    logger.info(f"🔧 BM25 Status: {bm25_status['method']} ({bm25_status['status']})")
    
    # Initialize Enhanced Basic RAG
    logger.info(f"🤖 Initializing Enhanced Basic RAG with top_k={args.top_k}...")
    rag = EnhancedBasicRAG(
        retriever=retriever, 
        agent_model=args.model, 
        top_k=args.top_k, 
        falcon_api_key=args.falcon_api_key
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process a single question if specified
    if args.single_question:
        logger.info(f"\n🔍 Processing single question: {args.single_question}")
        start_time = time.time()
        
        try:
            # Process the query
            answer, debug_info = rag.answer_query(args.single_question)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Create result object
            result = {
                "id": "single_question",
                "question": args.single_question,
                "model_answer": answer,
                "process_time": process_time,
                "retrieved_docs": debug_info["retrieved_docs"],
                "passages_used": debug_info["passages_used"],
                "document_metadata": debug_info["document_metadata"],
                "total_passages": debug_info["total_passages"],
                "retrieval_time": debug_info["retrieval_time"],
                "answer_generation_time": debug_info["answer_generation_time"],
                "retrieval_method": debug_info["retrieval_method"]
            }
            
            logger.info(f"✅ Answer: {answer}")
            logger.info(f"📊 Passages used: {len(debug_info['passages_used'])}")
            logger.info(f"⏱️  Total time: {process_time:.2f}s")
            logger.info(f"   └─ Retrieval: {debug_info['retrieval_time']:.2f}s")
            logger.info(f"   └─ Generation: {debug_info['answer_generation_time']:.2f}s")
            
            # Save result
            output_file = os.path.join(
                args.output_dir, f"enhanced_basic_rag_single_{timestamp}.json"
            )
            write_result_to_json(result, output_file)
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
        finally:
            rag.close()
            
        return
    
    # Load and process multiple questions
    questions = load_questions(args.data_file)
    if not questions:
        logger.error("No questions found. Exiting.")
        return
    
    # Process each question
    results = []
    
    for i, item in enumerate(questions):
        question_id = item.get("id", i + 1)
        question_text = item.get("question") or item.get("text")
        logger.info(f"\n🔍 Processing question {i+1}/{len(questions)}: {question_text}")
        start_time = time.time()
        
        try:
            # Process the query
            answer, debug_info = rag.answer_query(question_text)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Save result
            result = {
                "id": question_id,
                "question": question_text,
                "reference_answer": item.get("answer", ""),
                "model_answer": answer,
                "process_time": process_time,
                "retrieved_docs": debug_info["retrieved_docs"],
                "passages_used": debug_info["passages_used"],
                "document_metadata": debug_info["document_metadata"],
                "total_passages": debug_info["total_passages"],
                "retrieval_time": debug_info["retrieval_time"],
                "answer_generation_time": debug_info["answer_generation_time"],
                "retrieval_method": debug_info["retrieval_method"]
            }
            results.append(result)
            
            logger.info(f"✅ Answer: {answer[:100]}...")
            logger.info(f"📊 Passages used: {len(debug_info['passages_used'])}")
            logger.info(f"⏱️  Processing time: {process_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}", exc_info=True)
    
    # Clean up
    rag.close()
    
    # Save all results
    if results:
        random_num = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        if args.output_format == "jsonl":
            output_file = os.path.join(args.output_dir, f"enhanced_basic_rag_answers_{timestamp}_{random_num}.jsonl")
            write_results_to_jsonl(results, output_file)
        else:  # json
            for result in results:
                question_id = result["id"]
                output_file = os.path.join(
                    args.output_dir, f"enhanced_basic_rag_answer_{question_id}_{timestamp}.json"
                )
                write_result_to_json(result, output_file)
    
    logger.info(f"\n🎉 Processed {len(results)} questions successfully!")
    
    # Print summary statistics
    if results:
        avg_time = sum(r["process_time"] for r in results) / len(results)
        avg_retrieval_time = sum(r["retrieval_time"] for r in results) / len(results)
        avg_answer_time = sum(r["answer_generation_time"] for r in results) / len(results)
        avg_passages_used = sum(len(r["passages_used"]) for r in results) / len(results)
        avg_docs_retrieved = sum(len(r["retrieved_docs"]) for r in results) / len(results)
        
        logger.info(f"📊 Performance Summary:")
        logger.info(f"   Average total time: {avg_time:.2f}s")
        logger.info(f"   Average retrieval time: {avg_retrieval_time:.2f}s")
        logger.info(f"   Average answer time: {avg_answer_time:.2f}s")
        logger.info(f"   Average passages used: {avg_passages_used:.1f}")
        logger.info(f"   Average documents retrieved: {avg_docs_retrieved:.1f}")


if __name__ == "__main__":
    main()
