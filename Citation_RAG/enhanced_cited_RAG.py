#!/usr/bin/env python3
"""
Enhanced 4-Agent RAG System with Question Splitting and Parallel Processing
- Agent 1: Question Splitter (NEW)
- Agent 2: Answer Generator from abstracts (previously Agent 1)
- Agent 3: Document Evaluator (previously Agent 2)  
- Agent 4: Final Answer Generator with citations (previously Agent 3)
"""
import plyvel
import argparse
import json
import time
import datetime
import os
from tqdm import tqdm
import logging
import numpy as np
import random
import string
import re
from typing import List, Tuple, Dict, Any, Optional
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing as mp
from performance_monitor import monitor, time_block

# Import configuration
from config import E5_INDEX_DIR, BM25_INDEX_DIR, DB_PATH

# Your existing logging setup (unchanged)
def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/enhanced_4agent_rag_{timestamp}_{random_str}.log"

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("Enhanced_4Agent_RAG")

# Import the hybrid retriever components
from hybrid_retriever import Retriever

class QuestionSplitter:
    """
    Agent 1: NEW - Intelligent Question Splitting Agent
    Detects complex queries with multiple sub-questions and splits them appropriately
    """
    
    def __init__(self, agent_model):
        self.agent = agent_model
        logger.info("🔄 Agent 1 (Question Splitter) initialized")
    
    def _create_splitting_prompt(self, query: str) -> str:
        """Create prompt for question splitting analysis"""
        return f"""You are an intelligent question analyzer. Your task is to determine if a query contains multiple distinct sub-questions that would benefit from separate retrieval and research.

SPLITTING CRITERIA:
- Split if query contains multiple distinct topics connected by "and", "also", "what about"
- Split if query asks for comparisons between different concepts
- Split if query has multiple question words (what, how, why, when, where)
- DO NOT split simple clarifications or related aspects of the same topic

Examples:
Query: "What is quantum computing and how is it used in cryptography?"
Split: YES
Sub-questions: ["What is quantum computing?", "How is quantum computing used in cryptography?"]

Query: "What are the applications and limitations of machine learning?"
Split: NO (same topic, different aspects)
Sub-questions: []

Query: "How does BERT work and what is GPT-3?"
Split: YES  
Sub-questions: ["How does BERT work?", "What is GPT-3?"]

Query: "What is the difference between supervised and unsupervised learning?"
Split: NO (comparison of related concepts)
Sub-questions: []

Query: "What are neural networks and how do they learn and what are CNNs?"
Split: YES
Sub-questions: ["What are neural networks?", "How do neural networks learn?", "What are CNNs?"]

Now analyze this query:
Query: "{query}"

Respond with ONLY this format:
Split: YES/NO
Sub-questions: [list of questions] (empty list if Split: NO)"""

    def analyze_and_split(self, query: str) -> Tuple[bool, List[str]]:
        """
        Analyze query and split into sub-questions if beneficial
        
        Returns:
            Tuple of (should_split: bool, sub_questions: List[str])
        """
        with time_block("agent1_question_splitting"):
            logger.info(f"🧠 Agent 1: Analyzing query for splitting: {query}")
            
            # Simple heuristics first (fast check)
            if not self._quick_split_check(query):
                logger.info("📝 Quick check: No splitting needed")
                return False, []
            
            # Use LLM for complex analysis
            prompt = self._create_splitting_prompt(query)
            response = self.agent.generate(prompt)
            
            # Parse response
            should_split, sub_questions = self._parse_splitting_response(response, query)
            
            if should_split:
                logger.info(f"✂️ Agent 1: Split into {len(sub_questions)} sub-questions: {sub_questions}")
            else:
                logger.info("📝 Agent 1: No splitting recommended")
            
            return should_split, sub_questions
    
    def _quick_split_check(self, query: str) -> bool:
        """Fast heuristic check to avoid LLM calls for obvious cases"""
        query_lower = query.lower()
        
        # Skip very short queries
        if len(query.split()) < 6:
            return False
        
        # Look for splitting indicators
        split_indicators = [
            " and what ", " and how ", " and why ", " and when ", " and where ",
            "what about", "also what", "also how", "also why",
            "? and ", "? what", "? how", "? why", "? when", "? where"
        ]
        
        for indicator in split_indicators:
            if indicator in query_lower:
                return True
        
        # Count question words
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        question_count = sum(1 for word in question_words if word in query_lower)
        
        return question_count >= 2
    
    def _parse_splitting_response(self, response: str, original_query: str) -> Tuple[bool, List[str]]:
        """Parse the LLM response for splitting decision"""
        try:
            lines = response.strip().split('\n')
            should_split = False
            sub_questions = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('Split:'):
                    should_split = 'YES' in line.upper()
                elif line.startswith('Sub-questions:'):
                    # Extract list from the line
                    list_part = line.split(':', 1)[1].strip()
                    if list_part and list_part != '[]':
                        # Parse the list - handle both ["q1", "q2"] and simple comma-separated
                        try:
                            if list_part.startswith('[') and list_part.endswith(']'):
                                # JSON-like format
                                sub_questions = json.loads(list_part)
                            else:
                                # Comma-separated format
                                sub_questions = [q.strip().strip('"').strip("'") for q in list_part.split(',')]
                        except:
                            logger.warning(f"Failed to parse sub-questions: {list_part}")
                            sub_questions = []
            
            # Validation: ensure sub-questions are meaningful
            if should_split and sub_questions:
                # Filter out empty or too similar questions
                valid_questions = []
                for q in sub_questions:
                    q = q.strip()
                    if len(q) > 10 and q.endswith('?'):
                        valid_questions.append(q)
                
                if len(valid_questions) < 2:
                    logger.info("Not enough valid sub-questions, keeping original")
                    return False, []
                
                return True, valid_questions
            
            return False, []
            
        except Exception as e:
            logger.warning(f"Error parsing splitting response: {e}")
            return False, []


class EnhancedCitationHandler:
    """Enhanced citation handler with proper metadata extraction and context passages"""
    
    def __init__(self, index_dir: str = "test_index"):
        self.doc_to_citation = {}
        self.citation_to_doc = {}
        self.next_citation_num = 1
        self.index_dir = Path(index_dir)
        
        # Load arXiv papers for better metadata
        self.arxiv_papers = self._load_arxiv_papers()
        
        # Connect to metadata database
        self.metadata_db = self._connect_metadata_db()
    
    def _connect_metadata_db(self):
        """Connect to metadata database"""
        try:
            import sqlite3
            db_path = self.index_dir / "index_store.db"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            return conn
        except:
            return None
    
    def _load_arxiv_papers(self):
        """Load arXiv papers for metadata extraction"""
        papers = {}
        try:
            jsonl_files = list(self.index_dir.glob("*.jsonl"))
            
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        try:
                            paper = json.loads(line.strip())
                            paper_id = paper.get('paper_id', '')
                            
                            metadata = paper.get('metadata', {})
                            title = metadata.get('title', 'Unknown Title')
                            authors = metadata.get('authors', 'Unknown')
                            
                            # Extract year from versions
                            year = 'Unknown'
                            versions = paper.get('versions', [])
                            if versions:
                                created = versions[0].get('created', '')
                                year_match = re.search(r'(\d{4})', created)
                                if year_match:
                                    year = year_match.group(1)
                            
                            # Format authors properly
                            if 'authors_parsed' in paper:
                                authors_list = paper['authors_parsed']
                                if authors_list and len(authors_list) > 0:
                                    first_author = authors_list[0]
                                    if len(first_author) >= 2:
                                        formatted_author = f"{first_author[0]}, {first_author[1][0]}." if first_author[1] else first_author[0]
                                        if len(authors_list) > 1:
                                            authors = f"{formatted_author} et al."
                                        else:
                                            authors = formatted_author
                            
                            papers[paper_id] = {
                                'title': title,
                                'authors': authors,
                                'year': year,
                                'paper_id': paper_id,
                                'abstract': paper.get('abstract', {}).get('text', '')
                            }
                        except:
                            continue
            
            return papers
        except:
            return {}
    
    def _extract_paper_info(self, doc_text: str, doc_id: str, metadata: Dict = None) -> Dict:
        """Enhanced paper metadata extraction"""
        paper_info = {
            'title': 'Unknown Title',
            'authors': 'Unknown',
            'venue': 'arXiv',
            'year': 'Unknown',
            'paper_id': doc_id
        }
        
        try:
            # Extract from JSON in document text
            if '{' in doc_text and '"metadata"' in doc_text:
                try:
                    json_match = re.search(r'\{.*?"metadata".*?\}', doc_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        paper_data = json.loads(json_str)
                        
                        if 'metadata' in paper_data:
                            meta = paper_data['metadata']
                            if 'title' in meta:
                                paper_info['title'] = meta['title']
                            if 'authors' in meta:
                                paper_info['authors'] = meta['authors']
                        
                        if 'paper_id' in paper_data:
                            paper_info['paper_id'] = paper_data['paper_id']
                        
                        # Extract year from versions
                        if 'versions' in paper_data and paper_data['versions']:
                            created = paper_data['versions'][0].get('created', '')
                            year_match = re.search(r'(\d{4})', created)
                            if year_match:
                                paper_info['year'] = year_match.group(1)
                        
                        logger.debug(f"Extracted metadata from JSON in text for {doc_id}")
                        
                except Exception as e:
                    logger.debug(f"JSON parsing failed for {doc_id}: {e}")
            
            # Match with loaded arXiv papers by paper_id
            if doc_id in self.arxiv_papers:
                arxiv_data = self.arxiv_papers[doc_id]
                paper_info.update(arxiv_data)
                logger.debug(f"Found metadata for {doc_id} in arXiv papers database")
            
            # Final cleanup
            if len(paper_info['title']) > 150:
                paper_info['title'] = paper_info['title'][:150] + "..."
            
            # Ensure we have a paper_id
            if not paper_info['paper_id']:
                paper_info['paper_id'] = doc_id
                
        except Exception as e:
            logger.debug(f"Error extracting metadata for {doc_id}: {e}")
        
        return paper_info
    
    def _basic_text_cleaning(self, text: str) -> str:
        """Basic text cleaning for citation context"""
        # Remove JSON-like section markers
        text = re.sub(r"'section':\s*'[^']*',\s*'text':\s*'", "", text)
        text = re.sub(r"^\s*\{.*?'text':\s*'", "", text)
        text = re.sub(r'\{[^}]*\}', '', text)  
        
        # Remove technical markup
        text = re.sub(r'\{\{[^}]+\}\}', '[REF]', text)
        text = re.sub(r'\$[^$]+\$', '[MATH]', text)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '[LATEX]', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _extract_context_passage(self, answer_text: str, document_text: str, citation_num: int) -> str:
        """Extract specific sentence(s) used in the answer plus context"""
        try:
            # Clean the document text first
            try:
                from text_cleaner import DocumentTextCleaner
                cleaner = DocumentTextCleaner()
                clean_doc_text = cleaner.clean_for_citation_matching(document_text)
            except ImportError:
                clean_doc_text = self._basic_text_cleaning(document_text)
            
            # Find all sentences in the clean document
            sentences = re.split(r'[.!?]+', clean_doc_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
            
            # Look for content that appears in the answer near this citation
            citation_pattern = f'\\[{citation_num}\\]'
            citation_matches = list(re.finditer(citation_pattern, answer_text))
            
            if not citation_matches:
                # Fallback: return first few clean sentences
                return '. '.join(sentences[:2]) + '.' if sentences else clean_doc_text[:200] + "..."
            
            # For each citation, find the preceding text that likely came from this document
            relevant_sentences = set()
            
            for match in citation_matches:
                # Get text before this citation (up to 150 chars back)
                start_pos = max(0, match.start() - 150)
                context_text = answer_text[start_pos:match.start()].strip()
                
                # Find the sentence in context_text that likely came from the document
                context_sentences = re.split(r'[.!?]+', context_text)
                
                for context_sent in context_sentences[-2:]:  # Last 1-2 sentences before citation
                    if len(context_sent.strip()) < 15:
                        continue
                        
                    # Find similar sentences in the document
                    context_words = set(context_sent.lower().split())
                    
                    for i, doc_sent in enumerate(sentences):
                        doc_words = set(doc_sent.lower().split())
                        
                        # Check word overlap
                        overlap = len(context_words.intersection(doc_words))
                        overlap_ratio = overlap / max(len(context_words), 1)
                        
                        if overlap_ratio > 0.25 or overlap > 4:  # Good match
                            # Add this sentence plus context (±1 sentence)
                            start_idx = max(0, i - 1)
                            end_idx = min(len(sentences), i + 2)
                            
                            for j in range(start_idx, end_idx):
                                relevant_sentences.add(j)
            
            if relevant_sentences:
                # Sort and build context passage
                sorted_indices = sorted(relevant_sentences)
                context_parts = [sentences[i] for i in sorted_indices]
                result = '. '.join(context_parts) + '.'
                
                # Limit length
                if len(result) > 400:
                    result = result[:400] + "..."
                    
                return result
            
            # Fallback: return beginning of clean document
            fallback = '. '.join(sentences[:2]) + '.'
            return fallback if len(fallback) < 300 else fallback[:300] + "..."
            
        except Exception as e:
            logger.debug(f"Error extracting context passage: {e}")
            # Simple fallback with basic cleaning
            clean_text = self._basic_text_cleaning(document_text)
            return clean_text[:200] + "..." if len(clean_text) > 200 else clean_text
    
    def format_references(self, answer_text: str = None) -> str:
        """Format references with proper metadata and context passages"""
        if not self.citation_to_doc:
            return ""
        
        # Get all available citations
        citations_to_show = set(self.citation_to_doc.keys())
        
        # If answer text provided, filter to only used citations
        if answer_text:
            citation_matches = re.findall(r'\[(\d+)\]', answer_text)
            used_citations = set(int(num) for num in citation_matches)
            
            if used_citations:
                citations_to_show = used_citations.intersection(set(self.citation_to_doc.keys()))
        
        if not citations_to_show:
            return ""

        references = "\n\n## References\n\n"
        
        for citation_num in sorted(citations_to_show):
            doc_info = self.citation_to_doc[citation_num]
            paper_info = doc_info['paper_info']
            
            # Format academic reference
            ref_line = f"[{citation_num}] "
            
            # Add authors
            if paper_info['authors'] != 'Unknown':
                ref_line += f"{paper_info['authors']}. "
            
            # Add title in quotes
            title = paper_info['title'].replace('"', "'")
            ref_line += f'"{title}." '
            
            # Add venue and year with paper ID
            if paper_info.get('paper_id') and paper_info['paper_id'] != 'Unknown':
                if str(paper_info['paper_id']).startswith('arXiv:'):
                    ref_line += f"{paper_info['paper_id']}"
                else:
                    ref_line += f"arXiv:{paper_info['paper_id']}"
            else:
                ref_line += f"{paper_info['venue']}"
            
            if paper_info['year'] != 'Unknown':
                ref_line += f" ({paper_info['year']})"
            
            # Add context passage with actual sentences used
            if answer_text:
                context_passage = self._extract_context_passage(
                    answer_text, 
                    doc_info['text'], 
                    citation_num
                )
            else:
                context_passage = doc_info['text'][:300] + "..." if len(doc_info['text']) > 300 else doc_info['text']
            
            ref_line += f'\n    Passage: "{context_passage}"'
            
            references += ref_line + "\n\n"
        
        return references

    def add_document(self, doc_text: str, doc_id: str, metadata: Dict = None) -> int:
        """Add a document and return its citation number"""
        
        if doc_id not in self.doc_to_citation:
            citation_num = self.next_citation_num
            self.doc_to_citation[doc_id] = citation_num
            
            paper_info = self._extract_paper_info(doc_text, doc_id, metadata)
            
            self.citation_to_doc[citation_num] = {
                'doc_id': doc_id,
                'paper_info': paper_info,
                'text': doc_text
            }
            
            self.next_citation_num += 1
            logger.debug(f"Added document {doc_id} as citation [{citation_num}]: {paper_info['title'][:50]}...")
            return citation_num
        else:
            return self.doc_to_citation[doc_id]

    def get_citation_map(self) -> Dict[str, int]:
        """Get mapping from doc_id to citation number"""
        return self.doc_to_citation.copy()


class Enhanced4AgentRAG:
    """
    Enhanced 4-Agent RAG System with Question Splitting and Parallel Processing
    """
    
    def __init__(self, retriever, agent_model=None, n=0.0, falcon_api_key=None, index_dir="test_index", max_workers=4):
        """Initialize with enhanced 4-agent architecture"""
        
        self.retriever = retriever
        self.n = n
        self.index_dir = index_dir
        self.max_workers = max_workers
        
        # Initialize agents
        if isinstance(agent_model, str):
            if "falcon" in agent_model.lower() and falcon_api_key:
                from api_agent import FalconAgent
                self.agent1 = FalconAgent(falcon_api_key)  # Question Splitter
                self.agent2 = FalconAgent(falcon_api_key)  # Answer Generator
                self.agent3 = FalconAgent(falcon_api_key)  # Document Evaluator
                self.agent4 = FalconAgent(falcon_api_key)  # Final Answer Generator
                logger.info("Using Falcon agents with API for all four agent roles")
            else:
                from local_agent import LLMAgent
                self.agent1 = LLMAgent(agent_model)  # Question Splitter
                self.agent2 = LLMAgent(agent_model)  # Answer Generator
                self.agent3 = LLMAgent(agent_model)  # Document Evaluator
                self.agent4 = LLMAgent(agent_model)  # Final Answer Generator
                logger.info(f"Using local LLM agents with model {agent_model}")
        else:
            self.agent1 = agent_model  # Question Splitter
            self.agent2 = agent_model  # Answer Generator
            self.agent3 = agent_model  # Document Evaluator
            self.agent4 = agent_model  # Final Answer Generator
            logger.info("Using pre-initialized agent for all four agent roles")

        # Initialize question splitter
        self.question_splitter = QuestionSplitter(self.agent1)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Enhanced pre-warming
        logger.info("🔥 Enhanced 4-agent pre-warming...")
        try:
            # Warm up retriever
            dummy_abstracts = self.retriever.retrieve_abstracts("test", top_k=1)
            logger.info("✅ Retriever pre-warmed")
            
            # Warm up agents
            if hasattr(self.agent1, 'generate'):
                self.agent1.generate("test")
                logger.info("✅ All agents pre-warmed")
                
        except Exception as e:
            logger.warning(f"Pre-warming had issues: {e}")

    def _create_agent2_prompt(self, query, document):
        """Agent-2 prompt: Answer generation from abstracts"""
        return f"""You are an accurate and reliable AI assistant that can answer questions with the help of external documents. You should only provide the correct answer without repeating the question and instruction.

Document: {document}

Question: {query}

Answer:"""
    
    def _create_agent3_prompt(self, query, document, answer):
        """Agent-3 prompt: Document evaluation"""
        return f"""You are a noisy document evaluator that can judge if the external document is noisy for the query with unrelated or misleading information. Given a retrieved Document, a Question, and an Answer generated by an LLM (LLM Answer), you should judge whether both the following two conditions are reached: (1) the Document provides specific information for answering the Question; (2) the LLM Answer directly answers the question based on the retrieved Document. Please note that external documents may contain noisy or factually incorrect information. If the information in the document does not contain the answer, you should point it out with evidence. You should answer with "Yes" or "No" with evidence of your judgment, where "No" means one of the conditions (1) and (2) are unreached and indicates it is a noisy document.

Document: {document}

Question: {query}

LLM Answer: {answer}

Is this document relevant and supportive for answering the question?"""
    
    def _create_agent4_prompt_with_citations(self, original_query, filtered_documents, citation_handler):
        """Agent-4 prompt: Final answer generation with citations using ORIGINAL query"""
        
        def clean_text_basic(text):
            # Remove JSON-like section markers
            text = re.sub(r"'section':\s*'[^']*',\s*'text':\s*'", "", text)
            text = re.sub(r"^\s*\{.*?'text':\s*'", "", text)
            text = re.sub(r'\{[^}]*\}', '', text)
            
            # Remove technical markup
            text = re.sub(r'\{\{[^}]+\}\}', '[REF]', text)
            text = re.sub(r'\$[^$]+\$', '[MATH]', text)
            text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '[LATEX]', text)
            
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        # Build document text with citation numbers and clean formatting
        docs_with_citations = []
        
        for i, (doc_text, doc_id) in enumerate(filtered_documents):
            # Clean the document text first
            try:
                from text_cleaner import DocumentTextCleaner
                cleaner = DocumentTextCleaner()
                clean_text = cleaner.extract_clean_sentences(doc_text, max_sentences=10)
            except ImportError:
                # Fallback cleaning
                clean_text = clean_text_basic(doc_text)
                # Extract first few sentences
                sentences = re.split(r'[.!?]+', clean_text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:10]
                clean_text = '. '.join(sentences) + '.' if sentences else clean_text[:500]
            
            citation_num = citation_handler.add_document(clean_text, doc_id)
            
            # Get paper info for better document labeling
            paper_info = citation_handler.citation_to_doc[citation_num]['paper_info']
            doc_title = paper_info['title'][:80] + "..." if len(paper_info['title']) > 80 else paper_info['title']
            
            docs_with_citations.append(
                f"Document [{citation_num}] - \"{doc_title}\":\n{clean_text}"
            )
        
        docs_text = "\n\n" + "="*50 + "\n\n".join(docs_with_citations)
        
        return f"""You are an accurate and reliable AI assistant. Answer questions based ONLY on the provided documents with proper academic citations.

STRICT CITATION REQUIREMENTS - YOU MUST FOLLOW THESE:
1. You MUST add [1], [2], [3] etc. after EVERY claim you make
2. Every sentence that contains factual information MUST end with a citation
3. If you mention ANY concept, method, or fact, cite the document immediately
4. Use ONLY the document numbers shown: [1], [2], [3]
5. Do NOT write ANY sentence without a citation number
6. Do NOT add a references section - it will be added automatically
7. EXAMPLE: "Quantum walks can be controlled through coin operators [1]. The Hadamard coin produces different behavior than the Grover coin [2]."

WRONG (no citations): "Quantum walks are controlled by adjusting parameters."
CORRECT (with citations): "Quantum walks are controlled by adjusting parameters [1]."

Documents:
{docs_text}

Question: {original_query}

Remember: EVERY factual statement needs [1], [2], or [3] immediately after it. Answer:"""

    def _process_single_question(self, query: str, db=None) -> Tuple[List[Tuple], List]:
        """Process a single question and return (abstracts, filtered_documents)"""
        
        # PHASE 1: Retrieve ABSTRACTS for Agent2 & Agent3 filtering
        with time_block(f"retrieve_abstracts_{query[:20]}"):
            logger.info(f"📄 Retrieving abstracts for: {query[:50]}...")
            retrieved_abstracts = self.retriever.retrieve_abstracts(query, top_k=5)
            logger.info(f"Retrieved {len(retrieved_abstracts)} abstracts")
        
        # Step 2: Agent-2 generates answers from ABSTRACTS
        with time_block(f"agent2_generation_{query[:20]}"):
            logger.info(f"🤖 Agent-2 generating answers from abstracts for: {query[:50]}...")
            doc_answers = []
            for abstract_text, doc_id in tqdm(retrieved_abstracts):
                prompt = self._create_agent2_prompt(query, abstract_text)
                answer = self.agent2.generate(prompt)
                doc_answers.append((abstract_text, doc_id, answer))
        
        # Step 3: Agent-3 evaluates documents using ABSTRACTS
        with time_block(f"agent3_evaluation_{query[:20]}"):
            logger.info(f"⚖️ Agent-3 evaluating abstracts for: {query[:50]}...")
            scores = []
            for abstract_text, doc_id, answer in tqdm(doc_answers):
                prompt = self._create_agent3_prompt(query, abstract_text, answer)
                log_probs = self.agent3.get_log_probs(prompt, ["Yes", "No"])
                score = log_probs["Yes"] - log_probs["No"]
                scores.append(score)
        
        # Step 4: Calculate adaptive judge bar
        tau_q = np.mean(scores)
        sigma = np.std(scores)
        adjusted_tau_q = tau_q - self.n * sigma
        logger.info(f"📊 Adaptive judge bar for '{query[:30]}...': τq={tau_q:.4f}, adjusted: {adjusted_tau_q:.4f}")
        
        # Step 5: Filter documents based on abstract evaluation
        filtered_doc_ids = []
        filtered_abstracts = []
        for i, (abstract_text, doc_id, _) in enumerate(doc_answers):
            if scores[i] >= adjusted_tau_q:
                filtered_doc_ids.append(doc_id)
                filtered_abstracts.append((abstract_text, doc_id, scores[i]))
        
        filtered_abstracts.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"✅ Filtered to {len(filtered_doc_ids)} documents for: {query[:50]}...")
        
        return retrieved_abstracts, filtered_doc_ids

    def answer_query(self, query, db=None, choices=None):
        """
        ENHANCED: Process query with 4-agent approach and question splitting
        """
        
        with time_block("total_4agent_processing"):
            logger.info(f"🔍 Processing query with enhanced 4-agent approach: {query}")
            
            # Initialize enhanced citation handler
            citation_handler = EnhancedCitationHandler(self.index_dir)
            
            # PHASE 1: Agent-1 Question Splitting
            should_split, sub_questions = self.question_splitter.analyze_and_split(query)
            
            if should_split and sub_questions:
                logger.info(f"🔄 Processing {len(sub_questions)} sub-questions in parallel")
                questions_to_process = sub_questions
            else:
                logger.info("📝 Processing single question")
                questions_to_process = [query]
            
            # PHASE 2: Parallel Processing of Questions
            all_filtered_doc_ids = []
            
            if len(questions_to_process) > 1:
                # Parallel processing using thread pool
                with time_block("parallel_question_processing"):
                    logger.info(f"🚀 Processing {len(questions_to_process)} questions in parallel")
                    
                    # Submit all questions for parallel processing
                    future_to_question = {}
                    for sub_query in questions_to_process:
                        future = self.executor.submit(self._process_single_question, sub_query, db)
                        future_to_question[future] = sub_query
                    
                    # Collect results
                    for future in as_completed(future_to_question):
                        sub_query = future_to_question[future]
                        try:
                            retrieved_abstracts, filtered_doc_ids = future.result()
                            all_filtered_doc_ids.extend(filtered_doc_ids)
                            logger.info(f"✅ Completed processing: {sub_query[:50]}... -> {len(filtered_doc_ids)} docs")
                        except Exception as e:
                            logger.error(f"Error processing sub-question '{sub_query}': {e}")
            else:
                # Single question processing
                retrieved_abstracts, filtered_doc_ids = self._process_single_question(questions_to_process[0], db)
                all_filtered_doc_ids = filtered_doc_ids
            
            # Remove duplicates while preserving order
            seen = set()
            unique_filtered_doc_ids = []
            for doc_id in all_filtered_doc_ids:
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_filtered_doc_ids.append(doc_id)
            
            logger.info(f"📚 Total unique filtered documents: {len(unique_filtered_doc_ids)}")
            
            # PHASE 3: Get FULL TEXTS for Agent4
            with time_block("get_full_texts"):
                logger.info("📚 Retrieving FULL texts for final answer generation...")
                if unique_filtered_doc_ids:
                    full_texts = self.retriever.get_full_texts(unique_filtered_doc_ids, db=db)
                else:
                    logger.warn("⚠️ No documents passed the filter, using fallback")
                    # Fallback to some documents from the original query
                    fallback_abstracts, fallback_ids = self._process_single_question(query, db)
                    full_texts = self.retriever.get_full_texts(fallback_ids[:3], db=db)
                    if not full_texts:
                        # Last resort: use abstracts
                        full_texts = [(abstract_text, doc_id) for abstract_text, doc_id in fallback_abstracts[:3]]
        
            # PHASE 4: Agent-4 generates final answer using ORIGINAL query
            with time_block("agent4_generation"):
                logger.info("🎯 Agent-4 generating final answer with citations using ORIGINAL query...")
                prompt = self._create_agent4_prompt_with_citations(query, full_texts, citation_handler)  # Use original query!
                raw_answer = self.agent4.generate(prompt)
            
            # Generate references with enhanced context passages
            references = citation_handler.format_references(raw_answer)
            
            # Remove any references Agent-4 might have added
            if "## References" in raw_answer:
                raw_answer = re.split(r'\n\s*## References', raw_answer)[0]
            
            # Combine answer with references
            cited_answer = raw_answer.strip() + references
            
            citation_map = citation_handler.get_citation_map()
            
            # Enhanced debug info
            debug_info = {
                "original_query": query,
                "was_split": should_split,
                "sub_questions": sub_questions if should_split else [],
                "questions_processed": len(questions_to_process),
                "total_filtered_docs": len(unique_filtered_doc_ids),
                "full_texts_retrieved": len(full_texts),
                "total_citations": len(citation_map),
                "citation_map": citation_map,
                "passages_used": self._extract_passages_used(raw_answer, citation_handler),
                "document_metadata": self._extract_document_metadata(citation_handler),
                "performance_stats": monitor.get_stats() if hasattr(monitor, 'get_stats') else {}
            }
        
        return cited_answer, debug_info
    
    def _extract_passages_used(self, answer_text: str, citation_handler: EnhancedCitationHandler):
        """Extract the specific passages used in the answer"""
        # Find all citations in the answer
        citation_matches = re.findall(r'\[(\d+)\]', answer_text)
        used_citations = set(int(num) for num in citation_matches)
        
        passages_used = []
        for citation_num in used_citations:
            if citation_num in citation_handler.citation_to_doc:
                doc_info = citation_handler.citation_to_doc[citation_num]
                
                # Extract context passage for this citation
                context_passage = citation_handler._extract_context_passage(
                    answer_text, doc_info['text'], citation_num
                )
                
                passages_used.append({
                    "citation_num": citation_num,
                    "doc_id": doc_info['doc_id'],
                    "paper_title": doc_info['paper_info']['title'],
                    "paper_id": doc_info['paper_info']['paper_id'],
                    "authors": doc_info['paper_info']['authors'],
                    "year": doc_info['paper_info']['year'],
                    "context_passage": context_passage,
                    "passage_preview": context_passage[:200] + "..." if len(context_passage) > 200 else context_passage
                })
        
        return passages_used
    
    def _extract_document_metadata(self, citation_handler: EnhancedCitationHandler):
        """Extract document metadata for all citations"""
        metadata = {}
        for citation_num, doc_info in citation_handler.citation_to_doc.items():
            metadata[citation_num] = {
                "doc_id": doc_info['doc_id'],
                "paper_info": doc_info['paper_info']
            }
        return metadata

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("Enhanced 4-Agent RAG system closed")


def initialize_retriever(retriever_type: str, e5_index_dir: str, bm25_index_dir: str, db_path: str, top_k: int, db=None):
    """Initialize the retriever"""
    logger.info(f"🔍 Initializing {retriever_type} retriever...")
    return Retriever(e5_index_dir, bm25_index_dir, top_k=top_k)


# Your existing utility functions (unchanged)
def load_datamorgana_questions(file_path):
    """Load questions from file"""
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


def format_enhanced_result_to_schema(result):
    """Format result with enhanced 4-agent information"""
    formatted_result = {
        "id": result.get("id", 0),
        "question": result.get("question", ""),
        "answer": result.get("model_answer", ""),
        "was_split": result.get("was_split", False),
        "sub_questions": result.get("sub_questions", []),
        "questions_processed": result.get("questions_processed", 1),
        "citation_count": result.get("total_citations", 0),
        "total_filtered_docs": result.get("total_filtered_docs", 0),
        "full_texts_used": result.get("full_texts_retrieved", 0),
        "processing_time": result.get("process_time", 0),
        "retriever_type": result.get("retriever_type", "hybrid"),
        "passages_used": result.get("passages_used", []),
        "document_metadata": result.get("document_metadata", {})
    }
    
    return formatted_result


def write_enhanced_results_to_jsonl(results, output_file):
    """Write enhanced results to JSONL file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            formatted_result = format_enhanced_result_to_schema(result)
            f.write(json.dumps(formatted_result, ensure_ascii=False) + "\n")
    logger.info(f"Enhanced results written to {output_file}")


def write_enhanced_result_to_json(result, output_file):
    """Write single enhanced result to JSON file"""
    formatted_result = format_enhanced_result_to_schema(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Enhanced result written to {output_file}")


def main():
    """Main function with enhanced 4-agent support"""
    
    parser = argparse.ArgumentParser(description="Enhanced 4-Agent RAG with Question Splitting and Parallel Processing")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon3-10B-Instruct", help="Model for LLM agents")
    parser.add_argument("--n", type=float, default=0.5, help="Adjustment factor for adaptive judge bar")
    parser.add_argument("--retriever_type", choices=["e5", "bm25", "hybrid"], default="hybrid", 
                        help="Type of retriever")
    parser.add_argument("--index_dir", type=str, default="test_index", help="Directory containing metadata")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--data_file", type=str, default="quick_test_questions.jsonl", help="File containing questions")
    parser.add_argument("--single_question", type=str, default=None, help="Process a single question")
    parser.add_argument("--output_format", choices=["json", "jsonl", "debug"], default="jsonl", help="Output format")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    args = parser.parse_args()
    
    # Open the database
    logger.info(f"Opening database at {DB_PATH}...")
    try:
        db = plyvel.DB(DB_PATH, create_if_missing=False)
        logger.info("✅ Database opened successfully")
    except Exception as e:
        logger.error(f"Failed to open database: {e}")
        # Try alternative path if permission denied
        alt_db_path = os.path.join(os.path.dirname(__file__), "local_db")
        logger.info(f"Trying alternative database path: {alt_db_path}")
        db = plyvel.DB(alt_db_path, create_if_missing=True)
        
    retriever = initialize_retriever(
        args.retriever_type, 
        E5_INDEX_DIR, 
        BM25_INDEX_DIR, 
        DB_PATH, 
        args.top_k
    )
    
    logger.info(f"Initializing enhanced 4-agent RAG with n={args.n}, max_workers={args.max_workers}...")
    ragent = Enhanced4AgentRAG(
        retriever, 
        agent_model=args.model, 
        n=args.n, 
        index_dir=args.index_dir,
        max_workers=args.max_workers
    )
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "debug"), exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process single question
    if args.single_question:
        logger.info(f"\nProcessing single question with enhanced 4-agent system: {args.single_question}")
        start_time = time.time()
        
        try:
            cited_answer, debug_info = ragent.answer_query(args.single_question, db) 
            process_time = time.time() - start_time
            
            result = {
                "id": f"single_question_4agent_{args.retriever_type}",
                "question": args.single_question,
                "model_answer": cited_answer,
                "was_split": debug_info["was_split"],
                "sub_questions": debug_info["sub_questions"],
                "questions_processed": debug_info["questions_processed"],
                "total_citations": debug_info["total_citations"],
                "total_filtered_docs": debug_info["total_filtered_docs"],
                "full_texts_retrieved": debug_info["full_texts_retrieved"],
                "passages_used": debug_info["passages_used"],
                "document_metadata": debug_info["document_metadata"],
                "process_time": process_time,
                "retriever_type": args.retriever_type
            }
            
            logger.info(f"Cited Answer: {cited_answer}")
            logger.info(f"Was Split: {debug_info['was_split']}")
            if debug_info['was_split']:
                logger.info(f"Sub-questions: {debug_info['sub_questions']}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(f"Citations used: {debug_info['total_citations']}")
            
            # Save result
            if args.output_format == "debug":
                debug_output_file = os.path.join(args.output_dir, "debug", f"enhanced_4agent_single_{args.retriever_type}_debug_{timestamp}.json")
                with open(debug_output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Debug result saved to {debug_output_file}")
            else:
                output_file = os.path.join(args.output_dir, f"enhanced_4agent_single_{args.retriever_type}_{timestamp}.json")
                write_enhanced_result_to_json(result, output_file)
        
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
        finally:
            ragent.close()
            retriever.close()
        
        return
    
    # Process question file
    questions = load_datamorgana_questions(args.data_file)
    if not questions:
        logger.error("No questions found. Exiting.")
        return
    
    results = []
    
    for i, item in enumerate(questions):
        question_id = item.get("id", i + 1)
        logger.info(f"\nProcessing question {i+1}/{len(questions)} with enhanced 4-agent system: {item['question']}")
        start_time = time.time()
        
        try:
            cited_answer, debug_info = ragent.answer_query(item["question"], db) 
            process_time = time.time() - start_time
            
            result = {
                "id": question_id,
                "question": item["question"],
                "model_answer": cited_answer,
                "was_split": debug_info["was_split"],
                "sub_questions": debug_info["sub_questions"],
                "questions_processed": debug_info["questions_processed"],
                "total_citations": debug_info["total_citations"],
                "total_filtered_docs": debug_info["total_filtered_docs"],
                "full_texts_retrieved": debug_info["full_texts_retrieved"],
                "passages_used": debug_info["passages_used"],
                "document_metadata": debug_info["document_metadata"],
                "process_time": process_time,
                "retriever_type": args.retriever_type
            }
            results.append(result)
            
            logger.info(f"Cited Answer: {cited_answer[:200]}...")
            logger.info(f"Was Split: {debug_info['was_split']}")
            if debug_info['was_split']:
                logger.info(f"Sub-questions: {debug_info['sub_questions']}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(f"Citations used: {debug_info['total_citations']}")
            
            # Save debug info
            debug_output_file = os.path.join(args.output_dir, "debug", f"enhanced_4agent_question_{question_id}_{args.retriever_type}_debug_{timestamp}.json")
            with open(debug_output_file, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}", exc_info=True)
    
    # Clean up
    ragent.close()
    retriever.close()
    
    # Save all results
    random_num = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    
    if results:
        if args.output_format == "jsonl":
            output_file = os.path.join(args.output_dir, f"enhanced_4agent_answers_{args.retriever_type}_{timestamp}_{random_num}.jsonl")
            write_enhanced_results_to_jsonl(results, output_file)
        elif args.output_format == "json":
            for result in results:
                question_id = result["id"]
                output_file = os.path.join(args.output_dir, f"enhanced_4agent_answer_{question_id}_{args.retriever_type}_{random_num}.json")
                write_enhanced_result_to_json(result, output_file)
        else:  # debug
            output_file = os.path.join(args.output_dir, "debug", f"enhanced_4agent_all_results_{args.retriever_type}_debug_{timestamp}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Debug results saved to {output_file}")
    
    logger.info(f"\nProcessed {len(results)} questions with enhanced 4-agent system.")
    
    if results:
        avg_time = sum(r["process_time"] for r in results) / len(results)
        avg_filtered = sum(r["total_filtered_docs"] for r in results) / len(results)
        avg_citations = sum(r["total_citations"] for r in results) / len(results)
        avg_full_texts = sum(r["full_texts_retrieved"] for r in results) / len(results)
        split_count = sum(1 for r in results if r["was_split"])
        
        logger.info(f"Average processing time: {avg_time:.2f} seconds")
        logger.info(f"Questions split: {split_count}/{len(results)} ({split_count/len(results)*100:.1f}%)")
        logger.info(f"Average filtered documents: {avg_filtered:.1f}")
        logger.info(f"Average citations: {avg_citations:.1f}")
        logger.info(f"Average full texts used: {avg_full_texts:.1f}")
    
    db.close()
    logger.info("✅ Database closed")
    

if __name__ == "__main__":
    main()