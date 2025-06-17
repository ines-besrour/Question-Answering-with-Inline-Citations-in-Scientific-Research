#!/usr/bin/env python3
"""
4-Agent RAG System with Question Splitting and Parallel Processing
- Agent 1: Question Splitter 
- Agent 2: Answer Generator from abstracts 
- Agent 3: Document Evaluator  
- Agent 4: Final Answer Generator with citations
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Import configuration
from config import E5_INDEX_DIR, BM25_INDEX_DIR, DB_PATH

# Logging setup
def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/4agent_rag_{timestamp}_{random_str}.log"

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("4Agent_RAG")

class QuestionSplitter:
    """Agent 1: Intelligent Question Splitting Agent"""
    
    def __init__(self, agent_model):
        self.agent = agent_model
        logger.info("Agent 1 (Question Splitter) initialized")
    
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

Now analyze this query:
Query: "{query}"

Respond with ONLY this format:
Split: YES/NO
Sub-questions: [list of questions] (empty list if Split: NO)"""

    def analyze_and_split(self, query: str) -> Tuple[bool, List[str]]:
        """Analyze query and split into sub-questions if beneficial"""
        logger.info(f"Agent 1: Analyzing query for splitting: {query}")
        
        # Simple heuristics first (fast check)
        if not self._quick_split_check(query):
            logger.info("Quick check: No splitting needed")
            return False, []
        
        # Use LLM for complex analysis
        prompt = self._create_splitting_prompt(query)
        response = self.agent.generate(prompt)
        
        # Parse response
        should_split, sub_questions = self._parse_splitting_response(response, query)
        
        if should_split:
            logger.info(f"Agent 1: Split into {len(sub_questions)} sub-questions: {sub_questions}")
        else:
            logger.info("Agent 1: No splitting recommended")
        
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
                        try:
                            if list_part.startswith('[') and list_part.endswith(']'):
                                sub_questions = json.loads(list_part)
                            else:
                                sub_questions = [q.strip().strip('"').strip("'") for q in list_part.split(',')]
                        except:
                            logger.warning(f"Failed to parse sub-questions: {list_part}")
                            sub_questions = []
            
            # Validation: ensure sub-questions are meaningful
            if should_split and sub_questions:
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

class PaperTitleExtractor:
    """Utility class for extracting paper titles from document text"""
    
    @staticmethod
    def extract_title_from_text(doc_text: str, doc_id: str) -> str:
        """Extract paper title from document text using multiple patterns"""
        try:
            # Handle LevelDB format: "Content for [paper_id]:\n[Title]"
            leveldb_pattern = r'Content for [^:]*:\s*\n([^\n]+)'
            match = re.search(leveldb_pattern, doc_text)
            if match:
                title_candidate = match.group(1).strip()
                if (len(title_candidate) > 10 and len(title_candidate) < 300 and
                    not title_candidate.lower().startswith(('abstract:', 'introduction:', 'the abstract', 'in this', 'we '))):
                    return title_candidate
            
            # Look for title in first few lines
            lines = doc_text.split('\n')
            for i, line in enumerate(lines[:5]):
                line = line.strip()
                
                if not line or line.lower().startswith(('content for', 'time taken', 'opening')):
                    continue
                
                if (len(line) > 10 and len(line) < 300 and 
                    not line.lower().startswith(('abstract:', 'introduction:', 'the abstract', 'in this', 'we ', 'this paper', '{')) and
                    not re.match(r'^\d+', line) and
                    not line.endswith(':') and
                    line.count(' ') >= 2):
                    return line
            
            # Fallback to first substantial line
            for line in lines[:5]:
                line = line.strip()
                if len(line) > 15 and len(line) < 200:
                    return line[:150] + "..." if len(line) > 150 else line
            
            return f"Document {doc_id}"
            
        except Exception as e:
            logger.debug(f"Error extracting title for {doc_id}: {e}")
            return f"Document {doc_id}"
    
    @staticmethod
    def format_title_for_log(title: str, max_length: int = 80) -> str:
        """Format title for logging with length limit"""
        if len(title) <= max_length:
            return title
        return title[:max_length-3] + "..."

class MultiPaperCitationHandler:
    """Citation handler with comprehensive multi-paper extraction"""
    
    def __init__(self, index_dir: str = "test_index", agent_model=None):
        self.doc_to_citation = {}
        self.citation_to_doc = {}
        self.next_citation_num = 1
        self.index_dir = Path(index_dir)
        self.agent_model = agent_model
        
        # Load arXiv papers for better metadata
        self.arxiv_papers = self._load_arxiv_papers()
        logger.info("Citation handler initialized")

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
            
            logger.debug(f"Loaded {len(papers)} arXiv papers for metadata")
            return papers
        except Exception as e:
            logger.debug(f"Error loading arXiv papers: {e}")
            return {}

    def _extract_paper_info(self, doc_text: str, doc_id: str, metadata: Dict = None) -> Dict:
        """Extract paper metadata"""
        paper_info = {
            'title': 'Unknown Title',
            'authors': 'Unknown',
            'venue': 'arXiv',
            'year': 'Unknown',
            'paper_id': doc_id
        }
        
        try:
            paper_info['title'] = PaperTitleExtractor.extract_title_from_text(doc_text, doc_id)
            
            # Extract from JSON in document text
            if '{' in doc_text and '"metadata"' in doc_text:
                try:
                    json_match = re.search(r'\{.*?"metadata".*?\}', doc_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        paper_data = json.loads(json_str)
                        
                        if 'metadata' in paper_data:
                            meta = paper_data['metadata']
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
                        
                except Exception as e:
                    logger.debug(f"JSON parsing failed for {doc_id}: {e}")
            
            # Match with loaded arXiv papers by paper_id
            if doc_id in self.arxiv_papers:
                arxiv_data = self.arxiv_papers[doc_id]
                if paper_info['title'] == 'Unknown Title' or paper_info['title'] == f"Document {doc_id}":
                    paper_info['title'] = arxiv_data['title']
                if paper_info['authors'] == 'Unknown':
                    paper_info['authors'] = arxiv_data['authors']
                if paper_info['year'] == 'Unknown':
                    paper_info['year'] = arxiv_data['year']
            
            # Final cleanup
            if len(paper_info['title']) > 150:
                paper_info['title'] = paper_info['title'][:150] + "..."
            
            if not paper_info['paper_id']:
                paper_info['paper_id'] = doc_id
                
        except Exception as e:
            logger.debug(f"Error extracting metadata for {doc_id}: {e}")
        
        return paper_info

    def _basic_text_cleaning(self, text: str) -> str:
        """Basic text cleaning for better model processing"""
        # Remove JSON-like section markers
        text = re.sub(r"'section':\s*'[^']*',\s*'text':\s*'", "", text)
        text = re.sub(r"^\s*\{.*?'text':\s*'", "", text)
        text = re.sub(r'\{[^}]*\}', '', text)  
        
        # Remove technical markup
        text = re.sub(r'\{\{[^}]+\}\}', '[REF]', text)
        text = re.sub(r'\$[^$]+\$', '[MATH]', text)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '[LATEX]', text)
        
        # Clean whitespace but preserve paragraph structure
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove extremely long lines that might be metadata
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if len(line) > 1000 and line.count(' ') < 10:
                continue
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        return result

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

    def _find_all_used_citations(self, answer_text: str) -> List[int]:
        """Find all citation numbers used in the answer"""
        citation_matches = re.findall(r'\[(\d+)\]', answer_text)
        used_citations = sorted(set(int(num) for num in citation_matches))
        return used_citations

    def _extract_claims_for_citation(self, answer_text: str, citation_num: int) -> List[str]:
        """Extract specific claims attributed to a particular citation"""
        
        sentences = re.split(r'[.!?]+', answer_text)
        
        claims = []
        citation_pattern = f'\\[{citation_num}\\]'
        
        for sentence in sentences:
            if re.search(citation_pattern, sentence):
                clean_sentence = re.sub(r'\[\d+\]', '', sentence).strip()
                if len(clean_sentence) > 10:
                    claims.append(clean_sentence)
        
        return claims

    def _create_comprehensive_extraction_prompt(self, answer_text: str, used_citations: List[int]) -> str:
        """Create a comprehensive prompt that analyzes ALL citations at once"""
        
        # Build the documents section with clear labeling
        documents_section = "RESEARCH PAPERS:\n"
        documents_section += "=" * 80 + "\n\n"
        
        for citation_num in used_citations:
            if citation_num in self.citation_to_doc:
                doc_info = self.citation_to_doc[citation_num]
                paper_info = doc_info['paper_info']
                
                # Clean and truncate document for better processing
                clean_content = self._basic_text_cleaning(doc_info['text'])
                if len(clean_content) > 8000:
                    clean_content = clean_content[:8000] + "... [content truncated]"
                
                documents_section += f"PAPER [{citation_num}]: \"{paper_info['title']}\"\n"
                documents_section += f"Authors: {paper_info['authors']}\n"
                documents_section += f"Year: {paper_info['year']}\n"
                documents_section += f"Content:\n{clean_content}\n\n"
                documents_section += "-" * 50 + "\n\n"
        
        # Analyze claims by citation
        claims_section = "CLAIMS TO ANALYZE:\n"
        claims_section += "=" * 40 + "\n"
        
        for citation_num in used_citations:
            claims = self._extract_claims_for_citation(answer_text, citation_num)
            if claims:
                claims_section += f"\nCitation [{citation_num}] supports these claims:\n"
                for i, claim in enumerate(claims, 1):
                    claims_section += f"  {i}. \"{claim}\"\n"
        
        # Create the comprehensive prompt
        prompt = f"""You are an expert citation analyzer. Your task is to identify the EXACT passages from research papers that support specific claims in an academic answer.

{documents_section}

{claims_section}

COMPLETE ANSWER WITH CITATIONS:
{answer_text}

TASK: For each citation [1], [2], [3], etc., identify the specific passages from the corresponding paper that directly support the claims attributed to that citation.

REQUIREMENTS:
1. Analyze ALL citations comprehensively in their full context
2. Extract exact passages that support each claim
3. Include 1-2 sentences before/after for context when helpful
4. Handle cases where claims synthesize information from multiple parts of a paper
5. Be precise - only extract what actually supports the claims
6. Maintain the relationship between citations and their source papers

RESPONSE FORMAT (follow this structure exactly):

Citation [1]:
Passage: "[exact text from Paper [1] that supports the claims for citation 1]"

Citation [2]: 
Passage: "[exact text from Paper [2] that supports the claims for citation 2]"

Citation [3]:
Passage: "[exact text from Paper [3] that supports the claims for citation 3]"

[Continue for all citations found...]

IMPORTANT NOTES:
- Each passage should be from the correct corresponding paper
- Include enough context for understanding but stay focused
- If no direct support found for a citation, state "No direct support found in the paper"
- Maintain academic precision in your extractions

Begin comprehensive analysis now:"""

        return prompt

    def _parse_comprehensive_response(self, model_response: str, used_citations: List[int]) -> Dict[int, str]:
        """Parse the model's comprehensive response into individual citation passages"""
        citation_passages = {}
        
        try:
            # Parse each citation block
            for citation_num in used_citations:
                # Primary pattern to match this specific citation's content
                primary_pattern = rf'Citation \[{citation_num}\]:\s*\n?\s*Passage:\s*"([^"]+)"'
                match = re.search(primary_pattern, model_response, re.DOTALL | re.IGNORECASE)
                
                if match:
                    passage = match.group(1).strip()
                    
                    # Clean up the passage
                    passage = re.sub(r'\s+', ' ', passage)
                    passage = re.sub(r'\n+', ' ', passage)
                    
                    if passage and not passage.endswith(('.', '!', '?')):
                        passage += '.'
                    
                    citation_passages[citation_num] = passage
                    logger.debug(f"Extracted passage for citation [{citation_num}]: {len(passage)} chars")
                
                else:
                    # Try alternative patterns if primary fails
                    alt_patterns = [
                        rf'\[{citation_num}\][:\s]*Passage[:\s]*"([^"]+)"',
                        rf'Citation {citation_num}[:\s]*"([^"]+)"',
                        rf'Paper \[{citation_num}\][:\s]*"([^"]+)"',
                        rf'\[{citation_num}\][:\s]*"([^"]+)"'
                    ]
                    
                    found = False
                    for alt_pattern in alt_patterns:
                        alt_match = re.search(alt_pattern, model_response, re.DOTALL | re.IGNORECASE)
                        if alt_match:
                            passage = alt_match.group(1).strip()
                            passage = re.sub(r'\s+', ' ', passage)
                            
                            if passage and not passage.endswith(('.', '!', '?')):
                                passage += '.'
                            
                            citation_passages[citation_num] = passage
                            logger.debug(f"Alternative extraction for citation [{citation_num}]: {len(passage)} chars")
                            found = True
                            break
                    
                    if not found:
                        logger.warning(f"Could not extract passage for citation [{citation_num}]")
                        citation_passages[citation_num] = self._get_fallback_passage(citation_num)
            
            return citation_passages
            
        except Exception as e:
            logger.error(f"Error parsing comprehensive response: {e}")
            return self._fallback_extraction_all(used_citations)

    def _get_fallback_passage(self, citation_num: int) -> str:
        """Fallback extraction for a specific citation when model extraction fails"""
        if citation_num in self.citation_to_doc:
            doc_text = self.citation_to_doc[citation_num]['text']
            clean_text = self._basic_text_cleaning(doc_text)
            sentences = re.split(r'[.!?]+', clean_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            # Return first few sentences as fallback
            fallback_sentences = sentences[:3] if len(sentences) >= 3 else sentences
            fallback = '. '.join(fallback_sentences)
            if fallback and not fallback.endswith(('.', '!', '?')):
                fallback += '.'
            
            return fallback
        
        return f"Content from citation [{citation_num}] (fallback extraction)."

    def _fallback_extraction_all(self, used_citations: List[int]) -> Dict[int, str]:
        """Fallback when comprehensive extraction completely fails"""
        fallback_results = {}
        for citation_num in used_citations:
            fallback_results[citation_num] = self._get_fallback_passage(citation_num)
        return fallback_results

    def extract_all_citations_comprehensive(self, answer_text: str) -> Dict[int, str]:
        """Analyze ALL citations in the answer simultaneously in a single model call"""
        try:
            if not self.agent_model or not self.citation_to_doc:
                logger.warning("No agent model or citations available for extraction")
                return {}
            
            # Find all citation numbers used in the answer
            used_citations = self._find_all_used_citations(answer_text)
            if not used_citations:
                logger.warning("No citations found in answer text")
                return {}
            
            logger.info(f"Starting comprehensive analysis of {len(used_citations)} citations: {used_citations}")
            
            # Create comprehensive extraction prompt
            extraction_prompt = self._create_comprehensive_extraction_prompt(answer_text, used_citations)
            
            # Single model call to analyze ALL citations
            logger.info("Performing comprehensive multi-paper citation analysis...")
            model_response = self.agent_model.generate(extraction_prompt)
            
            # Parse the structured response
            citation_passages = self._parse_comprehensive_response(model_response, used_citations)
            
            # Log results
            successful_extractions = sum(1 for passage in citation_passages.values() 
                                       if not passage.startswith(("Content from citation", "fallback")))
            
            logger.info(f"Comprehensive extraction completed:")
            logger.info(f"  Total citations: {len(used_citations)}")
            logger.info(f"  Successful extractions: {successful_extractions}/{len(used_citations)}")
            logger.info(f"  Model calls used: 1 (vs {len(used_citations)} for per-citation approach)")
            
            return citation_passages
            
        except Exception as e:
            logger.error(f"Error in comprehensive citation extraction: {e}")
            return self._fallback_extraction_all(used_citations)

    def format_references_comprehensive(self, answer_text: str = None) -> str:
        """Format references using comprehensive multi-paper extraction"""
        if not self.citation_to_doc:
            return ""
        
        if not answer_text:
            logger.warning("No answer text provided, using simple reference formatting")
            return self._format_references_simple()
        
        # Comprehensive extraction of ALL citations at once
        all_passages = self.extract_all_citations_comprehensive(answer_text)
        
        if not all_passages:
            logger.warning("Comprehensive extraction failed, using simple formatting")
            return self._format_references_simple()
        
        # Build references section
        references = "\n\n## References\n\n"
        total_chars = 0
        successful_extractions = 0
        
        for citation_num in sorted(all_passages.keys()):
            if citation_num not in self.citation_to_doc:
                continue
                
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
            
            # Add the comprehensively extracted passage
            passage = all_passages[citation_num]
            ref_line += f'\n    Passage: "{passage}"'
            
            references += ref_line + "\n\n"
            total_chars += len(passage)
            
            # Count successful extractions (not fallbacks)
            if not passage.startswith(("Content from citation", "fallback", "No direct support")):
                successful_extractions += 1
        
        # Log comprehensive extraction statistics
        total_citations = len(all_passages)
        success_rate = (successful_extractions / total_citations) * 100 if total_citations > 0 else 0
        avg_length = total_chars // total_citations if total_citations > 0 else 0
        
        logger.info(f"COMPREHENSIVE MULTI-PAPER EXTRACTION RESULTS:")
        logger.info(f"  Total citations processed: {total_citations}")
        logger.info(f"  Successful extractions: {successful_extractions}/{total_citations} ({success_rate:.1f}%)")
        logger.info(f"  Total passage characters: {total_chars:,}")
        logger.info(f"  Average passage length: {avg_length:,} chars")
        logger.info(f"  Model calls used: 1 (vs {total_citations} for per-citation approach)")
        
        return references

    def _format_references_simple(self) -> str:
        """Simple fallback reference formatting when comprehensive extraction is not available"""
        references = "\n\n## References\n\n"
        
        for citation_num, doc_info in self.citation_to_doc.items():
            paper_info = doc_info['paper_info']
            
            # Format basic reference
            ref_line = f"[{citation_num}] "
            
            if paper_info['authors'] != 'Unknown':
                ref_line += f"{paper_info['authors']}. "
            
            title = paper_info['title'].replace('"', "'")
            ref_line += f'"{title}." '
            
            if paper_info.get('paper_id') and paper_info['paper_id'] != 'Unknown':
                if str(paper_info['paper_id']).startswith('arXiv:'):
                    ref_line += f"{paper_info['paper_id']}"
                else:
                    ref_line += f"arXiv:{paper_info['paper_id']}"
            
            if paper_info['year'] != 'Unknown':
                ref_line += f" ({paper_info['year']})"
            
            # Simple fallback passage
            fallback_passage = self._get_fallback_passage(citation_num)
            ref_line += f'\n    Passage: "{fallback_passage}"'
            
            references += ref_line + "\n\n"
        
        logger.info(f"Used simple reference formatting for {len(self.citation_to_doc)} citations")
        return references

class FourAgentRAG:
    """4-Agent RAG System with Full-Text Context Strategy and Context Management"""
    
    def __init__(self, retriever, agent_model=None, n=0.0, falcon_api_key=None, index_dir="test_index", max_workers=4, max_context_chars=100000):
        """Initialize with 4-agent architecture and full-text context management"""
        
        self.retriever = retriever
        self.n = n
        self.index_dir = index_dir
        self.max_workers = max_workers
        self.max_context_chars = max_context_chars  # 32K tokens * 4 chars/token
        
        logger.info(f"Full-text context limit set to {max_context_chars} characters (~32K tokens)")
        
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
        
        # Pre-warming
        logger.info("4-agent pre-warming...")
        try:
            dummy_abstracts = self.retriever.retrieve_abstracts("test", top_k=1)
            logger.info("Retriever pre-warmed")
            
            if hasattr(self.agent1, 'generate'):
                self.agent1.generate("test")
                logger.info("All agents pre-warmed")
                
        except Exception as e:
            logger.warning(f"Pre-warming had issues: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 chars per token"""
        return len(text) // 4

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

    def _clean_full_text(self, doc_text: str, doc_id: str) -> str:
        """Clean full text for better context utilization"""
        try:
            # Remove "Content for [paper_id]:" line
            clean_text = re.sub(r'Content for [^:]*:\s*\n', '', doc_text)
            
            # Remove excessive JSON metadata at the beginning if present
            if clean_text.startswith('{') and '"paper_id"' in clean_text[:500]:
                json_end = clean_text.find('\n\n')
                if json_end > 0:
                    clean_text = clean_text[json_end:].strip()
            
            # Basic cleanup
            clean_text = clean_text.strip()
            
            return clean_text
            
        except Exception as e:
            logger.debug(f"Error cleaning full text for {doc_id}: {e}")
            return doc_text

    def _prepare_documents_for_agent4_fulltext(self, full_texts: List[Tuple[str, str]], citation_handler, was_split: bool = False, sub_question_groups: List[List[str]] = None) -> List[str]:
        """Prepare documents for Agent 4 with full-text strategy"""
        docs_with_citations = []
        
        if was_split and sub_question_groups and len(sub_question_groups) >= 2:
            # Sub-questions mode: 16K per group
            context_limit_per_group = self.max_context_chars // 2  # 16K tokens each
            logger.info(f"Sub-questions mode: Using {context_limit_per_group} chars per group (~16K tokens)")
            
            # Process each sub-question group
            for group_idx, doc_ids_group in enumerate(sub_question_groups[:2]):  # Limit to 2 groups
                group_docs = [(text, doc_id) for text, doc_id in full_texts if doc_id in doc_ids_group]
                
                logger.info(f"Processing sub-question group {group_idx + 1}: {len(group_docs)} documents")
                
                group_context_used = 0
                group_docs_added = 0
                
                # Add documents from this group until context limit
                for doc_text, doc_id in group_docs:
                    # Clean the full text
                    clean_text = self._clean_full_text(doc_text, doc_id)
                    
                    # Check if adding this document would exceed group limit
                    estimated_doc_size = len(clean_text) + 300  # +300 for formatting
                    
                    if group_context_used + estimated_doc_size > context_limit_per_group and group_docs_added > 0:
                        logger.info(f"Group {group_idx + 1} context limit reached. Using {group_docs_added} documents ({group_context_used:,} chars)")
                        break
                    
                    # Add document with citation
                    citation_num = citation_handler.add_document(clean_text, doc_id)
                    
                    # Get paper info for better document labeling
                    paper_info = citation_handler.citation_to_doc[citation_num]['paper_info']
                    doc_title = paper_info['title'][:80] + "..." if len(paper_info['title']) > 80 else paper_info['title']
                    
                    formatted_doc = f"Document [{citation_num}] (Group {group_idx + 1}) - \"{doc_title}\":\n{clean_text}"
                    docs_with_citations.append(formatted_doc)
                    
                    group_context_used += estimated_doc_size
                    group_docs_added += 1
            
            total_context = sum(len(doc.split('\n', 1)[1]) for doc in docs_with_citations)  # Exclude header line
            total_docs = len(docs_with_citations)
            
        else:
            # Single question mode: Use full 32K context
            logger.info(f"Single question mode: Using full {self.max_context_chars} chars (~32K tokens)")
            
            total_context_used = 0
            documents_used = 0
            
            # Add documents until context limit
            for doc_text, doc_id in full_texts:
                # Clean the full text
                clean_text = self._clean_full_text(doc_text, doc_id)
                
                # Check if adding this document would exceed context limit
                estimated_doc_size = len(clean_text) + 300  # +300 for formatting
                
                if total_context_used + estimated_doc_size > self.max_context_chars and documents_used > 0:
                    logger.info(f"Context limit reached. Using {documents_used} out of {len(full_texts)} documents")
                    break
                
                # Add document with citation
                citation_num = citation_handler.add_document(clean_text, doc_id)
                
                # Get paper info for better document labeling
                paper_info = citation_handler.citation_to_doc[citation_num]['paper_info']
                doc_title = paper_info['title'][:80] + "..." if len(paper_info['title']) > 80 else paper_info['title']
                
                formatted_doc = f"Document [{citation_num}] - \"{doc_title}\":\n{clean_text}"
                docs_with_citations.append(formatted_doc)
                
                total_context_used += estimated_doc_size
                documents_used += 1
            
            total_context = total_context_used
            total_docs = documents_used
        
        logger.info(f"Total context size: {total_context:,} chars (~{self._estimate_tokens(str(total_context))} tokens)")
        logger.info(f"Using {total_docs}/{len(full_texts)} documents for Agent 4")
        
        return docs_with_citations

    def _create_agent4_prompt_with_citations_fulltext(self, original_query, full_texts, citation_handler, was_split: bool = False, sub_question_groups: List[List[str]] = None):
        """Agent-4 prompt with full-text context strategy"""
        
        # Prepare documents with full-text strategy
        docs_with_citations = self._prepare_documents_for_agent4_fulltext(
            full_texts, citation_handler, was_split, sub_question_groups
        )
        
        docs_text = "\n\n" + "="*50 + "\n\n".join(docs_with_citations)
        
        # Count available citation numbers
        available_citations = [str(i) for i in range(1, len(docs_with_citations) + 1)]
        citation_examples = ", ".join(available_citations)
        
        context_info = "32K context" if not was_split else "16K per sub-question group"
        
        return f"""You are an accurate and reliable AI assistant. Answer questions based ONLY on the provided documents with proper academic citations.

CONTEXT STRATEGY: Full-text documents with {context_info}

STRICT CITATION REQUIREMENTS - YOU MUST FOLLOW THESE:
1. You MUST add [{citation_examples}] after EVERY claim you make
2. Every sentence that contains factual information MUST end with a citation
3. If you mention ANY concept, method, or fact, cite the document immediately
4. Use ONLY the document numbers shown: [{citation_examples}]
5. Do NOT write ANY sentence without a citation number
6. Use MULTIPLE different documents - don't just cite [1] repeatedly
7. Do NOT add a references section - it will be added automatically
8. EXAMPLE: "Machine learning involves pattern recognition [1]. Neural networks are a popular approach [2]. Deep learning has shown success in many domains [3]."

WRONG (no citations): "Machine learning is a powerful technique."
CORRECT (with citations): "Machine learning is a powerful technique for pattern recognition [1]."

WRONG (only one citation): "ML works by finding patterns [1]. It uses algorithms [1]. It requires data [1]."
CORRECT (multiple citations): "ML works by finding patterns [1]. It uses algorithms [2]. It requires data [3]."

Documents:
{docs_text}

Question: {original_query}

Remember: Use information from MULTIPLE documents and cite each one appropriately with [{citation_examples}]. Answer:"""

    def _reorder_full_texts_by_agent3_scores(self, full_texts: List[Tuple[str, str]], agent3_scores: Dict[str, float]) -> List[Tuple[str, str]]:
        """Reorder full texts by Agent 3 scores (highest scores first)"""
        def get_score(item):
            doc_text, doc_id = item
            return agent3_scores.get(doc_id, 0.0)
        
        # Sort by Agent 3 scores in descending order
        reordered = sorted(full_texts, key=get_score, reverse=True)

        logger.info("Reordered documents by Agent 3 scores:")
        for i, (doc_text, doc_id) in enumerate(reordered[:5]):  # Log top 5
            score = agent3_scores.get(doc_id, 0.0)
            title = PaperTitleExtractor.extract_title_from_text(doc_text, doc_id)
            formatted_title = PaperTitleExtractor.format_title_for_log(title, max_length=60)
            logger.info(f"  [{i+1}] Score: {score:.3f} - {formatted_title}")
        
        return reordered

    def _group_documents_by_subquestions(self, filtered_doc_ids_per_question: List[List[str]], all_filtered_doc_ids: List[str]) -> List[List[str]]:
        """Group document IDs by sub-questions for context allocation"""
        if not filtered_doc_ids_per_question or len(filtered_doc_ids_per_question) < 2:
            return [all_filtered_doc_ids]
        
        groups = []
        for sub_question_docs in filtered_doc_ids_per_question:
            # Keep documents that are in the filtered set
            group = [doc_id for doc_id in sub_question_docs if doc_id in all_filtered_doc_ids]
            groups.append(group)
        
        logger.info(f"Document groups: {[len(group) for group in groups]}")
        return groups

    def _log_retrieved_papers(self, query: str, retrieved_abstracts: List[Tuple], phase: str = "RETRIEVAL"):
        """Log the titles of retrieved papers"""
        if not retrieved_abstracts:
            logger.info(f"{phase}: No papers retrieved for query: {query[:50]}...")
            return
        
        logger.info(f"{phase}: Retrieved {len(retrieved_abstracts)} papers for query: {query[:50]}...")
        
        for i, (abstract_text, doc_id) in enumerate(retrieved_abstracts, 1):
            title = PaperTitleExtractor.extract_title_from_text(abstract_text, doc_id)
            formatted_title = PaperTitleExtractor.format_title_for_log(title, max_length=70)
            logger.info(f"  [{i:2d}] {formatted_title}")

    def _log_filtered_papers(self, query: str, filtered_abstracts: List[Tuple], scores: List[float]):
        """Log the titles of papers that passed Agent 3 filtering"""
        if not filtered_abstracts:
            logger.info(f"FILTERING: No papers passed Agent 3 filter for query: {query[:50]}...")
            return
        
        logger.info(f"FILTERING: {len(filtered_abstracts)} papers passed Agent 3 filter for query: {query[:50]}...")
        
        # Sort by score for display
        combined = list(zip(filtered_abstracts, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        
        for i, ((abstract_text, doc_id, _), score) in enumerate(combined, 1):
            title = PaperTitleExtractor.extract_title_from_text(abstract_text, doc_id)
            formatted_title = PaperTitleExtractor.format_title_for_log(title, max_length=65)
            logger.info(f"  ✓ [{i:2d}] {formatted_title} (score: {score:.3f})")

    def _process_single_question_with_scores(self, query: str, db=None) -> Tuple[List[Tuple], List, Dict[str, float]]:
        """Process a single question and return (abstracts, filtered_documents, agent3_scores)"""
        
        # PHASE 1: Retrieve ABSTRACTS for Agent2 & Agent3 filtering
        logger.info(f"Retrieving abstracts for: {query[:50]}...")
        retrieved_abstracts = self.retriever.retrieve_abstracts(query, top_k=5)
        
        # Log retrieved papers with titles
        self._log_retrieved_papers(query, retrieved_abstracts, "RETRIEVAL")
        
        # Step 2: Agent-2 generates answers from ABSTRACTS
        logger.info(f"Agent-2 generating answers from abstracts for: {query[:50]}...")
        doc_answers = []
        for abstract_text, doc_id in tqdm(retrieved_abstracts):
            prompt = self._create_agent2_prompt(query, abstract_text)
            answer = self.agent2.generate(prompt)
            doc_answers.append((abstract_text, doc_id, answer))
        
        # Step 3: Agent-3 evaluates documents using ABSTRACTS
        logger.info(f"Agent-3 evaluating abstracts for: {query[:50]}...")
        scores = []
        agent3_scores = {}
        
        for abstract_text, doc_id, answer in tqdm(doc_answers):
            prompt = self._create_agent3_prompt(query, abstract_text, answer)
            log_probs = self.agent3.get_log_probs(prompt, ["Yes", "No"])
            score = log_probs["Yes"] - log_probs["No"]
            scores.append(score)
            agent3_scores[doc_id] = score
        
        # Step 4: Calculate adaptive judge bar
        tau_q = np.mean(scores)
        sigma = np.std(scores)
        adjusted_tau_q = tau_q - self.n * sigma
        logger.info(f"Adaptive judge bar for '{query[:30]}...': tau_q={tau_q:.4f}, adjusted: {adjusted_tau_q:.4f}")
        
        # Step 5: Filter documents based on abstract evaluation
        filtered_doc_ids = []
        filtered_abstracts = []
        for i, (abstract_text, doc_id, _) in enumerate(doc_answers):
            if scores[i] >= adjusted_tau_q:
                filtered_doc_ids.append(doc_id)
                filtered_abstracts.append((abstract_text, doc_id, scores[i]))
        
        filtered_abstracts.sort(key=lambda x: x[2], reverse=True)
        
        # Log filtered papers with titles and scores
        self._log_filtered_papers(query, filtered_abstracts, [x[2] for x in filtered_abstracts])
        
        return retrieved_abstracts, filtered_doc_ids, agent3_scores

    def answer_query(self, query, db=None, choices=None):
        """Process query with full-text context strategy"""
        
        logger.info(f"Processing query with 4-agent approach (full-text strategy): {query}")
        
        # Initialize citation handler
        citation_handler = MultiPaperCitationHandler(self.index_dir, self.agent4)
        
        # PHASE 1: Agent-1 Question Splitting
        should_split, sub_questions = self.question_splitter.analyze_and_split(query)
        
        if should_split and sub_questions:
            logger.info(f"Processing {len(sub_questions)} sub-questions in parallel")
            questions_to_process = sub_questions
        else:
            logger.info("Processing single question")
            questions_to_process = [query]
        
        # PHASE 2: Parallel Processing of Questions with Score Tracking
        all_filtered_doc_ids = []
        all_agent3_scores = {}
        filtered_doc_ids_per_question = []
        
        if len(questions_to_process) > 1:
            # Parallel processing using thread pool
            logger.info(f"Processing {len(questions_to_process)} questions in parallel")
            
            # Submit all questions for parallel processing
            future_to_question = {}
            for sub_query in questions_to_process:
                future = self.executor.submit(self._process_single_question_with_scores, sub_query, db)
                future_to_question[future] = sub_query
            
            # Collect results
            for future in as_completed(future_to_question):
                sub_query = future_to_question[future]
                try:
                    retrieved_abstracts, filtered_doc_ids, agent3_scores = future.result()
                    all_filtered_doc_ids.extend(filtered_doc_ids)
                    all_agent3_scores.update(agent3_scores)
                    filtered_doc_ids_per_question.append(filtered_doc_ids)
                    logger.info(f"Completed processing: {sub_query[:50]}... -> {len(filtered_doc_ids)} docs")
                except Exception as e:
                    logger.error(f"Error processing sub-question '{sub_query}': {e}")
        else:
            # Single question processing
            retrieved_abstracts, filtered_doc_ids, agent3_scores = self._process_single_question_with_scores(questions_to_process[0], db)
            all_filtered_doc_ids = filtered_doc_ids
            all_agent3_scores = agent3_scores
            filtered_doc_ids_per_question = [filtered_doc_ids]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_filtered_doc_ids = []
        for doc_id in all_filtered_doc_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_filtered_doc_ids.append(doc_id)
        
        logger.info(f"Total unique filtered documents: {len(unique_filtered_doc_ids)}")
        
        # PHASE 3: Get FULL TEXTS and Reorder by Agent 3 Scores
        logger.info("Retrieving FULL texts for final answer generation...")
        if unique_filtered_doc_ids:
            full_texts = self.retriever.get_full_texts(unique_filtered_doc_ids, db=db)
            
            # Reorder by Agent 3 scores
            full_texts = self._reorder_full_texts_by_agent3_scores(full_texts, all_agent3_scores)
            
            logger.info(f"FINAL ANSWER GENERATION: Retrieved {len(full_texts)} papers (reordered by Agent 3 scores)")
            for i, (doc_text, doc_id) in enumerate(full_texts, 1):
                title = PaperTitleExtractor.extract_title_from_text(doc_text, doc_id)
                formatted_title = PaperTitleExtractor.format_title_for_log(title, max_length=70)
                char_count = len(doc_text)
                score = all_agent3_scores.get(doc_id, 0.0)
                logger.info(f"[{i:2d}] Score: {score:.3f} - {formatted_title} ({char_count:,} chars)")
            
        else:
            logger.warning("No documents passed the filter, using fallback")
            # Fallback to some documents from the original query
            fallback_abstracts, fallback_ids, fallback_scores = self._process_single_question_with_scores(query, db)
            full_texts = self.retriever.get_full_texts(fallback_ids[:3], db=db)
            all_agent3_scores.update(fallback_scores)
            if not full_texts:
                # Last resort: use abstracts
                full_texts = [(abstract_text, doc_id) for abstract_text, doc_id in fallback_abstracts[:3]]
    
        # PHASE 4: Agent-4 generates final answer with full-text context
        strategy_info = f"FULL-TEXT ({'split questions' if should_split else 'single question'})"
        logger.info(f"Agent-4 generating final answer with full-text strategy... [{strategy_info}]")
        
        # Group documents by sub-questions if needed
        sub_question_groups = None
        if should_split and len(filtered_doc_ids_per_question) >= 2:
            sub_question_groups = self._group_documents_by_subquestions(
                filtered_doc_ids_per_question, 
                unique_filtered_doc_ids
            )
        
        # Use the full-text preparation method
        prompt = self._create_agent4_prompt_with_citations_fulltext(
            query, full_texts, citation_handler, should_split, sub_question_groups
        )
        raw_answer = self.agent4.generate(prompt)
        
        # Generate references with comprehensive context passages
        references = citation_handler.format_references_comprehensive(raw_answer)
        
        # Remove any references Agent-4 might have added
        if "## References" in raw_answer:
            raw_answer = re.split(r'\n\s*## References', raw_answer)[0]
        
        # Combine answer with references
        cited_answer = raw_answer.strip() + references
        
        citation_map = citation_handler.get_citation_map()
        
        # Debug info
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
            "agent3_scores": all_agent3_scores,
            "context_stats": {
                "max_context_chars": self.max_context_chars,
                "total_chars_available": sum(len(text) for text, _ in full_texts),
                "docs_available": len(full_texts),
                "strategy": "FULL-TEXT",
                "context_allocation": "32K single" if not should_split else "16K per sub-question"
            }
        }
    
        return cited_answer, debug_info
    
    def _extract_passages_used(self, answer_text: str, citation_handler) -> List[Dict]:
        """Extract the specific passages used in the answer"""
        # Find all citations in the answer
        citation_matches = re.findall(r'\[(\d+)\]', answer_text)
        used_citations = set(int(num) for num in citation_matches)
        
        passages_used = []
        
        # Check if this is the MultiPaperCitationHandler
        if hasattr(citation_handler, 'extract_all_citations_comprehensive'):
            try:
                # Use comprehensive extraction results
                comprehensive_passages = citation_handler.extract_all_citations_comprehensive(answer_text)
                
                # Build passages_used from comprehensive results
                for citation_num in used_citations:
                    if citation_num in citation_handler.citation_to_doc:
                        doc_info = citation_handler.citation_to_doc[citation_num]
                        
                        # Get the comprehensively extracted passage
                        context_passage = comprehensive_passages.get(citation_num, "No passage extracted")
                        
                        passages_used.append({
                            "citation_num": citation_num,
                            "doc_id": doc_info['doc_id'],
                            "paper_title": doc_info['paper_info']['title'],
                            "paper_id": doc_info['paper_info']['paper_id'],
                            "authors": doc_info['paper_info']['authors'],
                            "year": doc_info['paper_info']['year'],
                            "context_passage": context_passage,
                            "passage_preview": context_passage[:200] + "..." if len(context_passage) > 200 else context_passage,
                            "extraction_method": "comprehensive"
                        })
                
                logger.debug(f"Extracted {len(passages_used)} passages using comprehensive method")
                return passages_used
                
            except Exception as e:
                logger.warning(f"Comprehensive extraction failed: {e}, using fallback")
        
        # Fallback for when comprehensive extraction fails
        for citation_num in used_citations:
            if citation_num in citation_handler.citation_to_doc:
                doc_info = citation_handler.citation_to_doc[citation_num]
                
                # Use fallback passage extraction
                try:
                    if hasattr(citation_handler, '_get_fallback_passage'):
                        context_passage = citation_handler._get_fallback_passage(citation_num)
                    else:
                        # Simple fallback
                        clean_text = citation_handler._basic_text_cleaning(doc_info['text'])
                        sentences = re.split(r'[.!?]+', clean_text)
                        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
                        context_passage = '. '.join(sentences[:2]) + '.'
                except Exception as e:
                    logger.warning(f"Error extracting passage for citation {citation_num}: {e}")
                    context_passage = f"Error extracting passage: {str(e)}"
                
                passages_used.append({
                    "citation_num": citation_num,
                    "doc_id": doc_info['doc_id'],
                    "paper_title": doc_info['paper_info']['title'],
                    "paper_id": doc_info['paper_info']['paper_id'],
                    "authors": doc_info['paper_info']['authors'],
                    "year": doc_info['paper_info']['year'],
                    "context_passage": context_passage,
                    "passage_preview": context_passage[:200] + "..." if len(context_passage) > 200 else context_passage,
                    "extraction_method": "fallback"
                })
        
        return passages_used
    
    def _extract_document_metadata(self, citation_handler):
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
        logger.info("4-Agent RAG system closed")


def initialize_retriever(retriever_type: str, e5_index_dir: str, bm25_index_dir: str, db_path: str, top_k: int, alpha: float = 0.65, db=None):
    """Initialize the retriever with strategy and alpha support"""
    logger.info(f"Initializing {retriever_type} retriever with alpha={alpha}...")
    return Retriever(e5_index_dir, bm25_index_dir, top_k=top_k, strategy=retriever_type, alpha=alpha)

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


def format_result_to_schema(result):
    """Format result with 4-agent information"""
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
        "document_metadata": result.get("document_metadata", {}),
        "agent3_scores": result.get("agent3_scores", {}),
        "context_strategy": result.get("context_stats", {}).get("strategy", "FULL-TEXT"),
        "context_allocation": result.get("context_stats", {}).get("context_allocation", "32K"),
        "total_chars_available": result.get("context_stats", {}).get("total_chars_available", 0)
    }
    
    return formatted_result


def write_results_to_jsonl(results, output_file):
    """Write results to JSONL file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            formatted_result = format_result_to_schema(result)
            f.write(json.dumps(formatted_result, ensure_ascii=False) + "\n")
    logger.info(f"Results written to {output_file}")


def write_result_to_json(result, output_file):
    """Write single result to JSON file"""
    formatted_result = format_result_to_schema(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Result written to {output_file}")


def main():
    """Main function with 4-agent support"""
    
    parser = argparse.ArgumentParser(description="4-Agent RAG with Question Splitting and Parallel Processing")
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
    parser.add_argument("--db_path", type=str, default=None, help="Path to LevelDB database (overrides config)")
    parser.add_argument("--alpha", type=float, default=0.65, help="Weight for E5 in hybrid mode (0.0=BM25 only, 1.0=E5 only)")
    parser.add_argument("--max_context_chars", type=int, default=100000, help="Maximum context characters (~32K tokens)")
    
    args = parser.parse_args()
    
    # Use custom DB path if provided, otherwise use config default
    db_path_to_use = args.db_path if args.db_path else DB_PATH

    logger.info(f"Opening database at {db_path_to_use}...")
    try:
        db = plyvel.DB(db_path_to_use, create_if_missing=False)
        logger.info("Database opened successfully")
    except Exception as e:
        logger.error(f"Failed to open database: {e}")
        # Try alternative path if permission denied
        alt_db_path = os.path.join(os.path.dirname(__file__), "local_db")
        logger.info(f"Trying alternative database path: {alt_db_path}")
        db = plyvel.DB(alt_db_path, create_if_missing=True)
        
    # NEW (BM25-only, no conflicts):
    from bm25_only_retriever import create_bm25_only_retriever

    retriever = create_bm25_only_retriever(
        BM25_INDEX_DIR,  # Only need BM25 index!
        args.top_k
    )
    
    logger.info(f"Initializing 4-agent RAG with n={args.n}, max_workers={args.max_workers}...")
    ragent = FourAgentRAG(
        retriever, 
        agent_model=args.model, 
        n=args.n, 
        index_dir=args.index_dir,
        max_workers=args.max_workers,
        max_context_chars=args.max_context_chars
    )
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "debug"), exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process single question
    if args.single_question:
        logger.info(f"\nProcessing single question with 4-agent system: {args.single_question}")
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
                debug_output_file = os.path.join(args.output_dir, "debug", f"4agent_single_{args.retriever_type}_debug_{timestamp}.json")
                with open(debug_output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Debug result saved to {debug_output_file}")
            else:
                output_file = os.path.join(args.output_dir, f"4agent_single_{args.retriever_type}_{timestamp}.json")
                write_result_to_json(result, output_file)
        
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
        logger.info(f"\nProcessing question {i+1}/{len(questions)} with 4-agent system: {item['question']}")
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
            debug_output_file = os.path.join(args.output_dir, "debug", f"4agent_question_{question_id}_{args.retriever_type}_debug_{timestamp}.json")
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
            output_file = os.path.join(args.output_dir, f"4agent_answers_{args.retriever_type}_{timestamp}_{random_num}.jsonl")
            write_results_to_jsonl(results, output_file)
        elif args.output_format == "json":
            for result in results:
                question_id = result["id"]
                output_file = os.path.join(args.output_dir, f"4agent_answer_{question_id}_{args.retriever_type}_{random_num}.json")
                write_result_to_json(result, output_file)
        else:  # debug
            output_file = os.path.join(args.output_dir, "debug", f"4agent_all_results_{args.retriever_type}_debug_{timestamp}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Debug results saved to {output_file}")
    
    logger.info(f"\nProcessed {len(results)} questions with 4-agent system.")
    
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
    logger.info("Database closed")
    

if __name__ == "__main__":
    main()
