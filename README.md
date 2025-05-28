# Question-Answering-with-Inline-Citations-in-Scientific-Research

This project is a research and development initiative at ScadsAI (TU Dresden) that aims to build an advanced question answering (QA) system with integrated citation generation. The system is designed to answer questions based on scientific literature while providing precise source references, ensuring transparency and reliability.

---

## Overview

## Data Format

The system works with academic papers stored in a JSONL (JSON Lines) format, where each line represents a single paper in JSON format. Below is a detailed explanation of the paper object structure and its components:

### Paper Object

Each paper is represented as a JSON object with the following keys:

- **`paper_id`**: The arXiv identifier of the paper (e.g., `"2105.05862"`).
- **`_pdf_hash`**: Always set to `None`.
- **`_source_hash`**: SHA1 hash of the arXiv source file.
- **`_source_name`**: Name of the arXiv source file (e.g., `"2105.05862.gz"`).
- **`metadata`**: Contains additional paper metadata sourced from [Kaggle's arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv).
- **`discipline`**: Scientific discipline of the paper (e.g., "Physics").
- **`abstract`**: The abstract text extracted from the metadata.
- **`body_text`**: A list representing the main content of the paper, divided into sections. Each element in the list contains:
  - **`section`**: Name of the section.
  - **`sec_number`**: The section number.
  - **`sec_type`**: Type of the section (e.g., section, subsection).
  - **`content_type`**: The type of content (e.g., paragraph, listing).
  - **`text`**: The actual textual content.
  - **`cite_spans`**: A list of citation markers within the text. Each marker contains:
    - `start`: Starting character offset.
    - `end`: Ending character offset.
    - `text`: The text displayed as the citation.
    - `ref_id`: A key linking to the corresponding bibliographic entry in `bib_entries`.
  - **`ref_spans`**: A list of referenced non-textual content (e.g., formulas, figures). Each reference contains:
    - `start`: Starting character offset.
    - `end`: Ending character offset.
    - `text`: The surface text.
    - `ref_id`: A key linking to the corresponding entry in `ref_entries`.

---

## System Architecture

   - **Identification of Sources**:  
     As the LLMA3 model produces the answer, it identifies key parts of the response that correspond to information from the source documents.

   - **Mapping References**:  
     When generating an answer, the system maps embedded citations and formula references to the detailed information in the paper object, ensuring users can trace back to the original sources.
