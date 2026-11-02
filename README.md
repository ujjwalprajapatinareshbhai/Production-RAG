# Production RAG System

A production-oriented Retrieval-Augmented Generation (RAG) system designed to process PDF documents and retrieve relevant information using hybrid search, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking.

The project focuses on building a reliable document ingestion and retrieval pipeline with document lifecycle management, recovery mechanisms, and retrieval evaluation.

---

## Overview

This project implements a RAG pipeline that combines semantic and keyword-based retrieval to improve the relevance of retrieved document content.

### Key Features

- PDF document loading and text extraction
- Text cleaning and preprocessing
- Fixed-size, page-aware, and section-aware chunking
- Dense vector embeddings
- Sparse BM25 retrieval
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- Context selection
- Answer generation using Ollama
- Qdrant vector database
- PostgreSQL document registry
- Document lifecycle management
- Recovery and retry mechanisms for failed cleanup operations
- Evaluation scripts for comparing retrieval and chunking strategies
- Unit and integration testing

---

## Architecture

```text
                       PDF Documents
                             |
                             v
                       PDF Loader
                             |
                             v
                       Text Cleaning
                             |
                             v
                     Document Chunking
                             |
                             v
                    Embedding Generation
                             |
                  +----------+----------+
                  |                     |
                  v                     v
             Dense Vectors        Sparse Vectors
                  |                     |
                  v                     v
               Qdrant              BM25 Search
                  |                     |
                  +----------+----------+
                             |
                             v
                      Hybrid Retrieval
                             |
                             v
                 Reciprocal Rank Fusion
                             |
                             v
                     Cross-Encoder
                       Reranking
                             |
                             v
                     Context Selection
                             |
                             v
                     Answer Generation
                             |
                             v
                       Final Answer
```

### Supporting Infrastructure

- **Qdrant:** Stores document vectors.
- **PostgreSQL:** Maintains document metadata, ingestion status, and cleanup state.
- **Ollama:** Supports LLM-based answer generation.
- **FastAPI:** Exposes application functionality through API endpoints.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| API Framework | FastAPI |
| Vector Database | Qdrant |
| Relational Database | PostgreSQL |
| Dense Embeddings | Sentence Transformers |
| Sparse Retrieval | BM25 / Qdrant BM25 |
| Hybrid Ranking | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| PDF Processing | PyMuPDF |
| Configuration | Pydantic Settings |
| LLM Runtime | Ollama |
| Testing | Pytest |
| Containerization | Docker Compose |

---

## Project Structure

```text
Production-RAG/
│
├── app/
│   ├── answer.py
│   ├── api.py
│   ├── bm25_retriever.py
│   ├── chunker.py
│   ├── config.py
│   ├── context_selector.py
│   ├── db.py
│   ├── document_management.py
│   ├── embeddings.py
│   ├── hybrid_retriever.py
│   ├── ingestion.py
│   ├── migrate_sparse_vectors.py
│   ├── models.py
│   ├── pdf_loader.py
│   ├── qdrant_hybrid_retriever.py
│   ├── recovery.py
│   ├── registry.py
│   ├── reranker.py
│   ├── retriever.py
│   ├── search.py
│   ├── text_cleaner.py
│   └── vector_store.py
│
├── scripts/
│   ├── add_abstract_to_eval_collections.py
│   ├── compare_chunking_evaluation.py
│   ├── compare_chunking_retrieval.py
│   ├── compare_chunking_strategies.py
│   ├── compare_reranking_strategies.py
│   ├── compare_retrieval_strategies.py
│   ├── diagnose_abstract_retrieval.py
│   ├── diagnose_broad_improvements.py
│   ├── diagnose_dense_sparse.py
│   ├── diagnose_missed_questions.py
│   ├── diagnose_query_variants.py
│   ├── evaluate_abstract_boost.py
│   ├── evaluate_abstract_chunk.py
│   ├── evaluate_abstract_context.py
│   ├── evaluate_answer_quality.py
│   ├── evaluate_chunking.py
│   ├── evaluate_context_selection.py
│   ├── evaluate_intent_aware_abstract.py
│   ├── evaluate_multi_query.py
│   ├── evaluate_section_aware_chunking.py
│   ├── inspect_broad_improvements.py
│   ├── inspect_q2_evidence.py
│   ├── label_eval_abstract.py
│   ├── test_context_selection.py
│   ├── test_context_selection_real.py
│   └── test_context_selector.py
│
├── tests/
│   ├── conftest.py
│   ├── test_answer.py
│   ├── test_api.py
│   ├── test_bm25_retriever.py
│   ├── test_chunker.py
│   ├── test_context_selector_unit.py
│   ├── test_document_management.py
│   ├── test_ingestion.py
│   ├── test_ingestion_retrieval_integration.py
│   ├── test_pdf_chunking.py
│   ├── test_pdf_cleaning.py
│   ├── test_pdf_loader.py
│   ├── test_recovery.py
│   ├── test_registry.py
│   ├── test_retriever_status.py
│   ├── test_search.py
│   ├── test_section_aware_chunker.py
│   ├── test_text_cleaner.py
│   └── test_vector_store_integration.py
│
├── data/                  # Local PDFs and database storage
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

---

## Core Components

### 1. PDF Ingestion Pipeline

The ingestion pipeline processes PDF documents and prepares their content for retrieval.

Processing workflow:

1. Load PDF documents using PyMuPDF.
2. Extract text page by page.
3. Clean and normalize extracted text.
4. Divide content into meaningful chunks.
5. Generate dense and sparse representations.
6. Store vectors and metadata in Qdrant.
7. Track document processing status in PostgreSQL.

### 2. Document Chunking

The project supports multiple chunking strategies:

- Fixed-size chunking
- Page-aware chunking
- Section-aware chunking

Different strategies are evaluated to understand their impact on retrieval quality and preservation of document context.

### 3. Hybrid Retrieval

The retrieval pipeline combines dense and sparse search.

**Dense Retrieval**

Uses semantic embeddings to identify content that is conceptually related to the user's query.

**Sparse Retrieval**

Uses BM25-based keyword matching to identify documents containing relevant terms.

**Reciprocal Rank Fusion (RRF)**

Combines ranked results from dense and sparse retrieval into a unified ranking.

**Cross-Encoder Reranking**

Reranks retrieved candidates using a Cross-Encoder model to improve the relevance of the final results.

### 4. Context Selection

The context selection component prepares relevant retrieved chunks for answer generation.

It helps select useful information from retrieved candidates instead of passing every retrieved chunk directly to the LLM.

### 5. Document Lifecycle Management

PostgreSQL maintains document processing information and lifecycle state.

The system tracks document statuses and supports operations such as:

- Document registration
- Processing status updates
- Heartbeat updates
- Document deletion
- Stale document detection
- Pending vector cleanup

### 6. Recovery and Cleanup Mechanism

The recovery module handles incomplete document processing and failed vector cleanup operations.

Key capabilities include:

- Detection of stale processing records
- Identification of pending cleanup operations
- Retry of failed Qdrant cleanup operations
- Cleanup claim leases
- UUID claim tokens for concurrency safety
- Protection against outdated workers modifying newer cleanup claims
- BM25 index refresh after relevant cleanup operations

This separates document ingestion status from vector cleanup status, making recovery operations more reliable.

---

## Evaluation and Experimentation Scripts

The `scripts/` directory contains 26 Python scripts developed to evaluate, compare, inspect, and improve different components of the RAG pipeline.

These scripts support experimentation with chunking strategies, retrieval methods, reranking, context selection, abstract retrieval, and answer quality.

### 1. Chunking Evaluation

| Script | Purpose |
|---|---|
| `evaluate_chunking.py` | Evaluates document chunking behavior. |
| `compare_chunking_evaluation.py` | Compares chunking evaluation results. |
| `compare_chunking_retrieval.py` | Compares retrieval performance across chunking approaches. |
| `compare_chunking_strategies.py` | Compares different chunking strategies. |
| `evaluate_section_aware_chunking.py` | Evaluates section-aware chunking. |

### 2. Retrieval Strategy Evaluation

| Script | Purpose |
|---|---|
| `compare_retrieval_strategies.py` | Compares different retrieval approaches. |
| `diagnose_dense_sparse.py` | Investigates dense and sparse retrieval behavior. |
| `diagnose_missed_questions.py` | Investigates questions that retrieval fails to answer. |
| `diagnose_query_variants.py` | Examines retrieval behavior across query variations. |
| `diagnose_abstract_retrieval.py` | Investigates abstract-related retrieval behavior. |

### 3. Reranking Evaluation

| Script | Purpose |
|---|---|
| `compare_reranking_strategies.py` | Compares reranking strategies and their effect on result ranking. |

### 4. Context Selection and Answer Quality

| Script | Purpose |
|---|---|
| `evaluate_context_selection.py` | Evaluates context selection behavior. |
| `evaluate_answer_quality.py` | Evaluates answer quality. |
| `evaluate_abstract_context.py` | Evaluates abstract-related context selection. |
| `test_context_selection.py` | Tests context selection behavior. |
| `test_context_selection_real.py` | Tests context selection with real retrieval data. |
| `test_context_selector.py` | Tests the context selector. |

### 5. Abstract and Intent-Aware Retrieval Experiments

| Script | Purpose |
|---|---|
| `add_abstract_to_eval_collections.py` | Adds abstract-related data to evaluation collections. |
| `evaluate_abstract_boost.py` | Evaluates abstract boosting approaches. |
| `evaluate_abstract_chunk.py` | Evaluates abstract chunk retrieval. |
| `evaluate_intent_aware_abstract.py` | Evaluates intent-aware abstract retrieval. |
| `evaluate_multi_query.py` | Evaluates multi-query retrieval. |
| `label_eval_abstract.py` | Labels abstract-related evaluation data. |

### 6. Retrieval Investigation and Evidence Inspection

| Script | Purpose |
|---|---|
| `diagnose_broad_improvements.py` | Investigates broader retrieval improvements. |
| `inspect_broad_improvements.py` | Inspects potential retrieval improvements. |
| `inspect_q2_evidence.py` | Inspects evidence for a specific evaluation question. |

These scripts support iterative experimentation and help identify weaknesses in retrieval quality and context selection.

---

## Installation and Setup

### Prerequisites

Make sure the following tools are installed:

- Python 3.11
- Git
- Docker Desktop
- Docker Compose
- Ollama, if using Ollama for answer generation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Production-RAG.git
cd Production-RAG
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure the required values.

Important settings include:

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=document_chunks

DENSE_MODEL=sentence-transformers/all-MiniLM-L6-v2
SPARSE_MODEL=Qdrant/bm25
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma4:cloud

POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=production_rag
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=your_own_password
```

**Important:** Use your own PostgreSQL credentials. Never commit your `.env` file or actual passwords to GitHub.

### 5. Start Qdrant and PostgreSQL

Start the infrastructure using Docker Compose:

```powershell
docker compose up -d
```

Check the running containers:

```powershell
docker compose ps
```

The services are configured as follows:

| Service | Address |
|---|---|
| Qdrant | http://localhost:6333 |
| PostgreSQL | localhost:5433 |

### 6. Configure Ollama

Make sure Ollama is running and that the model configured in your `.env` file is available.

For example, to download a local model:

```powershell
ollama pull llama3.2:1b
```

If you use a different model, update `OLLAMA_MODEL` in `.env` to match it.

### 7. Run the FastAPI Application

Start the application:

```powershell
uvicorn app.api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

---

## Running Tests

The project uses Pytest for unit and integration testing.

Run the complete test suite:

```powershell
pytest -v
```

Run a specific test file:

```powershell
pytest tests/test_recovery.py -v
```

Run registry tests:

```powershell
pytest tests/test_registry.py -v
```

Run tests with a concise summary:

```powershell
pytest -q
```

### Test Coverage Areas

- PDF loading
- Text cleaning
- Chunk generation
- Section-aware chunking
- Document ingestion
- Vector storage
- BM25 retrieval
- Hybrid retrieval
- Context selection
- API endpoints
- Document lifecycle management
- Recovery and cleanup operations

**Latest recorded full test result: 141 passed, 0 failed.**

---

## Current Project Status

The following components have been implemented and tested:

- PDF loading and text extraction
- Text cleaning and preprocessing
- Multiple chunking strategies
- Dense and sparse retrieval
- Hybrid retrieval using RRF
- Cross-Encoder reranking
- Context selection
- Qdrant vector storage
- PostgreSQL document registry
- Document lifecycle management
- Stale document recovery
- Retryable vector cleanup
- UUID-based cleanup claim protection
- Unit and integration tests
- Retrieval and chunking evaluation scripts

The project is under active development, with continued work focused on retrieval quality, reliability, and scalability.

---

## Future Improvements

- Large-scale evaluation across thousands of PDF documents
- Retrieval latency and throughput benchmarking
- Concurrent document ingestion testing
- Further improvements to context selection
- Production deployment and monitoring
- Authentication and multi-tenant access controls
- Improved observability and operational metrics

---

## Author

**Ujjwal Prajapati**

GitHub : https://github.com/ujjwalprajapatinareshbhai

---

If you find this project useful, feel free to explore the code and evaluation scripts.