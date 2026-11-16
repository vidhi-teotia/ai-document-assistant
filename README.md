# AI Document Assistant

An AI-powered document intelligence platform that allows users to interact with PDF documents, retrieve relevant information, and receive context-aware answers using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs).

The application combines a React frontend, FastAPI backend, ChromaDB vector database, and locally hosted Ollama models to provide an interactive document question-answering experience.

## Features

- **PDF Document Processing:** Extract and preprocess text from PDF documents.
- **Semantic Search:** Generate embeddings and retrieve relevant document chunks using ChromaDB.
- **RAG-Based Question Answering:** Generate context-grounded answers using retrieved document content.
- **Structured Information Extraction:** Extract relevant healthcare claim information into structured records.
- **AI Agent:** Process natural-language queries and select appropriate document tools.
- **Claims Dashboard:** View available claim records and pipeline status.
- **Audit Trail:** Track AI agent query activity.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React 19, JavaScript, CSS |
| Backend | Python, FastAPI |
| LLM Integration | LangChain, Ollama |
| Chat Model | Llama 3.2 |
| Embedding Model | Nomic Embed Text |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF, pdfplumber |
| Data Processing | Pandas |
| Validation | Pydantic |

## Architecture

```text
                 React Frontend
                       |
                       v
                FastAPI Backend
                       |
          +------------+------------+
          |            |            |
          v            v            v
     PDF Pipeline    RAG Engine   AI Agent
          |            |            |
          v            v            v
    Text Extraction  Embeddings  Document Tools
          |            |            |
          v            v            v
    Text Cleaning   ChromaDB    LLM Processing
          |            |            |
          +------------+------------+
                       |
                       v
               Ollama LLM Models
                       |
                       v
                AI-Generated Answers
```

## Project Structure

```text
ai-document-assistant/
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── README.md
│
├── src/
│   ├── agents/
│   ├── api/
│   ├── config/
│   ├── extraction/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── rag/
│   └── validation/
│
├── data/
├── main.py
├── run_claim_extraction.py
├── requirements.txt
└── README.md
```

## Prerequisites

Install the following before running the application:

- Python 3.10 or later
- Node.js and npm
- Git
- Ollama

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vidhi-teotia/ai-document-assistant.git
cd ai-document-assistant
```

### 2. Set Up the Python Backend

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

### 3. Configure Ollama

Install Ollama and download the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Ensure Ollama is running locally before starting the application.

### 4. Start the FastAPI Backend

From the project root:

```bash
python -m uvicorn src.api.app:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Set Up the React Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm start
```

Frontend URL:

```text
http://localhost:3000
```

Configure the frontend API base URL to point to the running FastAPI backend.

## Application Modules

**Document Processing Pipeline**

Extracts text from PDF files, cleans the content, divides it into smaller chunks, and prepares it for semantic retrieval.

**Retrieval-Augmented Generation (RAG)**

Converts document chunks into vector embeddings, stores them in ChromaDB, retrieves relevant passages for user queries, and generates answers using an LLM.

**AI Agent**

Uses an LLM-based agent to interpret user questions, select available document tools, and process requests.

**Healthcare Claims Extraction**

Extracts structured information from healthcare claim documents and validates the resulting records using Pydantic schemas.

## Future Enhancements

- Cloud deployment and public demo access.
- Improved document citation and source highlighting.
- Support for additional document formats.
- Enhanced agent workflows and document analytics.

## Notes

- Ollama models run locally and must be installed separately.
- The `data/` directory is not included in the GitHub repository. Add your own sample PDF documents before running the document pipeline.
- Do not upload real patient records, confidential documents, API keys, or environment files.
- The application currently uses local storage for its vector database.

## Author

**Vidhi Teotia**

[GitHub](https://github.com/vidhi-teotia)

---

*AI Document Assistant — Turning unstructured documents into accessible knowledge.*