# AI Document Intelligence UI

This React app is a provided UI for the FastAPI backend.

## Run

Start the backend first from the project root:

```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Then run the frontend:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Screens

The UI includes:

```text
Pipeline status
Claims table
RAG question answering
LLM agent queries
Agent audit trail
```