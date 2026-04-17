# Fathy (فتحي) — Production-minded MVP

Fathy is a bilingual (Arabic/English) AI assistant that **does not “self-train”**.  
It simulates learning via a **real, explainable workflow**:

- Persistent memory (SQLite)
- Retrieval-first answering (RAG-style prompt injection)
- Teaching + feedback workflows (Teach + Memory Manager)

## Monorepo

```
/backend API
  /app
    main.py
    /core
      config.py
      logging.py
    /routes
      chat.py
      teach.py
      memory.py
      health.py
    /services
      ai_service.py
      memory_service.py
    /repositories
      memory_repo.py
    /schemas
      chat.py
      memory.py
    /db
      session.py
      models.py
  example.env
  requirements.txt

/frontend API
  src/...
  package.json
  next.config.js
  example.env
```

## Backend (FastAPI)

### 1) Setup

```bash
cd "backend API"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy example.env .env
```

### 2) Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
if not working :
python -m uvicorn app.main:app --reload

### API

- `GET  /health`
- `POST /chat` body: `{ "message": "..." }`
- `POST /teach` body: `{ "question": "...", "answer": "...", "tags": ["..."] }`
- `GET  /memory?q=...`
- `PUT  /memory/{id}`
- `DELETE /memory/{id}`

### Notes

- If `OPENAI_API_KEY` is **not** set, `/chat` will **not** call a model. It will answer from stored memory only (if any).
- Memory ranking is currently token-overlap + substring boost + small recency boost, implemented in `backend API/app/services/memory_service.py`.

## Frontend (Next.js)

### 1) Setup

Install Node.js 18+ (or 20+), then:

```bash
cd "frontend API"
copy example.env .env.local
npm install
```

> The provided `frontend API/example.env` uses `VITE_API_URL` as requested.  
> `frontend API/next.config.js` maps it to `NEXT_PUBLIC_API_URL` automatically.

### 2) Run

```bash
npm run dev
```

Open:

- Frontend: http://localhost:3000
- Backend: http://localhost:8000

## Upgrade path (Vector DB)

This MVP keeps memory in SQLite and ranks in Python for clarity and correctness.  
To upgrade to vectors later:

- Add an `embedding` column (or a separate table) to `MemoryItem`
- Compute embeddings on `/teach`
- Replace `MemoryService.search()` with FAISS/Pinecone/pgvector retrieval

