# Fathy (فتحي) — Unified Project Guide

This repository now uses a **single root markdown document** for setup, usage, auth, updates, and testing notes.

---

## Project Overview

Fathy is a bilingual (Arabic/English) AI assistant that **does not “self-train”**. It simulates learning via an explainable workflow:

- Persistent memory (SQLite)
- Retrieval-first answering (RAG-style prompt injection)
- Teaching + feedback workflows (Teach + Memory Manager)

## Monorepo

```text
/backend
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

---

## Backend (FastAPI)

### Setup

```bash
cd "backend"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy example.env .env
```

### Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# if needed:
python -m uvicorn app.main:app --reload
```

### API Endpoints

- `GET /health`
- `POST /chat` body: `{ "message": "..." }`
- `POST /teach` body: `{ "question": "...", "answer": "...", "tags": ["..."] }`
- `GET /memory?q=...`
- `PUT /memory/{id}`
- `DELETE /memory/{id}`

### Backend Notes

- If `OPENAI_API_KEY` is not set, `/chat` tries the local `fathy-llm/checkpoints/sft/latest.pt` checkpoint first, then falls back to stored memory if needed.
- Memory ranking uses token-overlap + substring boost + recency boost in `backend/app/services/memory_service.py`.

---

## Frontend (Next.js)

### Setup

```bash
cd "frontend"
copy example.env .env.local
npm install
```

> The provided `frontend API/example.env` uses `VITE_API_URL`; `frontend API/next.config.js` maps it to `NEXT_PUBLIC_API_URL`.

### Run

```bash
npm run dev
```

Open:

- Frontend: http://localhost:3000
- Backend: http://localhost:8000

---

## Quick Start Testing Checklist

### 1) Install Dependencies

```bash
cd "backend"
pip install -r requirements.txt
```

### 2) Start Both Servers

Terminal 1:

```bash
cd "backend"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
cd "frontend API"
npm run dev
```

### 3) Verify Main Features

- Bilingual logo in header: **Fathy فتحي**
- Settings language switch (English/العربية)
- Settings direction switch (LTR/RTL)
- Chat page optional API key input

---

## Authentication System

### Backend auth components

- `app/schemas/auth.py`
- `app/services/auth_service.py`
- `app/routes/auth.py`
- `app/routes/dependencies.py`

### Auth endpoints

- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/logout`
- `DELETE /auth/account`
- `GET /auth/me`

### Frontend auth components

- `src/lib/auth-context.tsx`
- `src/components/LoginClient.tsx`
- `src/components/SignupClient.tsx`
- `src/app/auth/login/page.tsx`
- `src/app/auth/signup/page.tsx`

### Auth notes

- JWT-based auth, token persistence in localStorage
- User menu supports logout and account deletion
- Password hashing via bcrypt/passlib

---

## Conversation Endpoints (Authenticated)

- `GET /conversations`
- `POST /conversations`
- `GET /conversations/{id}/messages`
- `DELETE /conversations/{id}`
- `PATCH /conversations/{id}/title`

`POST /chat` supports optional `conversation_id` and returns it.
`POST /chat/stream` supports SSE events and optional `conversation_id`.

---

## Recent Functional Updates

- PyJWT version aligned to `2.12.1`
- Bilingual logo (English + Arabic)
- API key management in chat/settings
- Language preference persistence and RTL support

### API key behavior

- API keys are stored locally in browser localStorage
- Client key can be passed per request
- If absent, backend falls back to environment key

---

## Troubleshooting

### Dependency install errors

- Re-run `pip install -r requirements.txt`
- Ensure active Python environment is correct

### API key issues

- Verify key format starts with `sk-`
- Clear and re-enter key from Settings

### UI issues

- Hard refresh browser cache
- Confirm frontend runs latest local code

---

## Upgrade Path (Vector DB)

Current MVP keeps memory in SQLite and ranks in Python. To move to vectors later:

- Add embedding storage for `MemoryItem`
- Compute embeddings on `/teach`
- Replace current search with FAISS/Pinecone/pgvector retrieval

---

## Production Notes

1. Set strong production `SECRET_KEY`
2. Configure HTTPS
3. Restrict CORS origins
4. Add rate limiting and monitoring
5. Consider reverse proxy (nginx)

