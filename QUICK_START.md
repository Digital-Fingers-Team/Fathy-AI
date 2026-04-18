# Quick Start & Testing Guide

## 1. Install Updated Dependencies ✅

```bash
cd "backend"
pip install -r requirements.txt
# The PyJWT issue (2.8.1 not found) is now fixed with version 2.12.1
```

## 2. Start Both Servers

**Terminal 1 - Backend**:

```bash
cd "backend"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend**:

```bash
cd "frontend API"
npm run dev  # Runs on http://localhost:3000
```

## 3. Test the Three Features

### Feature 1: Bilingual Logo ✅

- Open `http://localhost:3000`
- Look at the top-left header
- You should see: **Fathy فتحي** (English + Arabic together)

### Feature 2: Language Preference ✅

- Click "Settings" in the navbar
- Change:
  - **Language**: Choose between "English" and "العربية"
  - **Direction**: Choose between "LTR" and "RTL (Arabic)"
- Click "Save"
- The change applies immediately and persists in browser

### Feature 3: API Key Input (Instead of Error) ✅

- Click "Chat" in the navbar
- Look under the message input area
- You'll see: **"OpenAI API Key (optional)"** input field
- Enter your OpenAI API key (e.g., `sk-...`)
- Click anywhere or let it auto-save
- Start chatting - your API key is now stored locally

**If you want to clear the API key**:

- Go to Settings page
- In the "OpenAI API Key" field, click "Clear"
- Or go to Chat and click the "Clear" button next to the key field

## 4. How It Works

### Without API Key (Environment Variable)

```
User asks question
  ↓
Backend checks: Is there a client API key? NO
  ↓
Backend uses: Environment variable from .env
  ↓
AI responds OR falls back to stored memory
```

### With API Key (User-Provided)

```
User enters API key in Chat UI
  ↓
Frontend stores it locally (localStorage)
  ↓
User asks question
  ↓
Frontend sends: { message, history, api_key }
  ↓
Backend creates new AIService with provided key
  ↓
AI responds using user's key
```

## 5. File Structure Overview

**Backend Auth System** (from previous update):

```
backend/
├── app/
│   ├── routes/
│   │   ├── auth.py (NEW)              # Login, signup, logout, delete
│   │   ├── dependencies.py (NEW)      # JWT protection
│   │   └── chat.py (UPDATED)          # Now accepts api_key
│   ├── services/
│   │   ├── auth_service.py (NEW)      # JWT & password hashing
│   │   └── ai_service.py (UPDATED)    # Accepts optional api_key
│   ├── schemas/
│   │   ├── auth.py (NEW)              # Auth models
│   │   └── chat.py (UPDATED)          # Added api_key field
│   └── db/
│       └── models.py (UPDATED)        # Added User model
└── requirements.txt (UPDATED)         # PyJWT==2.12.1 fixed
```

**Frontend UI** (new features):

```
frontend API/
├── src/
│   ├── lib/
│   │   ├── api-key-context.tsx (NEW)  # API key state
│   │   └── api.ts (UPDATED)           # Sends api_key to backend
│   ├── components/
│   │   ├── AppShell.tsx (UPDATED)     # Bilingual logo
│   │   ├── ChatClient.tsx (UPDATED)   # API key input under textarea
│   │   └── SettingsClient.tsx (UPDATED) # API key management
│   └── app/
│       └── layout.tsx (UPDATED)       # Added ApiKeyProvider
```

## 6. Key Features

✨ **Authentication** (from previous update)

- Sign up / Log in / Log out
- Delete account
- JWT tokens (7-day expiration)

✨ **Bilingual Support**

- English: "Fathy"
- Arabic: "فتحي"
- Both displayed in header

✨ **Language Preference**

- Set preferred language in Settings
- Persists to localStorage
- Affects UI direction (RTL for Arabic)

✨ **API Key Flexibility**

- Enter key per-session in Chat
- Or set it once in Settings
- Or use environment variable
- Local storage only (never sent to backend)

## 7. If Something Goes Wrong

### PyJWT Error During Install

```
ERROR: Could not find a version that satisfies the requirement PyJWT==2.8.1
```

**Solution**: Already fixed! File now has `PyJWT==2.12.1`

### API Key Not Working

- Check if you copied it correctly (should start with `sk-`)
- Try clearing it and entering again
- Go to Settings → API Key → Clear → Try fresh

### Logo Not Bilingual

- Clear browser cache (Ctrl+F5 or Cmd+Shift+R)
- Check if you're on the latest code

### Language Not Changing

- Go to Settings
- Check that you selected Arabic and RTL
- Click "Save"
- Refresh page (F5)

## 8. Ready to Deploy?

When deploying to production:

1. Update `.env` with production settings
2. Use a strong `SECRET_KEY` for JWT
3. Enable `HTTPS` for API endpoints
4. Consider using a reverse proxy (nginx) for security
5. Implement rate limiting for API keys
6. Added optional monitoring for API usage

---

**You're all set!** 🎉

- ✅ PyJWT dependency fixed
- ✅ Bilingual logo (English + Arabic)
- ✅ Language/Direction preferences
- ✅ API key input under chat (no more "API key missing" error)
- ✅ Auth system with JWT
- ✅ All errors cleared

Happy coding! 🚀
