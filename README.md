# Bond Taxation RAG Chatbot

A Render-ready Flask chatbot built from the uploaded `Taxation.pdf` knowledge base.

## What is improved

- Structured, page-referenced `chunks.json`
- Lightweight hybrid retrieval: keywords, question patterns, intent tags, synonyms, and exact phrase boosts
- Strict answer prompt grounded only on selected chunks
- Deterministic fallback answers for common high-risk questions
- Retrieval trace in the UI for debugging relevance
- Render-ready `gunicorn` setup

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OLLAMA_API_KEY="your_ollama_api_key"
export OLLAMA_MODEL="gpt-oss:120b"
export OLLAMA_HOST="https://ollama.com"
python app.py
```

Open:

```text
http://127.0.0.1:1707/
```

## Render deployment

Create a Python Web Service and set:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 240
```

Add environment variables:

```text
OLLAMA_API_KEY=your_real_key
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_HOST=https://ollama.com
```

## Health checks

```text
/health
/api/health
/api/chunks
```

## Notes

- Do not commit `.env` or API keys.
- If the Ollama API key is missing or the model call fails, the app still returns a deterministic KB-based fallback answer.
- The app is designed for support education and does not provide personal tax advice.
