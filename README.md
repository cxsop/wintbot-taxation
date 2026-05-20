# Bond Taxation Chatbot - Render Ready

This is a simple Flask chatbot for listed NCD taxation queries.

It uses:
- Flask for the web app
- Keyword-based chunk retrieval from `chunks.json`
- Ollama Cloud API for model responses
- Gunicorn for production deployment on Render

## Project files

```text
app.py
chunks.json
templates/index.html
requirements.txt
render.yaml
.gitignore
README.md
```

## Environment variables

Do not commit API keys.

Add these in Render Dashboard > Environment:

```text
OLLAMA_API_KEY=your_real_ollama_api_key
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_HOST=https://ollama.com
```

Optional:

```text
FLASK_SECRET_KEY=any_long_random_string
```

Render will automatically provide:

```text
PORT
```

## Local run

If you want to test locally without a `.env` file, export the key only for the current terminal session:

```bash
export OLLAMA_API_KEY="your_real_ollama_api_key"
export OLLAMA_MODEL="gpt-oss:120b"
export OLLAMA_HOST="https://ollama.com"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:1707
```

## Render deployment

Option 1: Use `render.yaml`.

1. Push this folder to GitHub.
2. In Render, choose New > Blueprint.
3. Select your GitHub repo.
4. Add `OLLAMA_API_KEY` when Render asks for environment variables.
5. Deploy.

Option 2: Manual Web Service.

Use these settings:

```text
Language: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 240
```

Add environment variables:

```text
OLLAMA_API_KEY
OLLAMA_MODEL
OLLAMA_HOST
```

## Health check

After deploy, open:

```text
https://your-app-name.onrender.com/health
```

You should see:

```json
{
  "status": "ok",
  "provider": "ollama_cloud",
  "api_key_configured": true,
  "chunks_loaded": 20
}
```

## Test questions

```text
How is listed NCD interest taxed?
If I sell my bond before maturity, how is tax calculated?
Is indexation available for listed NCD?
Can capital loss from bond sale be set off against salary income?
```
