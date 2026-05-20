import json
import os
import re
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request, session


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))

# ---------------------------------------------------------------------
# Render-ready config
# ---------------------------------------------------------------------
# Add these in Render Dashboard > Environment:
# OLLAMA_API_KEY = your Ollama Cloud API key
# Optional:
# OLLAMA_MODEL = gpt-oss:120b
# OLLAMA_HOST = https://ollama.com

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com").rstrip("/")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b").strip()

# Render automatically provides PORT for web services.
PORT = int(os.environ.get("PORT", "1707"))

CHUNKS_FILE = Path("chunks.json")

MAX_SELECTED_CHUNKS = 5
MAX_CONTEXT_CHARS = 5200
REQUEST_TIMEOUT_SECONDS = 240


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am",
    "i", "me", "my", "you", "your", "we", "our",
    "to", "of", "for", "from", "in", "on", "at", "by",
    "and", "or", "but", "if", "then", "than",
    "what", "why", "how", "when", "where", "which",
    "can", "could", "should", "would", "do", "does", "did",
    "be", "been", "being", "with", "without", "as", "it",
    "this", "that", "these", "those", "please", "pls",
    "tell", "explain", "user", "customer", "sir", "madam",
    "about", "regarding", "query", "question"
}


SYNONYMS = {
    "sell": ["sale", "sold", "selling", "secondary market", "exit", "before maturity"],
    "sale": ["sell", "sold", "selling", "secondary market", "exit", "before maturity"],
    "maturity": ["redemption", "redeem", "face value", "matured"],
    "redemption": ["maturity", "redeem", "face value", "matured"],
    "interest": ["coupon", "income from other sources", "due date"],
    "coupon": ["interest", "income from other sources"],
    "tds": ["tax deducted", "section 193", "10 percent", "deduction"],
    "ltcg": ["long term", "long-term", "capital gain", "12 months"],
    "stcg": ["short term", "short-term", "capital gain", "12 months", "slab"],
    "indexation": ["cost inflation", "inflation benefit"],
    "loss": ["capital loss", "set off", "carry forward"],
    "setoff": ["set off", "loss adjustment", "capital loss"],
    "capital": ["capital gain", "capital loss", "capital asset"],
    "ncd": ["bond", "debenture", "listed ncd"],
    "bond": ["ncd", "debenture", "listed ncd"],
    "tax": ["taxation", "income tax", "capital gains", "tds"],
}


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("₹", " rs ")
    text = text.replace("%", " percent ")
    text = re.sub(r"[^a-z0-9\s\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    words = normalize_text(text).split()
    return [word for word in words if word not in STOPWORDS and len(word) > 1]


def expand_query(query):
    query_normalized = normalize_text(query)
    expanded = query_normalized

    for word, related_words in SYNONYMS.items():
        if word in query_normalized:
            expanded += " " + " ".join(related_words)

    return expanded


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("chunks.json file not found in the project folder.")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and isinstance(data.get("chunks"), list):
        chunks = data["chunks"]
    elif isinstance(data, list):
        chunks = data
    else:
        raise ValueError("chunks.json must be a list or an object with a 'chunks' list.")

    valid_chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]

    if not valid_chunks:
        raise ValueError("No valid chunk objects found in chunks.json.")

    return valid_chunks


TAX_CHUNKS = load_chunks()


def chunk_to_searchable_text(chunk):
    fields = [
        chunk.get("id", ""),
        chunk.get("title", ""),
        chunk.get("category", ""),
        chunk.get("chunk", ""),
        chunk.get("answer_guidance", ""),
    ]

    keywords = chunk.get("keywords", [])
    sample_questions = chunk.get("sample_questions", [])

    if isinstance(keywords, list):
        fields.append(" ".join(str(item) for item in keywords))

    if isinstance(sample_questions, list):
        fields.append(" ".join(str(item) for item in sample_questions))

    return normalize_text(" ".join(fields))


def score_chunk(query, chunk):
    if not isinstance(chunk, dict):
        return 0

    expanded_query = expand_query(query)
    query_text = normalize_text(expanded_query)
    query_tokens = set(tokenize(expanded_query))

    searchable_text = chunk_to_searchable_text(chunk)
    searchable_tokens = set(tokenize(searchable_text))

    score = len(query_tokens.intersection(searchable_tokens)) * 2

    keywords = chunk.get("keywords", [])
    if isinstance(keywords, list):
        for keyword in keywords:
            keyword_text = normalize_text(keyword)

            if not keyword_text:
                continue

            if keyword_text in query_text:
                score += 12

            keyword_tokens = set(tokenize(keyword_text))
            if keyword_tokens and keyword_tokens.issubset(query_tokens):
                score += 8

    sample_questions = chunk.get("sample_questions", [])
    if isinstance(sample_questions, list):
        for sample_question in sample_questions:
            sample_tokens = set(tokenize(sample_question))
            overlap = len(query_tokens.intersection(sample_tokens))
            if overlap >= 2:
                score += overlap * 3

    title_tokens = set(tokenize(chunk.get("title", "")))
    category_tokens = set(tokenize(chunk.get("category", "")))
    score += len(query_tokens.intersection(title_tokens)) * 4
    score += len(query_tokens.intersection(category_tokens)) * 4

    q = query_text
    category = normalize_text(chunk.get("category", ""))
    title = normalize_text(chunk.get("title", ""))

    if any(word in q for word in ["sell", "selling", "sold", "secondary", "exit", "before maturity"]):
        if "secondary" in category or "sale" in category or "sale" in title:
            score += 25
        if "capital" in category or "capital" in title:
            score += 8

    if any(word in q for word in ["maturity", "redeem", "redemption", "face value", "matured"]):
        if "redemption" in category or "redemption" in title or "maturity" in title:
            score += 25

    if any(word in q for word in ["ltcg", "long term", "long-term"]):
        if "long-term" in title or "ltcg" in searchable_text:
            score += 25

    if any(word in q for word in ["stcg", "short term", "short-term", "slab"]):
        if "short-term" in title or "stcg" in searchable_text or "slab" in searchable_text:
            score += 25

    if any(word in q for word in ["tds", "tax deducted", "section 193", "deduction"]):
        if "tds" in category or "tds" in title or "section 193" in searchable_text:
            score += 25

    if any(word in q for word in ["interest", "coupon", "income from other sources"]):
        if "interest" in category or "interest" in title or "coupon" in searchable_text:
            score += 25

    if any(word in q for word in ["indexation", "inflation", "cost inflation"]):
        if "indexation" in category or "indexation" in title:
            score += 30

    if any(word in q for word in ["loss", "set off", "set-off", "carry forward", "adjust"]):
        if "loss" in category or "loss" in title or "set" in title:
            score += 25

    if any(word in q for word in ["capital gain", "capital gains", "gain", "profit", "tax on sale"]):
        if "capital" in category or "capital" in title or "capital gain" in searchable_text:
            score += 15

    return score


def retrieve_chunks(query):
    scored_chunks = []

    for chunk in TAX_CHUNKS:
        scored_chunks.append({
            "score": score_chunk(query, chunk),
            "chunk": chunk,
        })

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    selected = []
    used_chars = 0

    for item in scored_chunks:
        if item["score"] <= 0:
            continue

        chunk = item["chunk"]
        chunk_text = json.dumps(chunk, ensure_ascii=False)

        if used_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
            continue

        selected.append({
            "score": item["score"],
            **chunk,
        })

        used_chars += len(chunk_text)

        if len(selected) >= MAX_SELECTED_CHUNKS:
            break

    return selected


def is_greeting_or_smalltalk(query):
    q = normalize_text(query)
    return q in {"hi", "hello", "hey", "test", "good morning", "good evening"}


def build_context(selected_chunks):
    parts = []

    for index, chunk in enumerate(selected_chunks, start=1):
        parts.append(
            f"""
CHUNK {index}
ID: {chunk.get("id", "")}
TITLE: {chunk.get("title", "")}
CATEGORY: {chunk.get("category", "")}
CONTENT: {chunk.get("chunk", "")}
ANSWER GUIDANCE: {chunk.get("answer_guidance", "")}
""".strip()
        )

    return "\n\n".join(parts)


def build_local_fallback_answer(selected_chunks):
    if not selected_chunks:
        return (
            "Sir/Madam, I can answer only from the listed NCD taxation knowledge base. "
            "Please ask about listed NCD tax treatment, interest taxation, TDS, capital gains, "
            "LTCG, STCG, indexation, redemption, secondary market sale, or capital loss set-off."
        )

    selected_text = " ".join(
        normalize_text(chunk.get("title", "") + " " + chunk.get("chunk", "") + " " + chunk.get("category", ""))
        for chunk in selected_chunks
    )

    if "secondary" in selected_text or "sell" in selected_text or "sale" in selected_text:
        return (
            "Sir/Madam,\n\n"
            "If you sell your listed NCD before maturity, the tax impact will be based on capital gain or capital loss.\n\n"
            "The calculation is:\n"
            "Sale Price minus Cost of Acquisition.\n\n"
            "The applicable tax treatment depends on the holding period:\n"
            "- If the NCD is held for more than 12 months, it is treated as long-term capital gain. The LTCG tax rate is 12.5%.\n"
            "- If the NCD is held for 12 months or less, it is treated as short-term capital gain. It is taxed as per the investor’s applicable slab rate.\n\n"
            "Please note, indexation benefit is not available for listed NCDs.\n\n"
            "For personal tax filing treatment, you may consult a CA or tax advisor."
        )

    if "indexation" in selected_text:
        return (
            "Sir/Madam,\n\n"
            "Indexation benefit is not available for listed NCDs.\n\n"
            "This means the cost of acquisition cannot be adjusted using indexation while calculating capital gains.\n\n"
            "For personal tax filing treatment, you may consult a CA or tax advisor."
        )

    if "interest" in selected_text or "coupon" in selected_text:
        return (
            "Sir/Madam,\n\n"
            "Coupon interest from listed NCDs is taxable under the head ‘Income from Other Sources’ on a due-date basis.\n\n"
            "TDS on NCD interest is deducted under Section 193 at 10% at the time of payment, wherever applicable.\n\n"
            "For personal tax filing treatment, you may consult a CA or tax advisor."
        )

    lines = ["Sir/Madam, based on the listed NCD taxation knowledge base:\n"]

    for chunk in selected_chunks[:3]:
        lines.append(f"- {chunk.get('chunk', '')}")

    lines.append("\nFor personal tax filing treatment, you may consult a CA or tax advisor.")
    return "\n".join(lines)


def looks_like_bad_model_answer(answer):
    if not answer:
        return True

    answer_lower = answer.lower()

    weak_phrases = [
        "find contact information",
        "do you want me to help you find",
        "i cannot provide tax advice",
        "i am not able to answer",
    ]

    if any(phrase in answer_lower for phrase in weak_phrases):
        return True

    return False


def looks_incomplete_answer(answer, done_reason=""):
    if not answer:
        return True

    if done_reason in {"length", "num_predict"}:
        return True

    answer_clean = answer.strip().lower()

    incomplete_endings = [
        "depends on",
        "depends on:",
        "based on",
        "based on:",
        "as follows",
        "as follows:",
        "the applicable tax rate depends on",
        "the tax rate depends on",
        "will apply based on",
        "is calculated as",
        "is calculated by",
        "the holding period:",
    ]

    if any(answer_clean.endswith(ending) for ending in incomplete_endings):
        return True

    if answer_clean[-1] not in [".", "!", "?", ")", "\"", "”"]:
        return True

    return False


def call_ollama_cloud(user_message, selected_chunks):
    if not OLLAMA_API_KEY:
        raise RuntimeError(
            "OLLAMA_API_KEY is not configured. Add it in Render Environment Variables."
        )

    context = build_context(selected_chunks)

    system_prompt = """
You are a bond taxation support assistant for listed NCD related queries.

Rules:
1. Answer only using the provided CONTEXT.
2. Give a complete answer. Never stop mid-sentence.
3. Do not give generic tax-advisor-only answers.
4. Do not say you can help find tax advisors.
5. Keep the answer practical for a customer support associate.
6. Use simple Indian English.
7. Address the customer as Sir/Madam.
8. If explaining sale before maturity, always include:
   - Capital gain/loss formula
   - More than 12 months = LTCG
   - 12 months or less = STCG
   - Applicable tax rate if available in context
9. End with a short disclaimer only if useful.
10. Do not invent rates, rules, dates, or tax sections beyond the CONTEXT.
""".strip()

    user_prompt = f"""
CONTEXT:
{context}

USER QUESTION:
{user_message}

TASK:
Answer the user question using only the CONTEXT.
Give a complete customer-facing response.
Do not end with an incomplete sentence.
""".strip()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 900,
        },
    }

    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()
    done_reason = data.get("done_reason", "")

    return answer, done_reason


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please enter a question.", "selected_chunks": []}), 400

    if is_greeting_or_smalltalk(user_message):
        return jsonify({
            "reply": (
                "Hello Sir/Madam. I can help with listed NCD taxation queries such as "
                "interest taxation, TDS, capital gains, LTCG, STCG, indexation, redemption, "
                "secondary market sale, and capital loss set-off."
            ),
            "selected_chunks": [],
        })

    selected_chunks = retrieve_chunks(user_message)

    if not selected_chunks:
        return jsonify({
            "reply": build_local_fallback_answer([]),
            "selected_chunks": [],
        })

    try:
        answer, done_reason = call_ollama_cloud(user_message, selected_chunks)

        if looks_like_bad_model_answer(answer) or looks_incomplete_answer(answer, done_reason):
            answer = build_local_fallback_answer(selected_chunks)

    except requests.exceptions.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else "unknown"
        answer = (
            "Sir/Madam, I could not get a response from the model provider.\n\n"
            f"Provider HTTP status: {status_code}\n\n"
            "Please check the OLLAMA_API_KEY, OLLAMA_MODEL, and OLLAMA_HOST environment variables in Render."
        )
    except Exception as error:
        answer = (
            "Sir/Madam, I could not connect to the model provider.\n\n"
            f"Reason: {str(error)}"
        )

    debug_chunks = [
        {
            "id": chunk.get("id"),
            "title": chunk.get("title"),
            "category": chunk.get("category"),
            "score": chunk.get("score"),
        }
        for chunk in selected_chunks
    ]

    return jsonify({"reply": answer, "selected_chunks": debug_chunks})


@app.route("/api/chunks", methods=["GET"])
def list_chunks():
    return jsonify({
        "count": len(TAX_CHUNKS),
        "chunks": [
            {
                "id": chunk.get("id"),
                "title": chunk.get("title"),
                "category": chunk.get("category"),
            }
            for chunk in TAX_CHUNKS
        ],
    })


@app.route("/api/reset", methods=["POST"])
@app.route("/reset", methods=["POST"])
def reset():
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"message": "Chat reset successfully."})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "provider": "ollama_cloud",
        "model": MODEL_NAME,
        "ollama_host": OLLAMA_HOST,
        "api_key_configured": bool(OLLAMA_API_KEY),
        "chunks_loaded": len(TAX_CHUNKS),
        "port": PORT,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
