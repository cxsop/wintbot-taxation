import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, jsonify, render_template, request


# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------
app = Flask(__name__)
app.url_map.strict_slashes = False

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "chunks.json"

APP_NAME = "Bond Taxation RAG Chatbot"
APP_VERSION = "2026.05.20"

# Render will inject PORT. Local fallback is 1707.
PORT = int(os.environ.get("PORT", "1707"))

# Ollama Cloud config. Keep API key only in deployment environment variables.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com").rstrip("/")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")

# Retrieval limits. These are tuned for a small KB and accurate support answers.
TOP_K = int(os.environ.get("TOP_K", "5"))
MIN_RETRIEVAL_SCORE = float(os.environ.get("MIN_RETRIEVAL_SCORE", "4"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "11000"))

# Generation limits.
GEN_TIMEOUT_SECONDS = int(os.environ.get("GEN_TIMEOUT_SECONDS", "240"))
GEN_NUM_PREDICT = int(os.environ.get("GEN_NUM_PREDICT", "900"))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.05"))


# ------------------------------------------------------------
# Knowledge base loading
# ------------------------------------------------------------
def load_kb() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {CHUNKS_PATH}")

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and isinstance(data.get("chunks"), list):
        meta = {key: value for key, value in data.items() if key != "chunks"}
        chunks = data["chunks"]
    elif isinstance(data, list):
        meta = {"kb_name": "Bond Taxation Knowledge Base"}
        chunks = data
    else:
        raise ValueError("chunks.json must be either a list of chunks or an object with a 'chunks' list.")

    valid_chunks: List[Dict[str, Any]] = []
    for chunk in chunks:
        if isinstance(chunk, dict) and chunk.get("id") and chunk.get("facts"):
            valid_chunks.append(chunk)

    if not valid_chunks:
        raise ValueError("No valid chunks found. Each chunk must have at least 'id' and 'facts'.")

    return meta, valid_chunks


KB_META, KB_CHUNKS = load_kb()


# ------------------------------------------------------------
# Retrieval helpers
# ------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "i", "me", "my", "mine", "you", "your", "yours", "we", "our", "ours",
    "to", "of", "for", "from", "in", "on", "at", "by", "with", "without", "into",
    "and", "or", "but", "if", "then", "than", "so", "because", "as", "it", "this",
    "that", "these", "those", "there", "here", "please", "pls", "kindly",
    "what", "why", "how", "when", "where", "which", "who", "whom",
    "can", "could", "should", "would", "do", "does", "did", "done", "will", "shall",
    "tell", "explain", "answer", "query", "question", "user", "customer", "client",
    "sir", "madam", "maam", "mam", "about", "regarding", "related"
}

DOMAIN_SYNONYMS = {
    "ncd": ["bond", "debenture", "listed bond", "listed ncd"],
    "bond": ["ncd", "debenture", "listed ncd", "listed bond"],
    "debenture": ["bond", "ncd", "listed ncd"],
    "interest": ["coupon", "payout", "repayment", "income from other sources"],
    "coupon": ["interest", "interest payout", "income from other sources"],
    "tds": ["tax deducted", "tax credit", "form 168", "form 26as", "section 193"],
    "form 26as": ["form 168", "tax credit statement", "tds statement"],
    "26as": ["form 168", "tax credit statement"],
    "form 168": ["form 26as", "tax credit statement", "tds statement"],
    "ais": ["annual information statement", "tax department"],
    "sell": ["sold", "selling", "sale", "secondary market", "exit", "before maturity"],
    "sale": ["sell", "sold", "selling", "secondary market", "exit"],
    "sold": ["sell", "sale", "selling", "secondary market"],
    "maturity": ["redemption", "redeemed", "matured", "face value", "principal repayment"],
    "redemption": ["maturity", "redeemed", "face value", "principal repayment"],
    "ltcg": ["long term", "long-term", "more than 12 months", "12.5%"],
    "stcg": ["short term", "short-term", "less than 12 months", "slab rate"],
    "loss": ["capital loss", "set off", "set-off", "carry forward", "stcl", "ltcl"],
    "gain": ["capital gain", "profit", "ltcg", "stcg"],
    "premium": ["above face value", "capital loss", "acquisition cost"],
    "discount": ["below face value", "capital gain", "face value"],
    "report": ["taxation report", "master report", "repayment summary", "tds summary"],
    "itr": ["income tax return", "tax filing", "filing"],
    "form 121": ["form 15g", "form 15h", "declaration", "tds exemption"],
    "15g": ["form 121", "form 15h", "declaration"],
    "15h": ["form 121", "form 15g", "declaration"]
}

INTENT_BOOSTS = [
    {
        "name": "interest_taxation",
        "needles": ["interest", "coupon", "accrued", "income from other sources", "payout"],
        "tags": ["interest taxation", "accrued interest", "coupon", "income from other sources"]
    },
    {
        "name": "tds_reflection",
        "needles": ["tds", "not showing", "not visible", "missing", "form 168", "26as", "form 26as", "ais", "mismatch", "reflect"],
        "tags": ["tds lifecycle", "form 168", "form 26as", "reflection delay", "tds missing", "reconciliation"]
    },
    {
        "name": "sale_capital_gain",
        "needles": ["sell", "sale", "sold", "selling", "before maturity", "secondary market", "exit"],
        "tags": ["sale", "secondary market", "capital gain", "capital loss", "sell before maturity"]
    },
    {
        "name": "holding_period",
        "needles": ["ltcg", "stcg", "holding period", "12 months", "long term", "short term", "slab", "12.5"],
        "tags": ["holding period", "ltcg", "stcg", "tax rate", "12 months"]
    },
    {
        "name": "maturity_redemption",
        "needles": ["maturity", "matured", "redeem", "redemption", "face value", "principal"],
        "tags": ["maturity", "redemption", "face value", "principal repayment"]
    },
    {
        "name": "loss_setoff",
        "needles": ["loss", "set off", "set-off", "carry forward", "stcl", "ltcl", "adjust"],
        "tags": ["capital loss", "set off", "carry forward", "stcl", "ltcl"]
    },
    {
        "name": "tax_reports",
        "needles": ["master report", "taxation report", "repayment summary", "tds summary", "bonds sold", "maturity details", "holding statement"],
        "tags": ["master report", "taxation report", "repayment summary", "tax report tabs", "filing mapping"]
    },
    {
        "name": "form121",
        "needles": ["form 121", "15g", "15h", "declaration"],
        "tags": ["form 121", "form 15g", "form 15h", "tds exemption declaration"]
    }
]


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = text.replace("₹", " rs ")
    text = text.replace("%", " percent ")
    text = text.replace("&", " and ")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9\.\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: Any) -> List[str]:
    cleaned = normalize_text(text)
    tokens = cleaned.split()
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def expand_query(query: str) -> str:
    normalized = normalize_text(query)
    expanded_terms = [normalized]

    for key, values in DOMAIN_SYNONYMS.items():
        if key in normalized:
            expanded_terms.extend(values)

    return " ".join(expanded_terms)


def chunk_search_text(chunk: Dict[str, Any]) -> str:
    parts: List[str] = []
    for field in ["id", "section", "support_answer"]:
        parts.append(str(chunk.get(field, "")))

    for field in ["intent_tags", "keywords", "question_patterns", "facts"]:
        value = chunk.get(field, [])
        if isinstance(value, list):
            parts.append(" ".join(str(item) for item in value))
        else:
            parts.append(str(value))

    return normalize_text(" ".join(parts))


def detect_intents(query: str) -> List[str]:
    q = normalize_text(query)
    detected: List[str] = []

    for intent in INTENT_BOOSTS:
        if any(normalize_text(needle) in q for needle in intent["needles"]):
            detected.append(intent["name"])

    return detected


def score_chunk(query: str, chunk: Dict[str, Any]) -> float:
    expanded = expand_query(query)
    q_norm = normalize_text(expanded)
    q_tokens = set(tokenize(expanded))

    c_text = chunk_search_text(chunk)
    c_tokens = set(tokenize(c_text))

    score = 0.0

    # BM25-inspired lightweight lexical score for small KB.
    overlap = q_tokens.intersection(c_tokens)
    score += len(overlap) * 2.0

    # Exact phrase / keyword boosts.
    for keyword in chunk.get("keywords", []):
        k_norm = normalize_text(keyword)
        if not k_norm:
            continue
        if k_norm in q_norm:
            score += 12.0
        k_tokens = set(tokenize(k_norm))
        if k_tokens and k_tokens.issubset(q_tokens):
            score += 5.0

    # Question-pattern similarity.
    for pattern in chunk.get("question_patterns", []):
        p_tokens = set(tokenize(pattern))
        if not p_tokens:
            continue
        common = q_tokens.intersection(p_tokens)
        score += min(len(common) * 3.0, 12.0)

    # Section/tag overlap.
    section_tokens = set(tokenize(chunk.get("section", "")))
    tag_tokens = set(tokenize(" ".join(chunk.get("intent_tags", []))))
    score += len(q_tokens.intersection(section_tokens)) * 4.0
    score += len(q_tokens.intersection(tag_tokens)) * 4.0

    # Intent-based boosting.
    q_clean = normalize_text(query)
    for intent in INTENT_BOOSTS:
        intent_present = any(normalize_text(needle) in q_clean for needle in intent["needles"])
        if not intent_present:
            continue

        chunk_tags = normalize_text(" ".join(chunk.get("intent_tags", [])))
        chunk_section = normalize_text(chunk.get("section", ""))
        chunk_keywords = normalize_text(" ".join(chunk.get("keywords", [])))
        chunk_all = f"{chunk_tags} {chunk_section} {chunk_keywords}"

        if any(normalize_text(tag) in chunk_all for tag in intent["tags"]):
            score += 18.0

    # Specific high-signal boosts.
    if "april" in q_clean and ("not visible" in q_clean or "not showing" in q_clean or "missing" in q_clean):
        if chunk.get("id") in {"BTX-017", "BTX-030", "BTX-019", "BTX-012"}:
            score += 30.0

    if "form 121" in q_clean or "15g" in q_clean or "15h" in q_clean:
        if chunk.get("id") in {"BTX-015", "BTX-033"}:
            score += 30.0

    if "gross" in q_clean or "net" in q_clean:
        if chunk.get("id") in {"BTX-004", "BTX-026"}:
            score += 25.0

    if "premium" in q_clean:
        if chunk.get("id") in {"BTX-009", "BTX-034"}:
            score += 30.0

    if "discount" in q_clean:
        if chunk.get("id") in {"BTX-010"}:
            score += 30.0

    return score


def retrieve_chunks(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    start = time.time()
    scored = []

    for chunk in KB_CHUNKS:
        score = score_chunk(query, chunk)
        scored.append({"score": score, "chunk": chunk})

    scored.sort(key=lambda item: item["score"], reverse=True)

    selected: List[Dict[str, Any]] = []
    used_chars = 0

    for item in scored:
        if item["score"] < MIN_RETRIEVAL_SCORE:
            continue

        chunk = dict(item["chunk"])
        chunk["score"] = round(item["score"], 2)
        chunk_text = json.dumps(chunk, ensure_ascii=False)

        if used_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
            continue

        selected.append(chunk)
        used_chars += len(chunk_text)

        if len(selected) >= top_k:
            break

    # If best score is strong but selected is empty due to a high threshold, allow one best chunk.
    if not selected and scored and scored[0]["score"] > 0:
        chunk = dict(scored[0]["chunk"])
        chunk["score"] = round(scored[0]["score"], 2)
        selected.append(chunk)

    best_score = selected[0]["score"] if selected else 0
    confidence = "high" if best_score >= 35 else "medium" if best_score >= 15 else "low"

    return {
        "chunks": selected,
        "confidence": confidence,
        "best_score": best_score,
        "intents": detect_intents(query),
        "elapsed_ms": round((time.time() - start) * 1000, 2)
    }


# ------------------------------------------------------------
# Answer construction
# ------------------------------------------------------------
def build_context(chunks: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        facts = "\n".join(f"- {fact}" for fact in chunk.get("facts", []))
        block = f"""
[CHUNK {index}]
ID: {chunk.get('id')}
SECTION: {chunk.get('section')}
SOURCE PAGES: {chunk.get('source_pages')}
INTENT TAGS: {', '.join(chunk.get('intent_tags', []))}
FACTS:
{facts}
SUPPORT-READY ANSWER:
{chunk.get('support_answer', '')}
""".strip()
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def deterministic_answer(query: str, chunks: List[Dict[str, Any]], confidence: str = "medium") -> str:
    if not chunks:
        return (
            "Sir/Madam, I can answer only from the uploaded bond taxation knowledge base. "
            "Please ask about bond interest taxation, TDS, Form 168/Form 26AS, AIS mismatch, "
            "capital gains/losses, maturity, premium/discount, Form 121, Master Report, or Taxation Report."
        )

    query_norm = normalize_text(query)
    chunk_ids = {chunk.get("id") for chunk in chunks}

    # Strong deterministic answers for common high-risk support questions.
    if chunk_ids.intersection({"BTX-006", "BTX-005"}) and any(term in query_norm for term in ["sell", "sold", "sale", "selling", "before maturity", "secondary"]):
        return (
            "Sir/Madam, if a listed NCD is sold before maturity, the tax impact is based on capital gain or capital loss.\n\n"
            "The calculation is:\n"
            "Sale value minus acquisition cost.\n\n"
            "If the sale value is higher than the acquisition cost, it is a capital gain. If the sale value is lower, it is a capital loss.\n\n"
            "The applicable tax treatment depends on the holding period:\n"
            "- If held for 12 months or less, it is treated as STCG and taxed as per the investor’s slab rate.\n"
            "- If held for more than 12 months, it is treated as LTCG and taxed at 12.5%.\n\n"
            "Interest received on the bond is taxed separately under Income from Other Sources. For personal tax filing treatment, the user may consult a CA or tax advisor."
        )

    if "april" in query_norm and any(term in query_norm for term in ["not visible", "not showing", "missing", "form 168", "26as"]):
        return (
            "Sir/Madam, April payouts fall under the Q1 filing cycle, which covers April to June. The issuer files the Q1 TDS return by July 31, and Form 168/Form 26AS generally reflects the TDS only after quarterly filing and processing.\n\n"
            "The platform may show TDS immediately based on the repayment date, but Form 168/Form 26AS will not update instantly. If the amount relates to March-end accrued income, the user should also check the previous financial year’s Form 168/Form 26AS for entries dated 31 March.\n\n"
            "If it is still not reflected after the expected filing timeline, the case can be checked internally."
        )

    if chunk_ids.intersection({"BTX-004", "BTX-026"}) and any(term in query_norm for term in ["gross", "net", "18000", "20000", "tds"]):
        return (
            "Sir/Madam, while filing ITR, the user should generally report the gross interest amount, not only the net amount received after TDS.\n\n"
            "For example, if interest is ₹20,000 and TDS is ₹2,000, the user may receive ₹18,000 in the bank account. However, the interest income to report is ₹20,000. The deducted ₹2,000 can be claimed as TDS credit, subject to it reflecting against the user’s PAN in Form 168/Form 26AS."
        )

    if chunk_ids.intersection({"BTX-009", "BTX-034"}) and "premium" in query_norm:
        return (
            "Sir/Madam, if a bond was purchased at a premium, the acquisition cost is higher than the face value. When the bond matures at face value, the difference can appear as capital loss.\n\n"
            "For example, if the investment amount was ₹1,01,000 and the maturity value is ₹1,00,000, the ₹1,000 difference is treated as capital loss. Interest earned on the bond is taxed separately as interest income."
        )

    if chunk_ids.intersection({"BTX-003", "BTX-031"}) and any(term in query_norm for term in ["no tds", "tds not deducted", "without tds"]):
        return (
            "Sir/Madam, TDS may not be deducted even though interest was received if the issuer is following the applicable threshold rule, generally around ₹10,000 depending on issuer type and process.\n\n"
            "If the total interest paid by that particular issuer during the financial year does not cross the applicable threshold, TDS may not be deducted. However, the interest income may still need to be reported while filing ITR."
        )

    # General fallback from selected chunks.
    lines = ["Sir/Madam, based on the uploaded bond taxation knowledge base:"]
    used_answers = []

    for chunk in chunks[:3]:
        answer = chunk.get("support_answer", "").strip()
        if answer and answer not in used_answers:
            used_answers.append(answer)

    for answer in used_answers:
        # Remove duplicate Sir/Madam if we are combining multiple snippets.
        cleaned = re.sub(r"^sir/madam,\s*", "", answer, flags=re.IGNORECASE).strip()
        lines.append(f"- {cleaned}")

    if confidence == "low":
        lines.append("\nI found only a low-confidence match in the current knowledge base, so please verify once before sharing it with the customer.")

    lines.append("\nFor personal tax filing treatment, the user may consult a CA or tax advisor.")
    return "\n".join(lines)


def is_incomplete_answer(answer: str, done_reason: str = "") -> bool:
    if not answer or not answer.strip():
        return True

    if done_reason in {"length", "limit"}:
        return True

    cleaned = answer.strip()
    lower = cleaned.lower()

    incomplete_endings = [
        "depends on", "depends on:", "based on", "based on:", "as follows", "as follows:",
        "the applicable tax rate depends on", "the tax treatment depends on", "calculated as",
        "calculated by", "will be", "it is", "the"
    ]

    if any(lower.endswith(ending) for ending in incomplete_endings):
        return True

    if cleaned[-1] not in ".!?)]}'\"":
        return True

    return False


def is_generic_or_unsafe(answer: str) -> bool:
    lower = normalize_text(answer)
    weak_phrases = [
        "find contact information for tax advisors",
        "i cannot provide tax advice",
        "consult a tax advisor"  # acceptable as ending disclaimer, but not as main answer
    ]

    if "find contact information" in lower:
        return True

    # If the answer is mostly generic disclaimer and too short, reject.
    if "consult a tax advisor" in lower and len(lower) < 450 and "sir" not in lower:
        return True

    return False


def call_ollama_cloud(query: str, chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not OLLAMA_API_KEY:
        raise RuntimeError("OLLAMA_API_KEY is not configured.")

    context = build_context(chunks)

    system_prompt = f"""
You are a strict bond taxation support assistant.

You must answer using ONLY the provided knowledge-base CONTEXT.
Your answer must be accurate, relevant, complete, and customer-support ready.
Use simple Indian English.
Address the customer as Sir/Madam.
Do not invent any law, rate, deadline, section, form, or tax rule beyond the CONTEXT.
Do not give a generic tax-advisor-only answer.
Do not ask if the user wants help finding tax advisors.

Formatting rules:
- Give the direct answer first.
- Use short paragraphs or bullets if helpful.
- For sale/capital gain questions, include formula, holding period, and tax treatment if present in context.
- For TDS/Form 168 mismatch questions, include filing timeline and reconciliation steps if present in context.
- End with this short disclaimer only when useful: "For personal tax filing treatment, you may consult a CA or tax advisor."

Knowledge base name: {KB_META.get('kb_name', 'Bond Taxation Knowledge Base')}
""".strip()

    user_prompt = f"""
CONTEXT:
{context}

USER QUESTION:
{query}

Answer the USER QUESTION using only the CONTEXT. Do not stop mid-sentence.
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": GEN_TEMPERATURE,
            "num_predict": GEN_NUM_PREDICT
        }
    }

    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        headers=headers,
        json=payload,
        timeout=GEN_TIMEOUT_SECONDS
    )
    response.raise_for_status()

    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()
    done_reason = data.get("done_reason", "")

    return answer, done_reason


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", app_name=APP_NAME)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("message", "")).strip()

    if not query:
        return jsonify({
            "reply": "Please enter a bond taxation question.",
            "selected_chunks": [],
            "confidence": "none"
        }), 400

    if normalize_text(query) in {"hi", "hello", "hey", "test"}:
        return jsonify({
            "reply": "Hello Sir/Madam. I can help with bond taxation queries such as interest taxation, TDS, Form 168/Form 26AS, capital gains/losses, maturity, premium/discount, Form 121, Master Report, and Taxation Report mapping.",
            "selected_chunks": [],
            "confidence": "none"
        })

    retrieval = retrieve_chunks(query)
    chunks = retrieval["chunks"]

    if not chunks:
        return jsonify({
            "reply": deterministic_answer(query, [], "low"),
            "selected_chunks": [],
            "confidence": "low",
            "intents": retrieval["intents"]
        })

    provider = "ollama_cloud"
    used_fallback = False
    error_message = None

    try:
        answer, done_reason = call_ollama_cloud(query, chunks)
        if is_incomplete_answer(answer, done_reason) or is_generic_or_unsafe(answer):
            used_fallback = True
            answer = deterministic_answer(query, chunks, retrieval["confidence"])
    except Exception as error:
        used_fallback = True
        error_message = str(error)
        answer = deterministic_answer(query, chunks, retrieval["confidence"])

    selected_chunks = [
        {
            "id": chunk.get("id"),
            "section": chunk.get("section"),
            "score": chunk.get("score"),
            "source_pages": chunk.get("source_pages", []),
            "intent_tags": chunk.get("intent_tags", [])
        }
        for chunk in chunks
    ]

    return jsonify({
        "reply": answer,
        "selected_chunks": selected_chunks,
        "confidence": retrieval["confidence"],
        "best_score": retrieval["best_score"],
        "intents": retrieval["intents"],
        "provider": provider,
        "used_fallback": used_fallback,
        "error": error_message,
        "retrieval_ms": retrieval["elapsed_ms"]
    })


@app.route("/api/chunks")
def api_chunks():
    return jsonify({
        "kb": KB_META,
        "count": len(KB_CHUNKS),
        "chunks": [
            {
                "id": chunk.get("id"),
                "section": chunk.get("section"),
                "source_pages": chunk.get("source_pages"),
                "intent_tags": chunk.get("intent_tags")
            }
            for chunk in KB_CHUNKS
        ]
    })


@app.route("/api/search", methods=["POST"])
def api_search():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    retrieval = retrieve_chunks(query)
    return jsonify(retrieval)


@app.route("/health")
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "provider": "ollama_cloud",
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "api_key_loaded": bool(OLLAMA_API_KEY),
        "chunks_loaded": len(KB_CHUNKS),
        "strict_slashes": app.url_map.strict_slashes
    })


@app.route("/<path:path>")
def fallback(path: str):
    if path.startswith("api/"):
        return jsonify({"error": "API route not found", "path": path}), 404
    return render_template("index.html", app_name=APP_NAME)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
