"""
Structured64 Agent — FastAPI backend
Proxies chat requests to Ollama Cloud API (OpenAI-compatible format).
"""
import os
import time
import json
import re
import datetime
from collections import defaultdict
from pathlib import Path
from threading import Lock

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Configuration from environment ---
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
MODEL = "glm-5.3-flash"

# Load system prompt from file; sites can request a persona prompt by key
SYSTEM_PROMPT = Path(__file__).parent.joinpath("system_prompt.txt").read_text()

_PROMPTS_DIR = Path(__file__).parent
_SITE_PROMPTS = {
    "shadeworks": _PROMPTS_DIR.joinpath("shadeworks_prompt.txt"),
    "lumiere": _PROMPTS_DIR.joinpath("lumiere_prompt.txt"),
    "climatecare": _PROMPTS_DIR.joinpath("climatecare_prompt.txt"),
}
SITE_PROMPTS = {
    "default": SYSTEM_PROMPT,
    **{k: p.read_text() for k, p in _SITE_PROMPTS.items() if p.exists()},
}

# --- Lead persistence ---
LEADS_FILE = Path(os.environ.get("LEADS_FILE", "/data/leads.jsonl"))
_leads_lock = Lock()


def save_lead(session_id: str, lead: dict, message: str, booking: bool):
    """Append a lead record to the JSONL file. One line per lead event."""
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "session_id": session_id,
        "lead": lead,
        "message": message[:500],
        "booking_suggested": booking,
    }
    LEADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _leads_lock:
        with open(LEADS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")


# --- Rate limiting ---
MAX_MESSAGES_PER_SESSION = 20
_session_counts: dict[str, list[float]] = defaultdict(list)
_session_lock = Lock()
RATE_WINDOW_SECONDS = 3600  # messages count within 1 hour


def check_rate_limit(session_id: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    with _session_lock:
        timestamps = _session_counts[session_id]
        # Prune old timestamps
        _session_counts[session_id] = [t for t in timestamps if now - t < RATE_WINDOW_SECONDS]
        if len(_session_counts[session_id]) >= MAX_MESSAGES_PER_SESSION:
            return False
        _session_counts[session_id].append(now)
        return True


# --- CORS ---
ALLOWED_ORIGINS = [
    "https://structured64.site",
    "https://www.structured64.site",
    "https://srv1188665.hstgr.cloud",
    "https://lumieremedspa.srv1188665.hstgr.cloud",
    "https://climatecarejacksonville.srv1188665.hstgr.cloud",
    "https://shadeworks.srv1188665.hstgr.cloud",
    "https://shadeworkspremium.srv1188665.hstgr.cloud",
    "http://localhost:*",
    "http://127.0.0.1:*",
]

app = FastAPI(title="Structured64 Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --- Models ---
class ChatMessage(BaseModel):
    role: str
    content: str


class LeadInfo(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    business: str | None = None


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    lead_info: LeadInfo | None = None
    session_id: str | None = None
    site: str | None = None


class ChatResponse(BaseModel):
    response: str
    lead_capture: dict | None = None
    booking_suggested: bool = False


# --- Ollama Cloud call ---
async def call_ollama(messages: list[dict]) -> str:
    """Call Ollama Cloud API (OpenAI-compatible) and return the assistant response."""
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 512,
    }
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama API error: {resp.status_code}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# --- Lead detection heuristic ---
LEAD_PATTERNS = {
    "name": [
        r"(?:my name is|i'm|i am|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:name)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ],
    "email": [
        r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
    ],
    "phone": [
        r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
        r"(\+\d{1,2}[\s-]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
    ],
    "business": [
        r"(?:my (?:business|company) is (?:called )?)([A-Z][\w\s&.,'-]+)",
        r"(?:i (?:own|run|operate))\s+(?:a|an)\s+([\w\s&.,'-]+)",
        r"(?:business name)[:\s]+([\w\s&.,'-]+)",
    ],
}

BOOKING_KEYWORDS = [
    "book", "book a call", "book a consultation", "schedule", "talk to a human",
    "talk to someone", "speak to someone", "get started", "consultation",
    "ready to move forward", "sign up", "let's do it", "let's go",
]


def extract_lead_info(message: str, existing: LeadInfo | None) -> dict:
    """Extract lead info from the user's message using regex patterns."""
    found = {}
    for field, patterns in LEAD_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                found[field] = match.group(1).strip()
                break
    return found if found else None


def check_booking_intent(message: str) -> bool:
    """Check if the user's message suggests they want to book."""
    msg_lower = message.lower()
    return any(keyword in msg_lower for keyword in BOOKING_KEYWORDS)


# --- Endpoints ---
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Rate limiting
    session_id = req.session_id or "anonymous"
    if not check_rate_limit(session_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 20 messages per session.")

    # Detect lead info and booking intent
    new_lead = extract_lead_info(req.message, req.lead_info)
    booking_suggested = check_booking_intent(req.message)

    # Merge lead info
    merged_lead = {}
    if req.lead_info:
        merged_lead.update(req.lead_info.model_dump(exclude_none=True))
    if new_lead:
        merged_lead.update(new_lead)

    # Build conversation for Ollama — persona prompt selected by site key
    messages = [{"role": "system", "content": SITE_PROMPTS.get(req.site or "default", SYSTEM_PROMPT)}]

    # Add context about captured lead info so the AI knows what it has
    if merged_lead:
        lead_context = f"[Lead info captured so far: {json.dumps(merged_lead)}]"
        messages.append({"role": "system", "content": lead_context})

    # Add conversation history
    for msg in req.history:
        messages.append({"role": msg.role, "content": msg.content})

    # Add current message
    messages.append({"role": "user", "content": req.message})

    # Call Ollama
    try:
        response_text = await call_ollama(messages)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    # Check if the AI response itself suggests booking
    response_lower = response_text.lower()
    if any(kw in response_lower for kw in ["book a call", "book a consultation", "schedule a call", "consultation"]):
        booking_suggested = True

    lead_capture = merged_lead if merged_lead else None

    # Persist lead data to disk
    if lead_capture:
        save_lead(session_id, lead_capture, req.message, booking_suggested)

    return ChatResponse(
        response=response_text,
        lead_capture=lead_capture,
        booking_suggested=booking_suggested,
    )


@app.options("/api/chat")
async def chat_options():
    return JSONResponse(content={"ok": True})