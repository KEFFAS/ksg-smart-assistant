#!/usr/bin/env python3

import os
import json
import re
import datetime
from pathlib import Path

import numpy as np
import gradio as gr
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import OpenAI
import pdfplumber

# ---------------------------
# ENV
# ---------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-nano-2025-08-07")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# PATHS
# ---------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except:
    BASE_DIR = Path.cwd()

DOCS_DIR = BASE_DIR / "docs" if (BASE_DIR / "docs").exists() else BASE_DIR
INDEX_FILE = BASE_DIR / "index_chunks.json"

# ---------------------------
# DB
# ---------------------------
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    with get_conn() as conn, conn.cursor() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT,
            email TEXT UNIQUE,
            is_admin BOOLEAN DEFAULT FALSE)""")

        c.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

        c.execute("""CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            session_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

init_db()

# ---------------------------
# EMBEDDING HELPERS
# ---------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------------------
# SAFE CHUNKING
# ---------------------------
def split_text(text, max_chars=1000):
    chunks = []
    current = ""

    for paragraph in text.split("\n"):
        if len(current) + len(paragraph) < max_chars:
            current += " " + paragraph
        else:
            chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks

# ---------------------------
# INDEXING
# ---------------------------
def build_index():
    chunks = []
    print("🔄 Building embeddings index...")

    for pdf in DOCS_DIR.glob("*.pdf"):
        try:
            with pdfplumber.open(pdf) as pdf_file:
                text = "\n".join(page.extract_text() or "" for page in pdf_file.pages)

            small_chunks = split_text(text)

            for i, chunk in enumerate(small_chunks):
                if len(chunk.strip()) < 40:
                    continue

                try:
                    emb = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk
                    ).data[0].embedding

                    chunks.append({
                        "id": f"{pdf.name}-{i}",
                        "text": chunk,
                        "source": pdf.name,
                        "embedding": emb
                    })

                except Exception as e:
                    print("Embedding skip:", e)

        except Exception as e:
            print("PDF error:", pdf, e)

    INDEX_FILE.write_text(json.dumps(chunks))
    print("✅ Index built:", len(chunks))
    return chunks

def load_index():
    if not INDEX_FILE.exists():
        return build_index()

    data = json.loads(INDEX_FILE.read_text())
    if len(data) == 0:
        return build_index()
    return data

# ---------------------------
# RETRIEVE (SEMANTIC)
# ---------------------------
def retrieve(query: str, top_k=5):
    if not query.strip():
        return []

    index = load_index()

    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    scored = []

    for c in index:
        sim = cosine_similarity(q_emb, c["embedding"])
        scored.append((sim, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [c for _, c in scored[:top_k]]

# ---------------------------
# LLM
# ---------------------------
def ask_llm(question, context):
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    if context:
        ctx = "\n\n".join(f"{c['source']}:\n{c['text']}" for c in context)
        prompt = f"""
Today's date: {today}

Context:
{ctx}

Question:
{question}

Rules:
- Do not present past dates as upcoming
"""
    else:
        prompt = f"""
You are a Kenya School of Government assistant.

Today's date: {today}

Question:
{question}
"""

    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Professional assistant"},
            {"role": "user", "content": prompt}
        ]
    )

    return res.choices[0].message.content.strip()

# ---------------------------
# DB HELPERS
# ---------------------------
def save_message(session_id, role, content):
    with get_conn() as conn, conn.cursor() as c:
        c.execute("INSERT INTO messages (session_id, role, content) VALUES (%s,%s,%s)",
                  (session_id, role, content))

def get_messages(session_id):
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as c:
        c.execute("SELECT role, content FROM messages WHERE session_id=%s ORDER BY id",
                  (session_id,))
        rows = c.fetchall()

    return [{"role": r["role"], "content": r["content"]} for r in rows]

def ensure_user(username, email):
    with get_conn() as conn, conn.cursor() as c:
        c.execute("SELECT id,is_admin FROM users WHERE email=%s", (email,))
        row = c.fetchone()
        if row:
            return row
        c.execute("INSERT INTO users(username,email) VALUES (%s,%s) RETURNING id",
                  (username, email))
        return c.fetchone()[0], False

def create_session(user_id):
    with get_conn() as conn, conn.cursor() as c:
        c.execute("INSERT INTO sessions(user_id) VALUES (%s) RETURNING id", (user_id,))
        return c.fetchone()[0]

# 🔥 ADMIN DATA (UPGRADED)
def list_all_sessions():
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as c:
        c.execute("""
        SELECT 
            s.id,
            u.username,
            u.email,
            (
                SELECT content FROM messages 
                WHERE session_id = s.id AND role='user'
                ORDER BY id DESC LIMIT 1
            ) as last_user_message,
            (
                SELECT content FROM messages 
                WHERE session_id = s.id AND role='assistant'
                ORDER BY id DESC LIMIT 1
            ) as last_ai_message
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        ORDER BY s.id DESC
        """)
        return c.fetchall()

# ---------------------------
# CHAT
# ---------------------------
def send_message(state_json, history_json, text, sid_override=None):
    if not text.strip():
        return json.loads(history_json), ""

    state = json.loads(state_json)
    sid = sid_override or state["session_id"]

    role = "admin" if state.get("is_admin") else "user"

    save_message(sid, role, text)

    context = retrieve(text)
    answer = ask_llm(text, context)

    save_message(sid, "assistant", answer)

    return get_messages(sid), ""

# ---------------------------
# ADMIN FUNCTIONS
# ---------------------------
def list_sessions_text():
    sessions = list_all_sessions()

    output = []
    for s in sessions:
        output.append(
            f"""
Session ID: {s['id']}
User: {s['username']} ({s['email']})

Last Question:
{s['last_user_message'] or 'N/A'}

AI Response:
{s['last_ai_message'] or 'N/A'}
----------------------------------------
"""
        )

    return "\n".join(output)

def load_session(sid):
    if not sid.strip():
        return "❗ Enter Session ID", []

    try:
        sid = int(sid)
        msgs = get_messages(sid)
        return f"Loaded {sid} ({len(msgs)} messages)", msgs
    except:
        return "❗ Invalid Session ID", []

def admin_send(state, history, msg, sid):
    if not sid.strip():
        return [], "❗ Enter Session ID"

    try:
        sid = int(sid)
    except:
        return [], "❗ Invalid Session ID"

    if not msg.strip():
        return get_messages(sid), ""

    save_message(sid, "admin", msg)

    context = retrieve(msg)
    answer = ask_llm(msg, context)

    save_message(sid, "assistant", answer)

    return get_messages(sid), ""

# ---------------------------
# LOGIN
# ---------------------------
def login(name, email):
    uid, is_admin = ensure_user(name, email)
    sid = create_session(uid)

    state = json.dumps({
        "user_id": uid,
        "session_id": sid,
        "is_admin": is_admin
    })

    return (
        gr.update(visible=False),
        gr.update(visible=not is_admin),
        gr.update(visible=is_admin),
        f"Welcome {name}",
        state,
        json.dumps([])
    )

# ---------------------------
# UI (KSG SMART ASSISTANT - FULL)
# ---------------------------
custom_css = """
body {
    background-color: #f5f5f5;
    font-family: Arial, sans-serif;
}

/* Header */
#header {
    background: linear-gradient(90deg, #000000, #5a3e1b);
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 15px;
    font-size: 26px;
    font-weight: bold;
}

#header span {
    color: #CBD300;
}

/* Panels */
.panel {
    background: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

/* Buttons */
.gr-button {
    background-color: #CBD300 !important;
    color: black !important;
    border-radius: 8px !important;
    font-weight: bold;
}

.gr-button:hover {
    background-color: #a8b000 !important;
}

/* Chat */
.chatbot {
    border-radius: 12px !important;
}

/* Admin */
.admin-box {
    border-left: 5px solid #5a3e1b;
    padding: 10px;
}
"""

with gr.Blocks(css=custom_css) as app:

    # ---------------- HEADER ----------------
    gr.HTML(
        "<div id='header'>KSG <span>SMART ASSISTANT</span></div>"
    )

    login_panel = gr.Column(visible=True)
    user_panel = gr.Column(visible=False)
    admin_panel = gr.Column(visible=False)

    # ---------------- LOGIN ----------------
    with login_panel:
        with gr.Column(elem_classes="panel"):
            gr.Markdown("### 🔐 Login")

            name = gr.Textbox(label="Full Name", placeholder="Enter your name")
            email = gr.Textbox(label="Email", placeholder="Enter your email")

            login_btn = gr.Button("Login")
            login_msg = gr.Markdown()

    # ---------------- USER PANEL ----------------
    with user_panel:
        with gr.Column(elem_classes="panel"):
            welcome = gr.Markdown("### 👤 Welcome",{name})
            
            chat = gr.Chatbot(type="messages", height=300)

            with gr.Row():
                msg = gr.Textbox(placeholder="Type your message here...", scale=4)
                send = gr.Button("Send", scale=1)

    # ---------------- ADMIN PANEL ----------------
    with admin_panel:
        with gr.Column(elem_classes="panel"):
            gr.Markdown("### 🧑‍💼 Admin Dashboard")

            with gr.Row():
                list_btn = gr.Button("📂 List Sessions")
                load_btn = gr.Button("📥 Load Session")

            sessions_out = gr.Textbox(
                lines=15,
                label="Session Overview"
            )

            sid_input = gr.Textbox(
                label="Session ID",
                placeholder="Enter session ID"
            )

            admin_chat = gr.Chatbot(type="messages", height=350)

            with gr.Row():
                admin_msg = gr.Textbox(
                    placeholder="Type admin reply...",
                    scale=4
                )
                admin_send_btn = gr.Button("Reply", scale=1)

    # ---------------- STATE ----------------
    state = gr.State("")
    history = gr.State(json.dumps([]))

    # ---------------- ACTIONS ----------------
    login_btn.click(
        login,
        [name, email],
        [login_panel, user_panel, admin_panel, login_msg, state, history]
    )

    send.click(
        send_message,
        [state, history, msg],
        [chat, msg]
    )

    list_btn.click(
        list_sessions_text,
        outputs=sessions_out
    )

    load_btn.click(
        load_session,
        inputs=[sid_input],
        outputs=[sessions_out, admin_chat]
    )

    admin_send_btn.click(
        admin_send,
        inputs=[state, history, admin_msg, sid_input],
        outputs=[admin_chat, admin_msg]
    )
# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    build_index()
    app.launch(server_name="0.0.0.0", server_port=10000)