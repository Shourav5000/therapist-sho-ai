from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv

# LangChain
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# RAG
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Typing
from typing import TypedDict, Annotated, List
import operator
import tempfile

load_dotenv()

app = Flask(__name__)
CORS(app)

print("🌿 Starting Therapist@Sho.ai...")

# ─────────────────────────────────────────────────────────────
# LLM — Claude via LangChain
# FIXED: updated model string to claude-sonnet-4-5
# ─────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_tokens=1024,
    streaming=True
)

# Quick LLM for fast tasks (sentiment, insights)
quick_llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_tokens=150
)

# ─────────────────────────────────────────────────────────────
# SENTIMENT — Claude-powered (no PyTorch needed)
# ─────────────────────────────────────────────────────────────
EMOTION_MAP = {
    "joy":      {"label": "Happy",     "emoji": "😊", "tone": "celebratory and warm"},
    "sadness":  {"label": "Sad",       "emoji": "😢", "tone": "extra gentle and nurturing"},
    "anger":    {"label": "Angry",     "emoji": "😤", "tone": "calm, validating, and de-escalating"},
    "fear":     {"label": "Anxious",   "emoji": "😰", "tone": "reassuring and grounding"},
    "surprise": {"label": "Surprised", "emoji": "😲", "tone": "curious and supportive"},
    "disgust":  {"label": "Disgusted", "emoji": "😣", "tone": "validating and empathetic"},
    "neutral":  {"label": "Neutral",   "emoji": "😐", "tone": "warm and inviting"},
}

def detect_emotion(text: str) -> str:
    """Use Claude to detect emotion — no PyTorch needed"""
    try:
        prompt = f"""Classify the primary emotion in this message into exactly one word.
Choose from: joy, sadness, anger, fear, surprise, disgust, neutral
Message: "{text[:300]}"
Respond with only one word, lowercase."""
        response = quick_llm.invoke([HumanMessage(content=prompt)])
        emotion = response.content.strip().lower()
        return emotion if emotion in EMOTION_MAP else "neutral"
    except Exception:
        return "neutral"

# ─────────────────────────────────────────────────────────────
# RAG — HuggingFace Embeddings + ChromaDB
# ─────────────────────────────────────────────────────────────
print("📚 Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
session_stores = {}

# ─────────────────────────────────────────────────────────────
# LANGCHAIN MEMORY — per session
# ─────────────────────────────────────────────────────────────
session_memories = {}

def get_memory(session_id: str) -> ChatMessageHistory:
    if session_id not in session_memories:
        session_memories[session_id] = ChatMessageHistory()
    return session_memories[session_id]

# ─────────────────────────────────────────────────────────────
# LANGGRAPH TOOLS
# ─────────────────────────────────────────────────────────────
@tool
def get_breathing_exercise(technique: str = "box") -> str:
    """Suggest a breathing exercise. Use when user is anxious, stressed, or overwhelmed.
    Options: box, 478, equal"""
    exercises = {
        "box":   "**Box Breathing (4-4-4-4):** Inhale 4 → Hold 4 → Exhale 4 → Hold 4. Repeat 4 times. Perfect for anxiety and stress.",
        "478":   "**4-7-8 Breathing:** Inhale 4 → Hold 7 → Exhale 8. Activates your parasympathetic nervous system for deep calm.",
        "equal": "**Equal Breathing (5-5):** Inhale 5 → Exhale 5. Simple and effective — great for beginners.",
    }
    return exercises.get(technique, exercises["box"])


@tool
def get_coping_strategies(issue: str) -> str:
    """Provide evidence-based coping strategies. Options: anxiety, depression, stress, anger, grief, loneliness, trauma"""
    strategies = {
        "anxiety":    "**For Anxiety:** 1) 5-4-3-2-1 grounding technique 2) Progressive muscle relaxation 3) Limit caffeine 4) Challenge anxious thoughts — ask 'Is this realistic?'",
        "depression": "**For Depression:** 1) Schedule one small enjoyable activity daily 2) Maintain routine 3) Reach out to one person today 4) Even a 10-min walk can shift mood",
        "stress":     "**For Stress:** 1) Time-box worries to 15 min/day 2) Break tasks into tiny steps 3) Practice saying no 4) Take micro-breaks every 90 minutes",
        "anger":      "**For Anger:** 1) STOP: Stop, Take a breath, Observe, Proceed 2) Physical release like a walk 3) Write an unsent letter 4) Find the hurt beneath the anger",
        "grief":      "**For Grief:** 1) Allow yourself to feel — no timeline 2) Create a ritual to honor your loss 3) Connect with others who understand 4) Be gentle on hard days",
        "loneliness": "**For Loneliness:** 1) Text one person today 2) Join a group around an interest 3) Volunteer — helping reduces isolation 4) Practice self-compassion",
        "trauma":     "**For Trauma:** 1) Ground yourself with your senses 2) Create safety routines 3) Consider EMDR or CPT therapy 4) Be patient with your nervous system",
    }
    return strategies.get(issue.lower(), f"For {issue}: Focus on self-compassion, grounding, and reaching out for support.")


@tool
def check_crisis_level(message: str) -> str:
    """Check for crisis signals. ALWAYS use when user mentions hopelessness, self-harm, or suicidal thoughts."""
    keywords = ["suicide", "kill myself", "end it", "don't want to live",
                "self-harm", "hurt myself", "no point", "give up", "can't go on"]
    is_crisis = any(k in message.lower() for k in keywords)
    if is_crisis:
        return """🆘 CRISIS DETECTED — Share these resources immediately:
• 📞 988 Suicide & Crisis Lifeline — Call or Text 988 (24/7, free, confidential)
• 💬 Crisis Text Line — Text HOME to 741741
• 🌐 SAMHSA Helpline — 1-800-662-4357
Strongly encourage the user to reach out to a trusted person or professional right now."""
    return "No immediate crisis detected. Continue supportive conversation."


@tool
def search_journal_context(query: str, session_id: str = "default") -> str:
    """Search user's uploaded journal for relevant context. Use when user references past events."""
    if session_id not in session_stores:
        return "No journal entries uploaded yet for this session."
    docs = session_stores[session_id].similarity_search(query, k=3)
    if not docs:
        return "No relevant journal entries found."
    return "Relevant journal context:\n\n" + "\n\n".join([f"• {d.page_content}" for d in docs])


# ─────────────────────────────────────────────────────────────
# LANGGRAPH AGENT
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    session_id: str
    emotion: str

tools_list = [get_breathing_exercise, get_coping_strategies, check_crisis_level, search_journal_context]
tools_node = ToolNode(tools_list)
llm_with_tools = llm.bind_tools(tools_list)

SYSTEM_PROMPT = """You are Therapist@Sho.ai — a warm, empathetic AI mental health companion built by Shourav Mandal.

Personality:
- Warm, gentle, non-judgmental
- Validate emotions BEFORE offering advice
- Speak like a trusted friend who is also a skilled therapist
- Use soft emojis occasionally (💚 💙 🌿)

Tools — use them proactively:
- get_breathing_exercise → user is anxious, stressed, or overwhelmed
- get_coping_strategies → user needs practical help with a specific issue
- check_crisis_level → ANY sign of hopelessness, self-harm, or suicidal thoughts
- search_journal_context → user references past events or personal history

Rules:
- NEVER diagnose or prescribe medication
- ALWAYS run check_crisis_level if crisis signals are present
- Keep responses to 3-5 sentences unless more detail is needed
- End with a follow-up question OR one actionable suggestion
- Remind users gently that you are not a replacement for a licensed therapist"""


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def call_model(state: AgentState):
    emotion = state.get("emotion", "neutral")
    info = EMOTION_MAP.get(emotion, EMOTION_MAP["neutral"])
    system = f"""{SYSTEM_PROMPT}

DETECTED EMOTION: {info['label']} {info['emoji']}
TONE TO USE: {info['tone']}
SESSION ID (for journal search): {state.get('session_id', 'default')}"""

    msgs = [SystemMessage(content=system)] + state["messages"]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tools_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
agent_graph = workflow.compile()

print("✅ LangGraph agent ready!")

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    session_id = data.get("session_id", "default")

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    # Claude-powered sentiment analysis
    emotion = detect_emotion(last_user_msg)
    emotion_info = EMOTION_MAP.get(emotion, EMOTION_MAP["neutral"])

    # Handle images attached to latest user message
    images = data.get("images", [])  # [{base64, mimeType}]

    # Build LangChain messages
    lc_messages = []
    for i, m in enumerate(messages):
        is_last = (i == len(messages) - 1)
        if m["role"] == "user":
            if is_last and images:
                content = []
                for img in images:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img["mimeType"],
                            "data": img["base64"]
                        }
                    })
                content.append({"type": "text", "text": m["content"] or "Please look at this image and share your thoughts about what you see."})
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    def generate():
        # Send emotion data first so frontend can display it
        yield f"data: {json.dumps({'emotion': emotion, 'emotion_label': emotion_info['label'], 'emotion_emoji': emotion_info['emoji']})}\n\n"

        try:
            state = {"messages": lc_messages, "session_id": session_id, "emotion": emotion}
            final_response = ""

            for chunk in agent_graph.stream(state, stream_mode="values"):
                if "messages" in chunk:
                    last_msg = chunk["messages"][-1]
                    # Skip tool messages and user messages — only stream final AI text
                    from langchain_core.messages import AIMessage as LCAIMessage, ToolMessage
                    if not isinstance(last_msg, LCAIMessage):
                        continue
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        continue
                    if hasattr(last_msg, "content") and last_msg.content:
                        content = last_msg.content
                        if isinstance(content, list):
                            content = " ".join(c.get("text","") for c in content if isinstance(c, dict))
                        if content and content != final_response:
                            new_text = content[len(final_response):]
                            if new_text:
                                final_response = content
                                yield f"data: {json.dumps({'text': new_text})}\n\n"

            # Save to LangChain memory
            memory = get_memory(session_id)
            memory.add_user_message(last_user_msg)
            memory.add_ai_message(final_response)

        except Exception as e:
            yield f"data: {json.dumps({'text': f'I had a small hiccup 💚 Please try again. ({str(e)[:60]})'})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/upload-journal", methods=["POST"])
def upload_journal():
    """RAG: Upload a PDF or TXT journal for the therapist to reference"""
    session_id = request.form.get("session_id", "default")
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename.lower()
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        return jsonify({"error": "Only PDF or TXT supported"}), 400

    try:
        suffix = ".pdf" if filename.endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)

        if session_id in session_stores:
            session_stores[session_id].add_documents(chunks)
        else:
            session_stores[session_id] = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=f"journal_{session_id}"
            )

        os.unlink(tmp_path)
        return jsonify({
            "success": True,
            "chunks": len(chunks),
            "message": f"Processed {len(chunks)} sections from your journal 💚"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-sentiment", methods=["POST"])
def analyze_sentiment():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    emotion = detect_emotion(text)
    info = EMOTION_MAP.get(emotion, EMOTION_MAP["neutral"])
    return jsonify({
        "emotion": emotion,
        "label": info["label"],
        "emoji": info["emoji"]
    })


@app.route("/mood-insight", methods=["POST"])
def mood_insight():
    data = request.json
    mood = data.get("mood", "")
    note = data.get("note", "")
    history = data.get("history", [])
    history_text = "Recent moods: " + ", ".join([h["label"] for h in history[-5:]]) if history else ""

    prompt = f"""User logged mood: "{mood}". Note: "{note or 'none'}". {history_text}
Give a warm 2-3 sentence insight. Acknowledge their feeling, offer one gentle suggestion, end warmly. Use one emoji."""

    response = quick_llm.invoke([HumanMessage(content=prompt)])
    return jsonify({"insight": response.content})


@app.route("/clear-memory", methods=["POST"])
def clear_memory():
    session_id = request.json.get("session_id", "default")
    if session_id in session_memories:
        del session_memories[session_id]
    return jsonify({"success": True})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "app": "Therapist@Sho.ai",
        "features": ["LangChain Memory", "LangGraph Agent", "RAG + ChromaDB", "Claude Sentiment Analysis"],
        "model": "claude-sonnet-4-5"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🌿 Therapist@Sho.ai running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
