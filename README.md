# 🌿 Therapist@Sho.ai

A private, judgment-free AI mental wellness companion powered by Claude AI, LangChain, and LangGraph.

**Built by Shourav Mandal** · [LinkedIn](https://linkedin.com/in/shourav-mandal) · [Portfolio](https://shourav.dev)

---

## Features

- 🧠 **AI Emotion Detection** — Claude detects mood from every message
- 🫁 **Breathing Exercises** — Box, 4-7-8, and Equal breathing with animated guide
- 📊 **Mood Tracker** — Log and visualize mood history with a live chart
- 🎵 **Ambient Sounds** — Rain, ocean waves, forest, fire, white noise, café
- 🔊 **Text-to-Speech** — Claude's responses read aloud
- 🎤 **Voice Input** — Speak your messages via Web Speech API
- 🖼 **Image Recognition** — Share images for Claude to analyze
- 📓 **Journal RAG** — Upload a PDF/TXT journal for Claude to reference
- 📋 **Session Summary** — AI-generated session insights
- 💾 **Export PDF** — Download your conversation
- 🆘 **Crisis Detection** — Auto-triggers emergency resources

## Tech Stack

- **Backend:** Python, Flask, LangChain, LangGraph, ChromaDB
- **AI:** Anthropic Claude Sonnet 4.5, HuggingFace Embeddings
- **Frontend:** Vanilla HTML/CSS/JS, Web Audio API, Web Speech API
- **Deployment:** Render / Docker

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/therapist-sho-ai
cd therapist-sho-ai

# 2. Create .env file
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
py therapist_api.py
# Open http://localhost:5000
```

## Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Add environment variable: `ANTHROPIC_API_KEY = your_key`
5. Deploy — done!

## Project Structure

```
therapist-sho-ai/
├── therapist_api.py   # Flask backend + LangGraph agent
├── index.html         # Full frontend (HTML/CSS/JS)
├── requirements.txt   # Python dependencies
├── render.yaml        # Render deployment config
├── Dockerfile         # Docker config
└── .env               # API keys (not committed)
```

---

> ⚠️ Not a replacement for licensed therapy. If you're in crisis, call or text **988**.
