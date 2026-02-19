# 🗺️ Concept Cartographer

**Interactive knowledge mapping — ask questions and watch concepts emerge as a graph.**

Concept Cartographer is a Gradio app that turns a chat into a growing *concept map*: it extracts key concepts and relationships from your conversation and renders them as a knowledge graph you can export.

- **Repo:** https://github.com/dagny099/concept-cartography-gradio  
- **Live demo:** https://ce668c1e9bcc6265ae.gradio.live/  
- **Built by:** Barbara Hidalgo-Sotelo (Cognitive Science + AI)  
- **LinkedIn:** https://www.linkedin.com/in/barbara-hidalgo-sotelo/

---

## What it does

As you chat, the app:

1. **Responds conversationally** to your question
2. **Extracts structured concepts + relationships** from the conversation
3. **Updates a persistent graph** across turns
4. **Visualizes the graph** in real time
5. Lets you **export the graph as JSON** for downstream use (Neo4j, notes, outlines, retrieval, etc.)

This is intentionally a “small, sharp demo” focused on *stateful* LLM tooling (not just a stateless chatbot).

---

## Quick start (local)

### Prerequisites
- Python **3.8+**
- An **OpenAI API key**

### Install
```bash
git clone https://github.com/dagny099/concept-cartography-gradio.git
cd concept-cartography-gradio

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configure environment
```bash
cp .env.example .env
# then edit .env and set: OPENAI_API_KEY=...
```

### Run
```bash
python concept_cartographer.py
```

The app starts at `http://localhost:7860`. If Gradio sharing is enabled, you’ll also see a public `gradio.live` link printed to the console.

---

## Versions

This project is currently developed and tested on **Gradio 6.6**.

If you run into UI differences across Gradio versions, pin Gradio explicitly:
```bash
pip install "gradio==6.6"
```

---

## How to use

1. Pick a **Domain** (e.g., AI/ML or Cognitive Science).  
   Domain selection nudges extraction toward domain-relevant concepts and relation types.
2. Ask a question (or click a starter prompt).
3. Watch the **Knowledge Graph** grow as you continue the conversation.
4. Use **Export Graph JSON** to download the current graph state.

---

## What gets exported

The export is meant to be easy to pipe into other workflows. Expect:
- A concept list (nodes)
- Relationship triples (edges)
- Optional metadata that can support downstream enrichment (e.g., sources, domain, timestamps)

---

## Architecture (high level)

Concept Cartographer makes **two LLM calls per user turn**:
- **Chat call** → generates a helpful response
- **Extraction call** → returns structured JSON (concepts + relations) that updates the graph

The graph is held in app state and re-rendered each turn.

---

## Deployment

A lightweight EC2 + systemd deployment guide is included:

- See `DEPLOYMENT.md` for an end-to-end walkthrough (venv setup, environment file, systemd service, logs).

---

## Contributing

Contributions are welcome, especially improvements that keep the demo:
- **fast**
- **visually clear**
- **cognitively informed**
- **maintainable**

Start with `CONTRIBUTING.md` for guidelines and suggested areas of work.

---

## License

MIT — see `LICENSE`.

---

## Contact

Barbara Hidalgo-Sotelo  
- GitHub: https://github.com/dagny099  
- LinkedIn: https://www.linkedin.com/in/barbara-hidalgo-sotelo/  
