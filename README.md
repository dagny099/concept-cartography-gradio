# 🗺️ Concept Cartographer

**Interactive knowledge mapping — ask questions and watch concepts emerge as a graph.**

Concept Cartographer is a Gradio app that turns a chat into a growing *concept map*: it extracts key concepts and relationships from your conversation and renders them as a knowledge graph you can export.

- **Repo:** https://github.com/dagny099/concept-cartography-gradio  
- **Live demo:** https://concept-cartographer.com/
- **Built by:** Barbara Hidalgo-Sotelo (Cognitive Science + AI)  
- **LinkedIn:** https://www.linkedin.com/in/barbara-hidalgo-sotelo/

---

## What it does

As you chat, the app:

1. **Responds conversationally** to your question
2. **Extracts structured concepts + relationships** from the conversation (single LLM call — fast and cheap)
3. **Updates a persistent graph** across turns
4. **Visualizes the graph** in real time with a color-coded legend
5. Shows **key connections** in plain language below the conversation
6. Lets you **export the graph as JSON or PNG** for downstream use (Neo4j, notes, outlines, presentations, etc.)

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

The app starts at `http://localhost:7870`. If Gradio sharing is enabled, you’ll also see a public `gradio.live` link printed to the console.

---

## Versions

This project is currently developed and tested on **Gradio 6.5.1** (pinned in `requirements.txt`).

If you run into UI differences across Gradio versions, pin explicitly:
```bash
pip install "gradio==6.5.1"
```

---

## How to use

1. Pick a **Domain** (e.g., AI/ML or Cognitive Science).
   Domain selection nudges extraction toward domain-relevant concepts and relation types.
2. Ask a question (or click a starter prompt).
3. Watch the **Knowledge Graph** grow as you continue the conversation.
4. Check the **🔗 Latest Connections** panel below the chat for a plain-language summary of what was just extracted.
5. Use **Export Graph JSON** or **Export Graph PNG** to download the current graph state.

---

## What gets exported

The export is meant to be easy to pipe into other workflows. Expect:
- A concept list (nodes)
- Relationship triples (edges)
- Optional metadata that can support downstream enrichment (e.g., sources, domain, timestamps)

---

## Architecture (high level)

Concept Cartographer makes a **single LLM call per user turn**, returning a structured JSON response that contains both a conversational narrative and the extracted ontology (concepts + relationships) simultaneously. This is ~50% faster and cheaper than a two-call approach.

The response shape:
```json
{
  "narrative":      "A 2-4 sentence explanation woven from the extracted concepts",
  "concepts":       [{"name": "...", "category": "Entity|Process|Theory|Method|Property"}],
  "relationships":  [{"from": "...", "to": "...", "type": "causes|requires|enables|..."}]
}
```

The graph is held in app state and re-rendered each turn. New concepts are filtered before being added: once the graph exceeds 30 nodes, only concepts that connect to an existing node are admitted — keeping the graph coherent rather than creating disconnected islands.

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
