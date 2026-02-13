# 🗺️ Concept Cartographer

**Interactive Knowledge Mapping powered by LLMs**

An intelligent chat interface that extracts concepts and relationships from conversations and visualizes them as dynamic knowledge graphs in real-time.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-6.0+-orange.svg)

---

## 🎯 Overview

Concept Cartographer bridges natural language conversation with formal knowledge representation. As you discuss topics with the AI, it automatically:

1. **Extracts key concepts** from the conversation
2. **Identifies relationships** between concepts  
3. **Builds a knowledge graph** that grows with each exchange
4. **Visualizes the ontology** in real-time

Built by [Dagny Barbierski](https://barbhs.com) | Cognitive Science + AI

### Why This Exists

Most LLM chat interfaces are ephemeral—knowledge disappears after the conversation ends. Concept Cartographer makes implicit knowledge structures explicit and exportable, turning conversations into reusable ontologies.

**Use cases:**
- Learning new topics and seeing conceptual connections
- Brainstorming and idea mapping
- Research literature synthesis
- Interview/presentation prep
- Building domain-specific knowledge bases

---

## ✨ Features

- **🤖 Dual LLM Workflow**: Separate calls for chat responses and concept extraction
- **🎨 Real-time Visualization**: NetworkX + Matplotlib graph rendering
- **🏷️ Domain-Aware Extraction**: Optimized prompts for AI/ML, Cognitive Science, Healthcare, Business, and General domains
- **💾 Stateful Architecture**: Graph persists and grows across conversation turns
- **📊 Structured Outputs**: JSON-mode extraction for reliable parsing
- **💰 Cost-Optimized**: Uses GPT-4o-mini for 94% cost reduction vs GPT-4
- **📤 Export Capability**: Download your knowledge graph as JSON
- **🎯 Guided UX**: Example prompts and domain selection reduce "blank canvas" paralysis

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/concept-cartographer.git
cd concept-cartographer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running Locally

```bash
python concept_cartographer.py
```

The app will launch at `http://localhost:7860` with a public Gradio share URL.

---

## 📖 Usage

### Basic Workflow

1. **Select a domain** from the dropdown (AI/ML, Cognitive Science, etc.)
2. **Ask a question** or click an example prompt
3. **Watch the graph** update in real-time as concepts are extracted
4. **Continue the conversation** to expand your knowledge map
5. **Export** your ontology as JSON when done

### Example Session

```
Domain: AI/ML
Question: "How do neural networks learn?"

→ Graph shows: Neural Networks, Backpropagation, Gradient Descent, 
   Loss Function, Weights, Training Data

Question: "How does this relate to deep learning?"

→ Graph expands with: Deep Learning, Hidden Layers, Feature Learning,
   plus relationships linking to existing concepts
```

### Tips for Best Results

- **Start broad, then narrow**: Begin with overview questions, drill down into specifics
- **Use the domain selector**: Improves extraction quality for specialized topics
- **Clear the graph** when switching to unrelated topics
- **Export frequently**: Save your knowledge graphs for later reference

---

## 🏗️ Architecture

### Technology Stack

- **Frontend**: Gradio 6.0+ (Blocks API for custom layouts)
- **LLM**: OpenAI GPT-4o-mini (cost-optimized)
- **Graph**: NetworkX (data structure) + Matplotlib (visualization)
- **State Management**: Global Python object (ephemeral per session)

### Data Flow

```
User Input
    ↓
Chat LLM Call (GPT-4o-mini)
    ↓
Assistant Response
    ↓
Concept Extraction LLM Call (GPT-4o-mini + JSON mode)
    ↓
Update NetworkX Graph
    ↓
Render Matplotlib Visualization
    ↓
Display to User
```

### Key Design Decisions

**Why two LLM calls?**
- Separates user-facing chat quality from extraction quality
- Allows different prompts/temperatures for each task
- Enables hybrid model strategies (e.g., GPT-4 for chat, mini for extraction)

**Why structured JSON outputs?**
- Guarantees parseable responses (no regex hacks)
- Enables programmatic use of extracted data
- Foundation for building real tools vs chatbots

**Why global state?**
- Simple for POC/demo purposes
- Keeps graph persistent within session
- Easy to upgrade to database for production

---

## 💰 Cost Analysis

Using GPT-4o-mini for both chat and extraction:

| Usage | Cost |
|-------|------|
| Per message | ~$0.004 |
| 100 messages | ~$0.40 |
| 1000 messages | ~$4.00 |

**Comparison**: Using GPT-4 for both would cost **$63** for 1000 messages (94% more expensive).

---

## 🔧 Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_key_here
```

### Customization Options

**Change LLM models:**
```python
# In chat_and_extract()
model="gpt-4o"  # Use more powerful model for chat

# In extract_concepts_and_relationships()
model="gpt-4o-mini"  # Keep mini for extraction
```

**Adjust graph layout:**
```python
# In render_graph()
pos = nx.spring_layout(concept_graph, k=2, iterations=50)
# Try: nx.kamada_kawai_layout() for hierarchical
#      nx.circular_layout() for simpler
```

**Add custom domains:**
```python
domain_hints = {
    "AI/ML": "Prioritize: algorithms, architectures...",
    "Your Domain": "Prioritize: your custom guidance...",
}
```

---

## 📦 Deployment

### Local Development
```bash
python concept_cartographer.py
```

### Production (EC2/Cloud)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on:
- Systemd service setup
- Nginx reverse proxy configuration
- SSL with Let's Encrypt
- Custom domain setup

**Quick deploy:**
```bash
# As systemd service
sudo systemctl start concept-cartographer

# View logs
sudo journalctl -u concept-cartographer -f
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add authentication/user sessions
- [ ] Persist graphs to database (PostgreSQL + pgvector)
- [ ] Interactive graph exploration (zoom, click nodes for details)
- [ ] Support for additional LLM providers (Anthropic, local models)
- [ ] Export to other formats (GraphML, Cypher, RDF)
- [ ] Graph diff/merge capabilities
- [ ] Collaborative multi-user graphs

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black concept_cartographer.py

# Type checking
mypy concept_cartographer.py
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI** for GPT models and structured output capabilities
- **Gradio** team for the excellent UI framework
- **NetworkX** developers for robust graph algorithms
- Inspired by semantic network research in cognitive science

---

## 📧 Contact

**Dagny Barbierski**
- Website: [barbhs.com](https://barbhs.com)
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🔗 Related Projects

- [ConvoScope](https://github.com/yourusername/convoscope) - Multi-LLM conversation management
- [ChronoScope](https://github.com/yourusername/chronoscope) - Timeline visualization from documents

---

## 📊 Project Stats

- **Lines of Code**: ~350
- **Development Time**: ~2 hours (including optimization)
- **Dependencies**: 5 core packages
- **Cost per 1K messages**: ~$4 (using GPT-4o-mini)

---

**Built with ❤️ by a cognitive scientist who builds things people actually use**
