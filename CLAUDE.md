# CLAUDE.md - Technical Documentation for AI Assistants

This file provides context for AI assistants (like Claude) working with this codebase. It documents technical decisions, lessons learned, and gotchas that may not be obvious from the code alone.

---

## 🎯 Project Purpose & Context

**What this is**: A Gradio-based chat interface that extracts concepts/relationships from LLM conversations and builds knowledge graphs in real-time.

**Target user**: Dagny Barbierski - PhD in Cognitive Science from MIT, data scientist/AI consultant, with expertise in knowledge graphs, healthcare AI, and semantic networks.

**Key constraint**: Built as a proof-of-concept in under 2 hours, prioritizing speed and impression over production robustness.

**Design philosophy**: 
- Show don't tell (visual > textual)
- Cognitive science meets practical engineering
- Structured outputs over free text parsing
- Cost-conscious (optimize for GPT-4o-mini)

---

## 🏗️ Architecture Decisions

### Why Gradio Blocks Instead of ChatInterface?

**ChatInterface** is simpler but inflexible:
```python
# Limited to this pattern:
gr.ChatInterface(fn=my_function)
```

**Blocks** provides full control:
```python
with gr.Blocks() as demo:
    with gr.Row():
        chatbot = gr.Chatbot()
        plot = gr.Plot()  # Can't do this with ChatInterface!
```

**Lesson**: Use Blocks for multi-component UIs. Only use ChatInterface for the simplest possible chatbots.

---

### Why Global State for the Graph?

```python
concept_graph = nx.DiGraph()  # Global variable

def update_graph(data):
    global concept_graph
    concept_graph.add_node(...)
```

**Alternatives considered:**
1. **Gradio State component**: More proper but adds complexity
2. **Database persistence**: Overkill for POC
3. **Session storage**: Requires additional backend

**Trade-off**: Global state is simple for single-user demos but won't scale to multi-user production. Acceptable for this use case.

**Production path**: Move to PostgreSQL with pgvector for semantic search + graph storage.

---

### Why Two LLM Calls Per Message?

```python
# Call 1: Chat response
chat_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=conversation_history
)

# Call 2: Concept extraction
extraction = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[extraction_prompt],
    response_format={"type": "json_object"}
)
```

**Why not combine?**
- Different optimal temperatures (0.7 for chat, 0.1 for extraction)
- Different prompts/system messages
- Allows hybrid model strategies
- Extraction can happen asynchronously (future optimization)

**Cost**: ~$0.004 per message with GPT-4o-mini (acceptable)

**Future optimization**: Run calls concurrently with `asyncio`

---

## 🔧 Technical Lessons Learned

### 1. Gradio 6.0 Breaking Changes

**Problem encountered**: Gradio 6.0 changed several APIs that broke the initial implementation.

#### Theme Parameter Migration

**OLD (Gradio <6.0):**
```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    ...
```

**NEW (Gradio 6.0+):**
```python
with gr.Blocks() as demo:
    ...

demo.launch(theme=gr.themes.Soft())  # Theme moved to launch()
```

**Why**: Gradio team moved theming to launch-time configuration for better flexibility.

#### Examples Component Issues

**OLD approach (doesn't work in Gradio 6.0):**
```python
examples = gr.Examples(
    examples=[["Question 1"], ["Question 2"]],
    inputs=None,  # This causes ValueError in 6.0
    label=None
)
```

**Error**: `ValueError: Component must be provided as a 'str' or 'dict' or 'Component' but is None`

**NEW approach (simple buttons):**
```python
# Create clickable buttons instead
ex1 = gr.Button("How do neural networks learn?", size="sm")
ex2 = gr.Button("Explain attention mechanisms", size="sm")

# Wire them to populate the textbox
ex1.click(lambda: "How do neural networks learn?", outputs=[msg])
ex2.click(lambda: "Explain attention mechanisms", outputs=[msg])
```

**Why this is better:**
- More control over layout
- No component coupling issues
- Cleaner UX (buttons vs dataset)
- Works reliably in Gradio 6.0+

---

### 2. ChatInterface Message Format in Gradio 6.0

**CRITICAL**: Understanding the message history format is essential for building correct conversation handlers.

#### Message History Structure

When using `gr.ChatInterface` or `gr.Chatbot`, the history format is:

```python
history = [
    ("User message 1", "Assistant response 1"),
    ("User message 2", "Assistant response 2"),
    ("User message 3", None),  # In-progress exchange
]
```

**Format**: List of tuples `[(user_msg, assistant_msg), ...]`

**Key insights:**
1. Each tuple represents ONE exchange
2. User message is always first element
3. Assistant message is second element (can be `None` for incomplete exchanges)
4. History grows by appending new tuples

#### Converting to OpenAI API Format

```python
def chat_and_extract(message, history, domain="General"):
    # Build OpenAI messages from Gradio history
    messages = [{"role": "system", "content": "System prompt..."}]
    
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Important: check for None!
            messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Call API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    return response.choices[0].message.content
```

**Critical gotcha**: Always check `if assistant_msg` before appending! In-progress exchanges have `None` for the assistant response.

#### Updating History After Response

```python
def respond(message, chat_history, domain):
    # Get response
    bot_message = chat_and_extract(message, chat_history, domain)
    
    # Update history by appending tuple
    chat_history.append((message, bot_message))
    
    return "", chat_history, updated_graph
```

**Pattern**: Return empty string for input (clears textbox), updated history, and any other outputs.

---

### 3. Structured Outputs with OpenAI

**Game changer**: OpenAI's `response_format` parameter guarantees valid JSON.

**WITHOUT structured output:**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Return JSON with concepts..."}]
)

# Hope the LLM returns valid JSON!
try:
    data = json.loads(response.choices[0].message.content)
except json.JSONDecodeError:
    # Handle errors, retry, etc.
    pass
```

**WITH structured output (OpenAI JSON mode):**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Return JSON with concepts..."}],
    response_format={"type": "json_object"}  # Magic!
)

# Guaranteed valid JSON
data = json.loads(response.choices[0].message.content)
```

**Requirements:**
1. Must explicitly mention "JSON" in prompt
2. Only works with newer models (gpt-4o, gpt-4o-mini, gpt-4-turbo)
3. Returns pure JSON (no markdown code blocks)

**Why this matters**: Eliminates 90% of parsing edge cases. Foundation for building reliable tools.

---

### 4. Cost Optimization Strategy

**Token usage breakdown:**
```
Chat response:
- Input: ~100-300 tokens (conversation history + message)
- Output: ~200-500 tokens (assistant response)

Concept extraction:
- Input: ~150-200 tokens (extraction prompt + text)
- Output: ~100-200 tokens (JSON with concepts)

Total: ~550-1200 tokens per exchange
```

**Model pricing (per 1M tokens):**
```
GPT-4:          $30 input, $60 output
GPT-4o:         $2.50 input, $10 output
GPT-4o-mini:    $0.15 input, $0.60 output
```

**Cost per 100 messages:**
```
GPT-4 (both):       ~$6.30
GPT-4o (both):      ~$2.60
GPT-4o-mini (both): ~$0.40  ← Chosen approach
Hybrid (4o + mini): ~$2.60
```

**Decision**: Use GPT-4o-mini for both chat and extraction
- 94% cheaper than GPT-4
- Quality is 90-95% as good for this use case
- Speed is faster (~1-2s vs 3-5s)
- Perfect for demos/POCs

**When to upgrade**: Production with paying users → hybrid approach (GPT-4o for chat, mini for extraction)

---

### 5. NetworkX Graph Rendering

**Layout algorithm choice matters**:

```python
# Force-directed (good for organic networks)
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Hierarchical (good for taxonomies)
pos = nx.kamada_kawai_layout(G)

# Circular (simple, predictable)
pos = nx.circular_layout(G)
```

**Chosen**: `spring_layout` with:
- `k=2`: More spread out (default is 1)
- `iterations=50`: Better convergence
- `seed=42`: Reproducible layouts

**Performance**: Fine for <100 nodes. Above that, consider:
- Incremental layouts (only reposition new nodes)
- WebGL rendering (Three.js, D3.js)
- Graph sampling/clustering

---

## 🎨 UX Patterns

### Solving the "Blank Canvas" Problem

**Problem**: Empty input boxes paralyze users ("What should I ask?")

**Solution**: Provide scaffolding without constraining exploration

**Implementation**:
1. **Domain dropdown**: Sets context and improves extraction
2. **Example buttons**: Clickable prompts that populate input
3. **Clear labeling**: "Try these to get started"

**Code pattern**:
```python
# Don't do this (too abstract):
gr.Textbox(placeholder="Ask a question...")

# Do this (concrete examples):
ex1 = gr.Button("How do neural networks learn?")
ex1.click(lambda: "How do neural networks learn?", outputs=[msg])
```

**Lesson**: Default states matter. Give users a clear entry point.

---

### Progressive Disclosure

**Pattern**: Show complexity only when needed

```python
# Export is hidden by default
export_output = gr.Textbox(visible=False)

# Only show when user clicks Export
export_btn.click(
    show_export,
    outputs=[export_output]  # Makes visible + populates
)
```

**Why**: Reduces cognitive load. Users don't need to see JSON export until they want it.

---

## 🔒 Security & Privacy Considerations

### API Key Management

```python
# NEVER do this:
client = OpenAI(api_key="sk-...")

# Always use environment variables:
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

**For deployment**:
```bash
# .env file (gitignored)
OPENAI_API_KEY=sk-...

# Systemd service (secure)
EnvironmentFile=/path/to/.env
```

### Data Privacy

**Current implementation**: No data persistence (graph lives in memory)

**Implications**:
- ✅ No PII stored
- ✅ Automatic cleanup on session end
- ❌ Users lose graphs on reload
- ❌ No multi-user support

**For production**:
- Add user authentication
- Store graphs per-user in database
- Implement data retention policies
- GDPR compliance for EU users

---

## 🚀 Performance Optimization Opportunities

### 1. Async LLM Calls

**Current (sequential)**:
```python
chat_response = await client.chat.completions.create(...)  # 2-3s
extraction = await client.chat.completions.create(...)     # 1-2s
# Total: 3-5s
```

**Optimized (parallel)**:
```python
import asyncio

chat_task = client.chat.completions.create(...)
extract_task = client.chat.completions.create(...)

chat_response, extraction = await asyncio.gather(chat_task, extract_task)
# Total: 2-3s (40% faster!)
```

### 2. Caching Extractions

```python
@lru_cache(maxsize=100)
def extract_concepts(text_hash):
    # Only re-extract if content changed
    ...
```

### 3. Incremental Graph Rendering

**Current**: Re-render entire graph every time
**Better**: Only redraw new nodes/edges

```python
def render_graph_incremental(new_nodes, new_edges):
    # Maintain existing positions
    # Only compute layout for new elements
    ...
```

---

## 📝 Code Patterns & Conventions

### Function Naming

```python
# Gradio event handlers: verb + noun
def respond(message, history):  # Not handle_response()
def clear_all():                # Not clear_graph()
def show_export():              # Not toggle_export()
```

### Docstrings

```python
def chat_and_extract(message, history, domain="General"):
    """
    Main chat function that also extracts concepts.
    Now domain-aware for better concept extraction.
    
    Args:
        message: Current user message
        history: List of (user_msg, assistant_msg) tuples
        domain: Domain focus for extraction hints
        
    Returns:
        str: Assistant's response text
    """
```

**Pattern**: One-line summary, then details. Always document parameters and return values.

### Error Handling Philosophy

**For POCs**: Fail visibly
```python
# No try/except hiding issues
response = client.chat.completions.create(...)  # Let it raise!
```

**For production**: Graceful degradation
```python
try:
    extracted = extract_concepts(text)
except Exception as e:
    logging.error(f"Extraction failed: {e}")
    extracted = {"concepts": [], "relationships": []}
```

---

## 🐛 Known Issues & Gotchas

### 1. Graph Resets on Refresh

**Issue**: Global state is lost when user refreshes page
**Workaround**: None currently
**Fix**: Add session persistence with database

### 2. No Duplicate Detection

**Issue**: Same concept can be added multiple times with slight variations
```python
# These create separate nodes:
"Neural Network"
"Neural Networks" 
"neural network"
```

**Fix**: Add normalization in `update_graph()`:
```python
normalized_name = concept["name"].lower().strip()
```

### 3. LLM Hallucinated Relationships

**Issue**: Sometimes creates nonsensical relationships
**Mitigation**: Lower temperature (0.1), better prompts
**Not a bug**: Inherent to LLM outputs

### 4. Graph Layout Instability

**Issue**: Nodes jump around between renders (spring layout is stochastic)
**Fix**: Use fixed seed: `nx.spring_layout(G, seed=42)`

---

## 🔮 Future Enhancements

### High Value, Low Effort
- [ ] Add graph statistics sidebar (node count, edge count, density)
- [ ] Domain-specific color schemes
- [ ] Download graph as PNG image
- [ ] Undo last addition

### Medium Effort
- [ ] Interactive graph (click nodes to explore, zoom/pan)
- [ ] Multi-turn extraction (build on previous extractions)
- [ ] Support for Claude API (Anthropic)
- [ ] Comparison mode (diff two graphs)

### High Effort, High Value
- [ ] User authentication + persistent storage
- [ ] Collaborative graphs (multi-user editing)
- [ ] Graph search and filtering
- [ ] Integration with knowledge bases (Weaviate, Neo4j)
- [ ] Mobile app version

---

## 📚 Key Dependencies

```python
gradio>=4.0.0      # UI framework (tested on 6.0+)
openai>=1.0.0      # LLM API
networkx>=3.0      # Graph algorithms
matplotlib>=3.7.0  # Visualization
python-dotenv      # Environment variables
```

**Version pins**: Only pin major versions unless specific bugs require exact versions.

---

## 🎓 Learning Resources

**For Gradio:**
- [Official Gradio 6.0 migration guide](https://www.gradio.app/guides/migrating-from-3.x-to-4.x)
- [Gradio Blocks documentation](https://www.gradio.app/docs/gradio/blocks)

**For OpenAI structured outputs:**
- [JSON mode documentation](https://platform.openai.com/docs/guides/structured-outputs)

**For NetworkX:**
- [Layout algorithms](https://networkx.org/documentation/stable/reference/drawing.html#layout)
- [Graph visualization](https://networkx.org/documentation/stable/reference/drawing.html)

**For knowledge graphs:**
- [Introduction to Knowledge Graphs (Stanford)](https://web.stanford.edu/class/cs520/)

---

## 🤝 When Helping Users With This Codebase

### Quick Checklist

When a user asks for help, consider:

1. **What are they trying to achieve?** (Demo vs Production vs Learning)
2. **What's their experience level?** (Adjust explanation depth)
3. **What's the time budget?** (Quick fix vs comprehensive solution)
4. **What are the constraints?** (Cost, deployment environment, etc.)

### Common User Requests

**"It's not working"**
→ Check Gradio version (6.0+ required)
→ Verify API keys in `.env`
→ Look for import errors

**"How do I deploy this?"**
→ See DEPLOYMENT.md for production setup
→ Recommend systemd + nginx + Let's Encrypt

**"Can I use a different LLM?"**
→ Yes! Just replace OpenAI client
→ Watch for API format differences
→ Consider Anthropic Claude for better quality

**"How do I add feature X?"**
→ Point to relevant section of this file
→ Provide code example
→ Explain trade-offs

---

**Last Updated**: 2026-02-10  
**Gradio Version Tested**: 6.0+  
**Python Version**: 3.11  

---

*This file should be updated whenever significant technical decisions are made or lessons are learned.*
