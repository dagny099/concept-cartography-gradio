# Concept Cartographer — Architecture Notes

*For developers extending or maintaining this project.*
*Last updated: v8 (Feb 28 2026)*

---

## 1. Caching Implementation

### How It Works Today (Phase 1: In-Memory)

The app caches responses for **first-turn prompts only** — meaning prompts sent with no prior conversation history. This is the common case for the 4 pre-selected example buttons and for anyone typing a fresh question on load.

#### The Cache Object

```python
_response_cache: dict[str, dict] = {}
```

A plain Python dictionary, keyed by a truncated SHA-256 hash. Lives in process memory. Resets on server restart.

#### Key Generation

```python
def _cache_key(model, system_prompt, user_msg, domain) -> str:
    blob = f"{model}|{system_prompt}|{user_msg}|{domain}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
```

The key is a deterministic function of **every input that affects the output**:

| Input | Why it's in the key |
|---|---|
| `model` | Different models produce different responses |
| `system_prompt` | Contains the domain lens and extraction rules |
| `user_msg` | The actual question |
| `domain` | Redundant with system_prompt, but cheap insurance |

#### Cache Value

Each entry stores three things:

```python
{
    "display":     "Narrative paragraph shown in the chat bubble",
    "extraction":  {
        "concepts":       [...],  # Ontology nodes → consumed by update_graph()
        "relationships":  [...]   # Ontology edges → consumed by update_graph()
    },
    "connections": "Markdown for the 🔗 Latest Connections panel"
}
```

`display` and `connections` serve the UI; `extraction` feeds the graph. Stored separately so each consumer gets exactly what it needs.

#### When the Cache Is Checked

Inside `chat_and_extract()`, the cache is checked **before** any API call:

```python
is_first_turn = len(history) == 0
if is_first_turn:
    key = _cache_key(CHAT_MODEL, system_prompt, message, domain)
    cache_hit = _response_cache.get(key)
    if cache_hit:
        update_graph(cache_hit["extraction"])
        return cache_hit["display"], cache_hit.get("connections", "")
```

Note that `chat_and_extract()` returns a **tuple** `(display_text, connections_md)` — both on cache hit and on a fresh API call.

#### Why First-Turn Only?

With no conversation history, the same inputs always produce a functionally equivalent output (modulo temperature randomness, which is acceptable). Once there's history, the response depends on prior turns — making the cache key space explode and hit rates drop to near zero.

#### Cache Key Stability

The system prompt is now **stable within a domain session** — it no longer includes a `max_concepts` value that varied with graph size. This means cache hits are fully consistent: asking the same first-turn question in the same domain always hits the cache, regardless of how large the graph has grown.

Concept filtering is handled **post-extraction** in `update_graph()` via `GRAPH_FREE_GROWTH_THRESHOLD = 30`, not in the prompt. Above that threshold, only concepts that connect to an already-existing node are admitted to the graph — keeping it coherent rather than accumulating disconnected islands. If an entire extraction is disconnected (topic pivot), all concepts are added anyway to avoid a confusing no-op.

#### What the Cache Does NOT Do

- Does not persist across server restarts
- Does not cache multi-turn responses (by design)
- Does not deduplicate near-identical queries (e.g., "How do NNs learn?" vs "How do neural networks learn?")
- Does not have a TTL or eviction policy (unnecessary at current scale — 5 domains × 4 examples = 20 max entries)

### Roadmap: Future Cache Phases

#### Phase 2: File-Based Persistence

Store `_response_cache` as a JSON file on disk. Load on startup, write after each new entry.

```
cache/
  responses.json    ← serialized _response_cache dict
```

**Key benefit:** You could ship a **pre-warmed cache** in the repo, so the 20 example prompts are instant from first boot — zero API calls for the demo happy path.

**Effort:** ~30 minutes. Add `json.dump` on write, `json.load` on startup, handle file-not-found gracefully.

**Watch out for:** File locking if you ever run multiple workers (not a concern for single-process Gradio).

#### Phase 3: SQLite + TTL

For production with real traffic. Schema:

```sql
CREATE TABLE cache (
    key TEXT PRIMARY KEY,
    display TEXT,
    extraction JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hit_count INTEGER DEFAULT 0
);
```

Adds TTL-based expiry (e.g., 7 days), hit counting for analytics, and survives process restarts without a flat file.

**When this matters:** If you deploy with `share=True` or behind a reverse proxy and want to understand which prompts people actually use.

#### Phase 4: OpenAI Prompt Caching

OpenAI automatically caches long system prompts server-side (50% off cached tokens). Only activates for prompts >1024 tokens. Your current system prompt is ~200 tokens, so this won't help today. But if you add few-shot examples or domain knowledge to the system prompt in the future, it becomes relevant.

---

## 2. Domain & Question Configuration Strategy

### Current State

Domains and their associated data live in **three separate locations** in the codebase:

| Data | Location | What it controls |
|---|---|---|
| Domain names | `gr.Dropdown(choices=[...])` | The UI dropdown options |
| Example prompts | `DOMAIN_EXAMPLES` dict | The 4 suggestion buttons per domain |
| Extraction hints | `domain_hints` dict inside `_build_system_prompt()` | How the LLM prioritizes concept categories |
| Button colors | `DOMAIN_CSS_CLASSES` dict + CSS rules | Visual tinting of example buttons |

Adding a new domain currently means editing all four places — which is error-prone and not contributor-friendly.

### Recommended Approach: Single Config File

Consolidate everything into one `domains.yaml` (or `domains.json`) file:

```yaml
# domains.yaml
domains:
  AI/ML:
    color: "#45B7D1"            # Used for button tint + potentially graph theming
    css_class: "domain-aiml"
    extraction_hint: "algorithms (Method), architectures (Entity), theories (Theory), processes (Process)"
    examples:
      - "How do neural networks learn?"
      - "Explain attention mechanisms in transformers"
      - "What is transfer learning?"
      - "How does reinforcement learning work?"

  Cognitive Science:
    color: "#FF6B6B"
    css_class: "domain-cogsci"
    extraction_hint: "brain regions (Entity), cognitive theories (Theory), mental processes (Process), properties (Property)"
    examples:
      - "How does working memory differ from long-term memory?"
      - "What is the dual-process theory of thinking?"
      - "How do bilinguals switch between languages?"
      - "What role does attention play in perception?"

  # ... etc
```

Then at startup, the app reads this file and derives everything:

```python
import yaml

with open("domains.yaml") as f:
    DOMAIN_CONFIG = yaml.safe_load(f)["domains"]

DOMAIN_NAMES = list(DOMAIN_CONFIG.keys())
DOMAIN_EXAMPLES = {k: v["examples"] for k, v in DOMAIN_CONFIG.items()}
DOMAIN_HINTS = {k: v["extraction_hint"] for k, v in DOMAIN_CONFIG.items()}
DOMAIN_CSS_CLASSES = {k: v["css_class"] for k, v in DOMAIN_CONFIG.items()}
```

**Why YAML over JSON:** YAML allows comments (helpful for explaining why certain prompts were chosen), multiline strings for extraction hints, and is more readable for non-developers who might want to add their own domain.

**Why not a database or admin UI:** Overkill for the current scale. A config file is version-controlled, diffable, and requires zero infrastructure. If you eventually want users to create their own domains at runtime (a la "custom lenses"), that's a different product feature.

### What Makes a Good Example Prompt

Not all questions produce good graphs. From testing, the best example prompts share these qualities:

1. **Ask "how" or "what is the relationship between"** — these naturally elicit multi-concept answers with clear edges
2. **Target a scope of 4-8 concepts** — too narrow ("What is a neuron?") produces a star graph; too broad ("Explain all of biology") overwhelms
3. **Involve processes, not just definitions** — "How does X work?" produces richer edge types than "What is X?"
4. **Connect to other prompts in the same domain** — the 4 examples should produce graphs that overlap when combined, so a user who tries all 4 sees a rich, interconnected map

### Migration Path

1. **v9 (minimal):** Move `DOMAIN_EXAMPLES` and `domain_hints` to a shared `DOMAIN_CONFIG` dict at the top of the file. Single source of truth, no external file yet.
2. **v10 (external config):** Extract to `domains.yaml`. Add `pyyaml` to requirements.
3. **vNext (contributor-friendly):** Add a `CONTRIBUTING.md` section on "Adding a New Domain" that walks through editing the YAML file and testing with `python concept_cartographer.py --domain "Your Domain"`.

---

## 3. Open-Source Assessment

### The Case For

**It's a strong portfolio piece that gains value from being open.** Here's why:

**Differentiated positioning.** Most LLM demo repos are thin wrappers — "chat with PDF," "talk to your database." Concept Cartographer does something structurally different: it makes implicit knowledge structures explicit. The single-call JSON architecture, the post-hoc graph filtering strategy (`GRAPH_FREE_GROWTH_THRESHOLD`), the domain-lens system — these are design decisions worth studying, not just code worth running. Open-sourcing lets people see the thinking, which is the real showcase for someone positioning as an AI solutions consultant.

**Community as signal.** Stars, forks, and issues on a public repo are social proof that carries weight in hiring conversations and consulting pitches. "I built a tool that 200 people starred" is more compelling than "I built a tool."

**The contribution surface is inviting.** The domain config system (once externalized to YAML) is an obvious, low-barrier contribution path. Someone who knows nothing about Gradio or LLMs can still add a "Legal" or "Environmental Science" domain with good example prompts. That's rare in AI projects and could drive organic engagement.

**Educational value.** The codebase demonstrates several patterns that are hard to find clean examples of: structured JSON output from LLMs, Gradio state management (the immutable-list pattern), adaptive prompt engineering, single-call architecture vs. multi-call pipelines. This is exactly the kind of thing people search GitHub for.

### The Case Against (and mitigations)

**API key exposure risk.** The app requires an OpenAI API key. Someone cloning the repo and deploying carelessly could expose their key.
→ *Mitigation:* `.env.example` file, `.gitignore` for `.env`, and a warning in the README. Standard practice.

**Demo abuse.** If you deploy a public instance with your key, the 3-turn limit protects you. But someone could fork, remove the limit, and run up costs on their own key — which is their problem, not yours.

**Competitive concern.** Could someone take this and commercialize it?
→ *Mitigation:* Choose a license that matches your intent. **MIT** if you want maximum adoption and don't care about commercial use. **AGPL-3.0** if you want to ensure modifications to deployed versions stay open. **Apache 2.0** as a middle ground (permissive but with patent protection). For a portfolio/consulting context, MIT is probably right — you want people to use it and associate it with your name.

**Maintenance burden.** Open-source repos that go quiet look worse than no repo at all.
→ *Mitigation:* Set expectations in the README: "This is a portfolio project and learning tool. Issues and PRs welcome but response times may vary." Honest framing prevents disappointment.

### Recommended Launch Checklist

- [ ] **Choose a license** (recommend MIT for your use case)
- [ ] **Write a README** that leads with a screenshot/GIF of the app in action — visual-first, code-second
- [ ] **Tag a release** (v1.0.0) so people can pin to a stable version
- [ ] **Add the 30-second demo** — this is the single highest-ROI asset for a repo like this. Tools like [Gifski](https://gif.ski/) or screen recording → GIF conversion make this easy

## Few Other Customization Ideas I like:

### Add Domain Presets:
```python
presets = {
    "Science": "Focus on theories, methods, and phenomena",
    "Business": "Focus on processes, strategies, and metrics",
    "Tech": "Focus on systems, architectures, and patterns"
}
```

### Add Graph Statistics:
```python
stats = f"""
**Graph Stats:**
- Concepts: {len(concept_graph.nodes)}
- Relationships: {len(concept_graph.edges)}
- Clusters: {nx.number_connected_components(concept_graph.to_undirected())}
"""
```

### Different Layouts:
```python
# Replace spring_layout with:
pos = nx.kamada_kawai_layout(concept_graph)  # Better for hierarchical
# or
pos = nx.circular_layout(concept_graph)  # Simpler, faster
```
