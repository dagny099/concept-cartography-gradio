import gradio as gr
import os
import json
import base64
import hashlib
import tempfile
from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global graph to maintain state
concept_graph = nx.DiGraph()

# Track token usage
total_tokens_used = 0
total_api_calls = 0

# Turn limiting for LinkedIn demo
current_turn_count = 0
MAX_TURNS = 3
LINKEDIN_POST_URL = "https://www.linkedin.com/in/barbara-hidalgo-sotelo/"

# UI assets
PENDING_ASSISTANT_TEXT = "..."

# Domain-specific example prompts — curated for graph variety and "aha" moments
DOMAIN_EXAMPLES: dict[str, list[str]] = {
    "AI/ML": [
        "How do neural networks learn?",
        "Explain attention mechanisms in transformers",
        "What is transfer learning?",
        "How does reinforcement learning work?",
    ],
    "Cognitive Science": [
        "How does working memory differ from long-term memory?",
        "What is the dual-process theory of thinking?",
        "How do bilinguals switch between languages?",
        "What role does attention play in perception?",
    ],
    "Healthcare": [
        "How does the immune system fight infection?",
        "What is the difference between a virus and a bacterium?",
        "How do clinical trials work?",
        "What is precision medicine?",
    ],
    "Business": [
        "What is a platform business model?",
        "How do network effects create moats?",
        "What makes OKRs different from KPIs?",
        "How does design thinking apply to strategy?",
    ],
    "General": [
        "How does the greenhouse effect work?",
        "What is systems thinking?",
        "How do supply chains create resilience?",
        "What is the relationship between language and thought?",
    ],
}

# ── Phase 1 Response Cache ──────────────────────────────────────────
# In-memory dict: hash(model + system_prompt + user_message + domain) → {chat, extraction}
# Survives across turns but resets on server restart. Zero dependencies.
# For the 4 pre-selected example prompts this means instant responses on repeat use.
_response_cache: dict[str, dict] = {}

def _cache_key(model: str, system: str, user_msg: str, domain: str) -> str:
    """Deterministic cache key from the inputs that affect the response."""
    blob = f"{model}|{system}|{user_msg}|{domain}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

def build_favicon_head() -> str:
    """
    Deployment-proof favicon:
    - Reads a local icon from ./assets/
    - Embeds it as a base64 data URI in the <head>, so it works on Gradio share links,
      reverse proxies, and most static hosting setups without extra routing.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "assets", "favicon.png"),
        os.path.join(base_dir, "assets", "favicon.ico"),
        os.path.join(base_dir, "assets", "icon.png"),
        os.path.join(base_dir, "assets", "icon.ico"),
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    raw = f.read()
                b64 = base64.b64encode(raw).decode("ascii")
                mime = "image/png" if path.lower().endswith(".png") else "image/x-icon"
                return f'<link rel="icon" type="{mime}" href="data:{mime};base64,{b64}">'
        except Exception:
            # If anything goes wrong, silently fall back to no favicon rather than crashing the app
            pass
    return ""

FAVICON_HEAD = build_favicon_head()


def adaptive_concept_cap(existing_node_count: int) -> int:
    """Adaptive cap to keep the graph readable as it grows."""
    if existing_node_count < 25:
        return 12
    if existing_node_count < 60:
        return 8
    return 6


def _build_system_prompt(domain: str, max_concepts: int) -> str:
    """Single system prompt: one call returns narrative + ontology."""
    domain_hints = {
        "AI/ML": "algorithms (Method), architectures (Entity), theories (Theory), processes (Process)",
        "Cognitive Science": "brain regions (Entity), cognitive theories (Theory), mental processes (Process), properties (Property)",
        "Healthcare": "conditions (Entity), treatments (Method), symptoms (Property), processes (Process)",
        "Business": "processes (Process), metrics (Property), strategies (Method), stakeholders (Entity)",
        "General": "diverse types: entities, processes, theories, methods, properties",
    }
    hint = domain_hints.get(domain, domain_hints["General"])

    return f"""You are a knowledge-graph assistant using a {domain} lens. For every user message, return ONLY valid JSON:

{{
  "narrative": "A 2-4 sentence natural-language explanation that answers the question while weaving in the key concepts and how they connect. Write as if explaining to a smart colleague — not a list, but a flowing explanation that makes the graph intuitive.",
  "concepts": [
    {{"name": "ShortName", "category": "Entity|Process|Theory|Method|Property"}}
  ],
  "relationships": [
    {{"from": "ConceptA", "to": "ConceptB", "type": "causes|requires|part_of|type_of|enables"}}
  ]
}}

Rules:
- narrative: answer the question conversationally, naturally referencing the concepts by name. 2-4 sentences.
- concepts: up to {max_concepts}. Prioritize: {hint}. Names 1-3 words.
- relationships: only between listed concepts. Use specific types.
- Return NOTHING outside the JSON object."""


def format_chat_display(parsed: dict) -> str:
    """
    Build conversation-window markdown from the structured response.
    Shows the narrative, then a compact concept inventory.
    """
    narrative = parsed.get("narrative", "")
    concepts = parsed.get("concepts", [])

    parts = [narrative]

    if concepts:
        # Compact inline list: Concept (Category), Concept (Category), ...
        tags = [f"**{c['name']}** ({c.get('category', '?')})" for c in concepts]
        parts.append(f"\n📍 {' · '.join(tags)}")

    return "\n".join(parts)


def chat_and_extract(message, history, domain="General"):
    """
    Single-call architecture: one API request returns both a conversational summary
    and structured ontology data. ~50% faster and cheaper than the old two-call approach.
    """
    global total_tokens_used, total_api_calls

    CHAT_MODEL = "gpt-4o-mini"
    existing_node_count = len(concept_graph.nodes)
    max_concepts = adaptive_concept_cap(existing_node_count)
    system_prompt = _build_system_prompt(domain, max_concepts)

    # ── Cache check (first-turn only: no history means deterministic output) ──
    is_first_turn = len(history) == 0
    if is_first_turn:
        key = _cache_key(CHAT_MODEL, system_prompt, message, domain)
        cache_hit = _response_cache.get(key)
        if cache_hit:
            update_graph(cache_hit["extraction"])
            return cache_hit["display"]

    # ── Build messages from history ──
    messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            role = msg["role"]
            # For user messages, send the raw text (not the decorated display text)
            content = msg.get("raw", msg["content"]) if role == "user" else msg["content"]
            # For assistant messages, send only the summary back as context
            # (the full display markdown would confuse the model)
            if role == "assistant":
                content = msg.get("raw_summary", content)
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    # ── Single structured API call ──
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        total_tokens_used += response.usage.total_tokens
        total_api_calls += 1

        # Guard against truncation
        if response.choices[0].finish_reason != "stop":
            print(f"⚠️ Truncated (finish_reason={response.choices[0].finish_reason})")
            return "Sorry, the response was cut short. Try a more specific question."

        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️ API/parse error: {e}")
        return "Something went wrong — please try again."

    # ── Validate & clean ──
    parsed.setdefault("narrative", "")
    parsed.setdefault("concepts", [])
    parsed.setdefault("relationships", [])

    for c in parsed["concepts"]:
        c["name"] = c.get("name", "").replace('"', '').replace("'", "").strip()
        c.setdefault("category", "Entity")

    # ── Update graph ──
    extraction = {"concepts": parsed["concepts"], "relationships": parsed["relationships"]}
    update_graph(extraction)

    # ── Build display text ──
    display_text = format_chat_display(parsed)

    # ── Cache (first-turn only) ──
    if is_first_turn:
        _response_cache[key] = {"display": display_text, "extraction": extraction}

    return display_text

def update_graph(extracted_data):
    """
    Add new concepts and relationships to the knowledge graph.
    This maintains state across multiple conversation turns.
    """
    global concept_graph
    
    # Add concepts as nodes with their categories
    for concept in extracted_data.get("concepts", []):
        concept_graph.add_node(
            concept["name"],
            category=concept.get("category", "Unknown")
        )
    
    # Add relationships as edges
    for rel in extracted_data.get("relationships", []):
        if rel["from"] in concept_graph.nodes and rel["to"] in concept_graph.nodes:
            concept_graph.add_edge(
                rel["from"],
                rel["to"],
                relationship=rel["type"]
            )

def render_graph():
    """
    Render the current state of the knowledge graph.
    Uses matplotlib + networkx for quick visualization.
    """
    if len(concept_graph.nodes) == 0:
        # Return empty figure if no concepts yet
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Start chatting to build your concept map!', 
                ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
        return fig
    
    # Create figure
    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Use spring layout for nice node positioning
    pos = nx.spring_layout(concept_graph, k=2, iterations=50, seed=42)
    
    # Color nodes by category
    category_colors = {
        "Theory": "#FF6B6B",
        "Method": "#4ECDC4",
        "Entity": "#45B7D1",
        "Property": "#FFA07A",
        "Process": "#98D8C8",
        "Domain": "#F7DC6F",
        "Unknown": "#95A5A6"
    }
    
    node_colors = [
        category_colors.get(concept_graph.nodes[node].get("category", "Unknown"), "#95A5A6")
        for node in concept_graph.nodes
    ]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        concept_graph, pos, 
        node_color=node_colors,
        node_size=2000,
        alpha=0.9,
        ax=ax
    )
    
    # Draw edges with arrows
    nx.draw_networkx_edges(
        concept_graph, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        width=2,
        alpha=0.6,
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        concept_graph, pos,
        font_size=9,
        font_weight='bold',
        ax=ax
    )
    
    # Draw edge labels (relationship types)
    edge_labels = nx.get_edge_attributes(concept_graph, 'relationship')
    nx.draw_networkx_edge_labels(
        concept_graph, pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color='darkblue',
        ax=ax
    )
    
    ax.set_title("Concept Map", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=color, markersize=10, label=category)
        for category, color in category_colors.items()
        if category != "Unknown"
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    return fig

def clear_graph():
    """Reset the knowledge graph."""
    global concept_graph
    concept_graph = nx.DiGraph()
    return render_graph()

def export_graph():
    """Export the current graph as JSON."""
    data = {
        "nodes": [
            {
                "name": node,
                "category": concept_graph.nodes[node].get("category", "Unknown")
            }
            for node in concept_graph.nodes
        ],
        "edges": [
            {
                "from": edge[0],
                "to": edge[1],
                "relationship": concept_graph.edges[edge].get("relationship", "relates_to")
            }
            for edge in concept_graph.edges
        ]
    }
    return json.dumps(data, indent=2)

def export_graph_image():
    """Export current graph visualization as PNG file for viewing/download."""
    if len(concept_graph.nodes) == 0:
        return None

    # Get the current graph figure
    fig = render_graph()

    # Save to temporary file
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix='.png',
        prefix='concept_graph_'
    )

    # Save with high DPI for clarity
    fig.savefig(
        tmp.name,
        dpi=150,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close(fig)

    return tmp.name

def store_last_prompt(message: str) -> tuple:
    """Store the last prompt AND reveal the reuse button."""
    text = (message or "").strip()
    # Return the stored text plus a visibility update for the button
    return text, gr.update(visible=bool(text))

def reuse_last_prompt(last: str) -> str:
    return last or ""

def build_usage_stats_html():
    """Build the HTML string for the usage stats display."""
    # GPT-4o-mini pricing: ~$0.375/1M tokens (blended input/output estimate)
    estimated_cost = (total_tokens_used / 1_000_000) * 0.375
    stats = f"gpt-4o-mini · {total_api_calls} calls · {total_tokens_used:,} tokens · ${estimated_cost:.4f}"
    return f'<div style="color: #888; font-size: 12px; padding: 2px 0; margin-top: 4px; text-align: left;">{stats}</div>'


def build_turn_counter_html(count):
    """Build the HTML string for the message counter with hover tooltip."""
    remaining = MAX_TURNS - count
    tooltip = f"Demo preview: {MAX_TURNS} messages. Clone the repo locally for unlimited access."
    if remaining > 0:
        label = f"💬 {remaining} message{'s' if remaining != 1 else ''} left"
        return f'<div class="msg-counter" style="text-align: right;"><span class="info-icon" title="{tooltip}">{label} ⓘ</span></div>'
    else:
        label = "🔒 Demo limit reached"
        return f'<div class="msg-counter locked" style="text-align: right;"><span class="info-icon" title="{tooltip}">{label} ⓘ</span></div>'

# Build the Gradio interface using Blocks (more flexible than ChatInterface)
custom_css = """
/* Mobile responsiveness - stack columns on small screens */
@media (max-width: 768px) {
    .gradio-container .row {
        flex-direction: column !important;
    }
    .gradio-container .column {
        width: 100% !important;
    }
}

/* Compact message counter with hover tooltip */
.msg-counter {
    font-size: 13px;
    color: #666;
    white-space: nowrap;
    padding: 2px 0;
    margin-top: 2px;
    display: flex;
    align-items: center;
}
.msg-counter .info-icon {
    border-bottom: 1px dotted #999;
    cursor: help;
}
.msg-counter.locked {
    color: #c0392b;
}

/* Domain-tinted example buttons */
.domain-aiml button { background: #E8F4FD !important; border-color: #45B7D1 !important; }
.domain-aiml button:hover { background: #D1EAF8 !important; }
.domain-cogsci button { background: #FDE8E8 !important; border-color: #FF6B6B !important; }
.domain-cogsci button:hover { background: #FACFCF !important; }
.domain-health button { background: #E8F8F0 !important; border-color: #98D8C8 !important; }
.domain-health button:hover { background: #D0F0E3 !important; }
.domain-biz button { background: #FDF5E0 !important; border-color: #F7DC6F !important; }
.domain-biz button:hover { background: #FAECC5 !important; }
.domain-general button { background: #F0EEFA !important; border-color: #B0A4E3 !important; }
.domain-general button:hover { background: #E2DAEF !important; }
"""

# Map domain names to CSS class names
DOMAIN_CSS_CLASSES = {
    "AI/ML": "domain-aiml",
    "Cognitive Science": "domain-cogsci",
    "Healthcare": "domain-health",
    "Business": "domain-biz",
    "General": "domain-general",
}

with gr.Blocks(title="Concept Cartographer") as demo:
    
    gr.Markdown("""
    # 🗺️ Concept Cartographer

    **Interactive Knowledge Mapping** - Ask questions and watch concepts emerge as a graph

    *Built by Barbara Hidalgo-Sotelo | Cognitive Science + AI*
    """)

    # Top row: Input (left) + Domain (right) — aligned horizontally
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Explain what an ontology is and who cares about them",
            label="Plot a Thought",
            scale=1
        )
        domain = gr.Dropdown(
            choices=["AI/ML", "Cognitive Science", "Healthcare", "Business", "General"],
            value="AI/ML",
            label="Choose a Lens",
            scale=1
        )
        last_prompt = gr.State("")

    with gr.Row():
        submit_btn = gr.Button("Send ↑", variant="primary", size="lg")

    # "Try these" prompt buttons — update dynamically when domain changes
    gr.Markdown("### 💡 Try these to get started:")
    with gr.Row(elem_classes=["domain-aiml"]) as example_row:
        ex1 = gr.Button(DOMAIN_EXAMPLES["AI/ML"][0], size="sm")
        ex2 = gr.Button(DOMAIN_EXAMPLES["AI/ML"][1], size="sm")
        ex3 = gr.Button(DOMAIN_EXAMPLES["AI/ML"][2], size="sm")
        ex4 = gr.Button(DOMAIN_EXAMPLES["AI/ML"][3], size="sm")

    # Main 2-column layout: Conversation (left) + Graph (right)
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_label=True,
            )

            # Stats and counter on same line
            with gr.Row():
                usage_display = gr.HTML(value=build_usage_stats_html(), scale=2)
                turn_counter = gr.HTML(value=build_turn_counter_html(0), scale=1)

            # "Reuse last" — hidden until a prompt has been sent
            reuse_btn = gr.Button(
                "↩ Reuse last prompt",
                size="sm",
                variant="secondary",
                visible=False,
            )

        with gr.Column(scale=1):
            graph_plot = gr.Plot(label="Knowledge Graph", value=render_graph())

            # Action buttons below graph
            with gr.Row():
                clear_btn = gr.Button("Clear All", size="sm")
                view_tab_btn = gr.Button("Export Graph PNG", size="sm")
                export_btn = gr.Button("Export Graph JSON", size="sm")

            graph_file = gr.File(visible=False, label="Save Image", interactive=True)

            export_output = gr.Textbox(
                label="Graph Export (JSON)",
                lines=10,
                interactive=True,
                buttons=["copy"],
                visible=False
            )

            with gr.Accordion("How it works", open=True):
                gr.Markdown(
                    "1. **Chat** about any topic\n"
                    "2. **Watch** concepts get extracted\n"
                    "3. **See** relationships visualized\n"
                    "4. **Export** your ontology\n\n"
                    "*Colors represent concept categories. Arrows show relationships.*"
                )
    
    # Event handlers
    def add_user_message(message, chat_history, current_domain):
        """
        Add user message to chat history immediately (fast UI feedback),
        and add a single pending assistant bubble that will be replaced.

        Returns the message back into the textbox (not "") so the user
        can see what they asked. It gets overwritten when they type next.
        """
        if not message.strip():
            return message, chat_history

        new_history = list(chat_history)

        display_user = f"{message} — {current_domain} lens" if current_domain else message
        new_history.append({"role": "user", "content": display_user, "raw": message, "lens": current_domain})
        new_history.append({"role": "assistant", "content": PENDING_ASSISTANT_TEXT})
        return message, new_history

    def respond(chat_history, current_domain):
        """Handle chat with domain awareness and turn limiting."""
        global current_turn_count

        # Work on a fresh copy so Gradio sees a new list reference
        chat_history = list(chat_history)

        # Check if max turns reached - if so, don't process
        if current_turn_count >= MAX_TURNS:
            # Remove the pending ellipsis message
            if chat_history and chat_history[-1]["role"] == "assistant" and chat_history[-1]["content"] == PENDING_ASSISTANT_TEXT:
                chat_history = chat_history[:-1]
            return (
                chat_history,  # No update
                render_graph(),
                build_usage_stats_html(),
                build_turn_counter_html(current_turn_count),
                gr.update(interactive=False),  # Keep textbox disabled
                gr.update(interactive=False)   # Keep button disabled
            )

        # Guard: history must have at least [user_msg, pending_"..."] at the tail
        if (
            len(chat_history) < 2
            or chat_history[-1].get("role") != "assistant"
            or chat_history[-1].get("content") != PENDING_ASSISTANT_TEXT
            or chat_history[-2].get("role") != "user"
        ):
            # Unexpected state — remove dangling placeholder if present and bail
            if chat_history and chat_history[-1].get("content") == PENDING_ASSISTANT_TEXT:
                chat_history = chat_history[:-1]
            return (
                chat_history,
                render_graph(),
                build_usage_stats_html(),
                build_turn_counter_html(current_turn_count),
                gr.update(),
                gr.update(),
            )

        # Get the last user message (second to last, since last is the pending ellipsis)
        message = chat_history[-2].get("raw", chat_history[-2]["content"])

        # Get bot response (single structured call — returns formatted display text)
        display_text = chat_and_extract(message, chat_history[:-2], current_domain)

        # Replace the pending ellipsis placeholder with the formatted response.
        # Store the narrative portion so subsequent turns send only the concise
        # narrative as history context, not the full concept inventory.
        chat_history[-1]["content"] = display_text
        # Narrative is everything before the 📍 concept line
        raw_narrative = display_text.split("\n📍")[0].strip()
        chat_history[-1]["raw_summary"] = raw_narrative

        # Increment turn counter
        current_turn_count += 1

        # If we just hit the limit, add LinkedIn teaser message
        if current_turn_count >= MAX_TURNS:
            teaser_message = (
                "🎉 **Thanks for trying the demo!**\n\n"
                "You've reached the 3-message limit for this preview version. "
                "Want to explore unlimited concept mapping?\n\n"
                f"👉 **Comment on my [LinkedIn post]({LINKEDIN_POST_URL}) to get full access!**\n\n"
                "I'd love to hear what you think of this tool!"
            )
            chat_history.append({
                "role": "assistant",
                "content": teaser_message
            })

        # Determine if inputs should now be disabled
        inputs_disabled = current_turn_count >= MAX_TURNS

        return (
            chat_history,
            render_graph(),
            build_usage_stats_html(),
            build_turn_counter_html(current_turn_count),
            gr.update(interactive=not inputs_disabled),  # Disable msg textbox
            gr.update(interactive=not inputs_disabled)   # Disable submit button
        )
    
    def clear_all():
        """Clear chat, graph, and reset turn counter."""
        global total_tokens_used, total_api_calls, current_turn_count
        total_tokens_used = 0
        total_api_calls = 0
        current_turn_count = 0  # Reset counter

        return (
            [],  # Empty chat
            clear_graph(),  # Empty graph
            build_usage_stats_html(),
            build_turn_counter_html(0),
            gr.update(interactive=True),  # Re-enable msg
            gr.update(interactive=True),  # Re-enable submit
            "",                            # Reset last_prompt state
            gr.update(visible=False),      # Hide reuse button
        )
    
    def show_export():
        """Show the export text box with JSON."""
        json_data = export_graph()
        return gr.update(visible=True, value=json_data)

    
    # Wire up the events - two-step process for better UX
    # Step 1: Add user message immediately (shows user's query right away)
    # Step 2: Get bot response (shows loading, then response)
    submit_btn.click(
        store_last_prompt,
        inputs=msg,
        outputs=[last_prompt, reuse_btn],
        show_progress="hidden",
    ).then(
        add_user_message,
        inputs=[msg, chatbot, domain],
        outputs=[msg, chatbot],
        show_progress="hidden"
    ).then(
        respond,
        inputs=[chatbot, domain],
        outputs=[chatbot, graph_plot, usage_display, turn_counter, msg, submit_btn],
        show_progress="minimal"
    )

    msg.submit(
        store_last_prompt,
        inputs=msg,
        outputs=[last_prompt, reuse_btn],
        show_progress="hidden",
    ).then(
        add_user_message,
        inputs=[msg, chatbot, domain],
        outputs=[msg, chatbot],
        show_progress="hidden"
    ).then(
        respond,
        inputs=[chatbot, domain],
        outputs=[chatbot, graph_plot, usage_display, turn_counter, msg, submit_btn],
        show_progress="minimal"
    )
    reuse_btn.click(
        reuse_last_prompt,
        inputs=last_prompt,
        outputs=msg,
        show_progress=False
    )

    # ── Auto-submit helper: clicking an example button fills + sends in one click ──
    def _auto_submit_chain(btn):
        """Wire a button so it populates msg, then triggers the full submit pipeline."""
        btn.click(
            lambda text: text,              # copy button label → msg
            inputs=[btn],
            outputs=[msg],
            show_progress="hidden",
        ).then(
            store_last_prompt,
            inputs=msg,
            outputs=[last_prompt, reuse_btn],
            show_progress="hidden",
        ).then(
            add_user_message,
            inputs=[msg, chatbot, domain],
            outputs=[msg, chatbot],
            show_progress="hidden",
        ).then(
            respond,
            inputs=[chatbot, domain],
            outputs=[chatbot, graph_plot, usage_display, turn_counter, msg, submit_btn],
            show_progress="minimal",
        )

    for btn in [ex1, ex2, ex3, ex4]:
        _auto_submit_chain(btn)

    # ── Domain dropdown changes → update example button labels + color tint ──
    def update_example_labels(selected_domain):
        prompts = DOMAIN_EXAMPLES.get(selected_domain, DOMAIN_EXAMPLES["General"])
        css_class = DOMAIN_CSS_CLASSES.get(selected_domain, "domain-general")
        return (
            gr.update(value=prompts[0]),
            gr.update(value=prompts[1]),
            gr.update(value=prompts[2]),
            gr.update(value=prompts[3]),
            gr.update(elem_classes=[css_class]),
        )

    domain.change(
        update_example_labels,
        inputs=[domain],
        outputs=[ex1, ex2, ex3, ex4, example_row],
        show_progress="hidden",
    )
    
    clear_btn.click(
        clear_all,
        outputs=[chatbot, graph_plot, usage_display, turn_counter, msg, submit_btn, last_prompt, reuse_btn],
        show_progress="minimal"
    )
    
    export_btn.click(
        show_export,
        outputs=[export_output]
    )

    # Wire up the view in new tab button
    view_tab_btn.click(
        export_graph_image,
        outputs=[graph_file]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[graph_file]
    )

    def reset_on_load():
        """Reset all global state on page load so every visitor starts fresh."""
        global concept_graph, current_turn_count, total_tokens_used, total_api_calls
        concept_graph = nx.DiGraph()
        current_turn_count = 0
        total_tokens_used = 0
        total_api_calls = 0
        return (
            [],
            render_graph(),
            build_usage_stats_html(),
            build_turn_counter_html(0),
            gr.update(interactive=True),
            gr.update(interactive=True),
            "",                            # Reset last_prompt
            gr.update(visible=False),      # Hide reuse button
        )

    demo.load(
        reset_on_load,
        outputs=[chatbot, graph_plot, usage_display, turn_counter, msg, submit_btn, last_prompt, reuse_btn],
        show_progress="minimal"
    )

if __name__ == "__main__":
    demo.launch(
        head=FAVICON_HEAD,
        server_name="0.0.0.0",
        server_port=7860,
#        share=True,
        show_error=True,
        theme=gr.themes.Soft(),  # Moved here for Gradio 6.0
        css=custom_css  # CSS also moved to launch() for Gradio 6.0
    )