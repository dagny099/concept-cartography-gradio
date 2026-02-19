import gradio as gr
import os
import json
from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
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
LINKEDIN_POST_URL = "https://www.linkedin.com/in/barbara-hidalgo-sotelo/"  # Set to your LinkedIn post URL if desired


def adaptive_concept_cap(existing_node_count: int) -> int:
    """Adaptive cap to keep the graph readable as it grows."""
    if existing_node_count < 25:
        return 12
    if existing_node_count < 60:
        return 8
    return 6


def extract_concepts_and_relationships(text, domain="General", max_concepts=8):
    """
    Use LLM to extract concepts and relationships from text.
    Optimized version with domain awareness, structured outputs, and robust error handling.
    
    Args:
        text: Conversation text to analyze
        domain: Domain focus for better extraction (AI/ML, Cognitive Science, etc.)
        
    Returns: dict with 'concepts' and 'relationships'
    """
    global total_tokens_used, total_api_calls
    
    # Domain-specific extraction hints with category guidance
    domain_hints = {
        "AI/ML": "Include algorithms (Method), architectures (Entity), theories (Theory), and processes (Process)",
        "Cognitive Science": "Include brain regions (Entity), cognitive theories (Theory), mental processes (Process), and properties (Property)",
        "Healthcare": "Include conditions (Entity), treatments (Method), symptoms (Property), and processes (Process)",
        "Business": "Include processes (Process), metrics (Property), strategies (Method), and stakeholders (Entity)",
        "General": "Include diverse types: entities, processes, theories, methods, and properties"
    }
    
    hint = domain_hints.get(domain, domain_hints["General"])
    
    # Simplified prompt to reduce JSON errors
    prompt = f"""Extract key concepts and relationships. {hint}

Text: {text}

Return valid JSON with this exact structure:
{{
  "concepts": [
    {{"name": "Concept1", "category": "Entity"}},
    {{"name": "Concept2", "category": "Process"}}
  ],
  "relationships": [
    {{"from": "Concept1", "to": "Concept2", "type": "enables"}}
  ]
}}

Rules:
- Up to {max_concepts} concepts maximum
- Use varied categories: Entity, Process, Theory, Method, Property
- Use specific relationship types: causes, requires, part_of, type_of, enables
- Keep concept names short (1-3 words)
- Avoid quotes and special characters in names"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise ontology extractor. Return ONLY valid JSON, nothing else."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        
        # Track usage
        total_tokens_used += response.usage.total_tokens
        total_api_calls += 1
        
        content = response.choices[0].message.content.strip()
        
        # Clean common JSON issues before parsing
        content = content.replace("'", '"')  # Replace single quotes
        content = content.replace('\n', ' ')  # Remove newlines that might break strings
        
        # Parse JSON
        result = json.loads(content)
        
        # Validate and clean structure
        if "concepts" not in result:
            result["concepts"] = []
        if "relationships" not in result:
            result["relationships"] = []
            
        # Clean concept names (remove problematic characters)
        for concept in result.get("concepts", []):
            if "name" in concept:
                # Remove quotes and clean
                concept["name"] = concept["name"].replace('"', '').replace("'", "").strip()
            if "category" not in concept:
                concept["category"] = "Entity"  # Default fallback
                
        return result
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing error: {e}")
        if 'content' in locals():
            print(f"Raw content: {content[:300]}...")
        
        # Return empty structure as fallback
        return {
            "concepts": [],
            "relationships": []
        }
        
    except Exception as e:
        print(f"⚠️ Extraction error: {e}")
        return {
            "concepts": [],
            "relationships": []
        }

def chat_and_extract(message, history, domain="General"):
    """
    Main chat function that also extracts concepts.
    Now domain-aware for better concept extraction.

    Returns both the chat response AND extracted concepts.
    """
    # Build conversation history for context
    messages = [{"role": "system", "content": ("You are a helpful assistant that explains concepts clearly and concisely. "f"Use a {domain} framing lens: pick examples, terminology, and emphasis that fit this domain.")}]

    # Handle Gradio 6.0 messages format (list of dicts with 'role' and 'content')
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": message})

    global total_tokens_used, total_api_calls

    # Get chat response (using cost-optimized model)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=400
    )

    # Track usage
    total_tokens_used += response.usage.total_tokens
    total_api_calls += 1

    assistant_response = response.choices[0].message.content
    
    # Extract concepts from the full conversation turn
    full_context = f"User: {message}\nAssistant: {assistant_response}"
    existing_node_count = len(concept_graph.nodes)
    max_concepts = adaptive_concept_cap(existing_node_count)
    extracted = extract_concepts_and_relationships(full_context, domain, max_concepts=max_concepts)

    # Update the global graph
    update_graph(extracted)
    
    return assistant_response

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
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix='.png',
        prefix='concept_graph_'
    )

    # Save with high DPI for clarity
    fig.savefig(
        temp_file.name,
        dpi=150,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close(fig)

    return temp_file.name

def get_usage_stats():
    """Get current token usage statistics and estimated cost."""
    # GPT-4o-mini pricing (as of 2025): $0.15/1M input tokens, $0.60/1M output tokens
    # Using average estimate of 0.375/1M tokens (assuming ~50% input/output split)
    estimated_cost = (total_tokens_used / 1_000_000) * 0.375
    return f"gpt-4o-mini · {total_api_calls} calls · {total_tokens_used:,} tokens · ${estimated_cost:.4f}"


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


def build_usage_stats_html():
    """Build the HTML string for the usage stats display."""
    stats = get_usage_stats()
    return f'<div style="color: #888; font-size: 12px; padding: 2px 0; margin-top: 4px; text-align: left;">{stats}</div>'

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
"""

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
            #info="Ask a question or enter an idea",
            scale=1
        )
        domain = gr.Dropdown(
            choices=["AI/ML", "Cognitive Science", "Healthcare", "Business", "General"],
            value="AI/ML",
            label="Choose a Lens",
            #info="Tailors concept extraction to your field (e.g., prioritizes algorithms for AI/ML, theories for Cognitive Science)",
            scale=1
        )

    # Send button centered below input
    with gr.Row():
        submit_btn = gr.Button("Send ↑", variant="primary", size="lg")

    # "Try these" prompt buttons — full-width row
    gr.Markdown("### 💡 Try these to get started:")
    with gr.Row():
        ex1 = gr.Button("How do neural networks learn?", size="sm")
        ex2 = gr.Button("Explain attention mechanisms", size="sm")
        ex3 = gr.Button("What is transfer learning?", size="sm")
        ex4 = gr.Button("How does RL work?", size="sm")

    # Main 2-column layout: Conversation (left) + Graph (right)
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="Conversation",  #💬
                height=400,
                show_label=True,
            )

            # Stats and counter on same line (left/right justified)
            with gr.Row():
                usage_display = gr.HTML(value=build_usage_stats_html(), scale=2)
                turn_counter = gr.HTML(value=build_turn_counter_html(0), scale=1)

            # "How it works" below stats
            gr.Markdown("""
            ### How it works:
            1. **Chat** about any topic
            2. **Watch** concepts get extracted
            3. **See** relationships visualized
            4. **Export** your ontology
            """)

        with gr.Column(scale=1):
            graph_plot = gr.Plot(label=" Knowledge Graph", value=render_graph()) #📊 
 
            # Action buttons below graph
            with gr.Row():
                clear_btn = gr.Button("Clear All", size="sm")
                view_tab_btn = gr.Button("Export Graph PNG", size="sm")
                export_btn = gr.Button("Export Graph JSON", size="sm")

            graph_file = gr.File(visible=False, label="Save Image", interactive=True)

            # Hidden export textbox
            export_output = gr.Textbox(
                label="Graph Export (JSON)",
                lines=10,
                interactive=True,
                buttons=["copy"],
                visible=False
            )

    # Note of explanation
    gr.Markdown("""*Colors represent concept categories. Arrows show relationships.*""")
    
    # Event handlers
    def add_user_message(message, chat_history):
        """Add user message to chat history immediately, with pending indicator."""
        if not message.strip():
            return "", chat_history
        chat_history.append({"role": "user", "content": message})
        # Add a pending message that will be replaced by the actual response
        chat_history.append({"role": "assistant", "content": "..."})
        return "", chat_history

    def respond(chat_history, current_domain):
        """Handle chat with domain awareness and turn limiting."""
        global current_turn_count

        # Check if max turns reached - if so, don't process
        if current_turn_count >= MAX_TURNS:
            # Remove the pending "..." message
            if chat_history and chat_history[-1]["role"] == "assistant" and chat_history[-1]["content"] == "...":
                chat_history = chat_history[:-1]
            return (
                chat_history,  # No update
                render_graph(),
                build_usage_stats_html(),
                build_turn_counter_html(current_turn_count),
                gr.update(interactive=False),  # Keep textbox disabled
                gr.update(interactive=False)   # Keep button disabled
            )

        # Get the last user message (second to last, since last is the "..." placeholder)
        message = chat_history[-2]["content"]

        # Get bot response (this takes time)
        bot_message = chat_and_extract(message, chat_history[:-2], current_domain)

        # Replace the "..." placeholder with the actual response
        chat_history[-1]["content"] = bot_message

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
            gr.update(interactive=True)   # Re-enable submit
        )
    
    def show_export():
        """Show the export text box with JSON."""
        json_data = export_graph()
        return gr.update(visible=True, value=json_data)

    
    # Wire up the events - two-step process for better UX
    # Step 1: Add user message immediately (shows user's query right away)
    # Step 2: Get bot response (shows loading, then response)
    submit_btn.click(
        add_user_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        show_progress=False
    ).then(
        respond,
        inputs=[chatbot, domain],
        outputs=[chatbot, graph_plot, usage_display, turn_counter, msg, submit_btn],
        show_progress="minimal"
    )

    msg.submit(
        add_user_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        show_progress=False
    ).then(
        respond,
        inputs=[chatbot, domain],
        outputs=[chatbot, graph_plot, usage_display, turn_counter, msg, submit_btn],
        show_progress="minimal"
    )
    
    # Wire up example buttons
    ex1.click(lambda: "How do neural networks learn?", outputs=[msg])
    ex2.click(lambda: "Explain attention mechanisms in transformers", outputs=[msg])
    ex3.click(lambda: "What is transfer learning?", outputs=[msg])
    ex4.click(lambda: "How does reinforcement learning work?", outputs=[msg])
    
    clear_btn.click(
        clear_all,
        outputs=[chatbot, graph_plot, usage_display, turn_counter, msg, submit_btn],
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
        )

    demo.load(
        reset_on_load,
        outputs=[chatbot, graph_plot, usage_display, turn_counter, msg, submit_btn],
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
#        share=True,
        show_error=True,
        theme=gr.themes.Soft(),  # Moved here for Gradio 6.0
        css=custom_css  # CSS also moved to launch() for Gradio 6.0
    )
