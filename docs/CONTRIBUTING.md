# Contributing to Concept Cartographer

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## 🎯 Project Vision

Concept Cartographer aims to bridge natural language conversation with formal knowledge representation. Contributions should align with these core principles:

1. **Visual over textual**: Show knowledge structures, don't just describe them
2. **Fast and impressive**: Prioritize user delight and quick wins
3. **Cognitively informed**: Leverage insights from cognitive science
4. **Practical engineering**: Build things that actually work and are maintainable

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- OpenAI API key (for testing)

### Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/concept-cartographer.git
cd concept-cartographer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Run the app
python concept_cartographer.py
```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 2. Make Changes

- Write clear, documented code
- Follow existing code style (see Style Guide below)
- Add docstrings to functions
- Update CLAUDE.md with technical insights if relevant

### 3. Test Your Changes

```bash
# Manual testing
python concept_cartographer.py

# Test different scenarios:
# - Various domains (AI/ML, Cognitive Science, etc.)
# - Edge cases (empty graphs, very long conversations)
# - Error cases (invalid API keys, network issues)
```

### 4. Commit

```bash
git add .
git commit -m "Brief description of changes"
```

**Commit message format:**
- Use present tense ("Add feature" not "Added feature")
- Be specific but concise
- Reference issues if applicable (#123)

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Screenshots if UI is affected
- Link to related issues
- Testing notes

## 🎨 Code Style Guide

### Python Style

Follow PEP 8 with these specifics:

```python
# Good: Clear function names with docstrings
def extract_concepts_and_relationships(text, domain="General"):
    """
    Extract concepts from text with domain awareness.
    
    Args:
        text: Input text to analyze
        domain: Domain focus for better extraction
        
    Returns:
        dict: Concepts and relationships in JSON format
    """
    pass

# Bad: Unclear names, no documentation
def extract(t, d="General"):
    pass
```

### Gradio Patterns

```python
# Good: Descriptive component names
msg = gr.Textbox(label="Ask a question")
submit_btn = gr.Button("Send", variant="primary")

# Bad: Generic names
tb1 = gr.Textbox()
btn = gr.Button("Send")
```

### Event Handlers

```python
# Good: Clear, single-purpose handlers
def respond(message, history, domain):
    """Handle chat and update graph."""
    bot_message = chat_and_extract(message, history, domain)
    history.append((message, bot_message))
    return "", history, render_graph()

# Bad: Doing too much in one function
def handle_everything(msg, hist, dom, graph, state, ...):
    # Too many responsibilities
    pass
```

## 🐛 Bug Reports

When filing a bug report, include:

1. **Description**: What happened vs what you expected
2. **Steps to reproduce**: Exact steps to trigger the bug
3. **Environment**: 
   - Python version
   - Gradio version (`pip show gradio`)
   - OS
4. **Error messages**: Full traceback if applicable
5. **Screenshots**: If UI is affected

**Example:**
```markdown
### Bug: Graph doesn't update after certain questions

**Environment:**
- Python 3.11
- Gradio 6.0.1
- macOS 14.0

**Steps:**
1. Select "AI/ML" domain
2. Ask "What is supervised learning?"
3. Ask "How does it differ from unsupervised?"
4. Graph shows first concepts but not second

**Expected:** Graph should show both sets of concepts

**Actual:** Second extraction doesn't appear

**Error log:**
[paste error if any]
```

## ✨ Feature Requests

For feature requests, describe:

1. **Problem**: What user need does this address?
2. **Proposed solution**: How would it work?
3. **Alternatives**: Other ways to solve this?
4. **Priority**: Nice-to-have vs critical?

**Example:**
```markdown
### Feature: Export graph as PNG image

**Problem:** Users want to share their knowledge graphs in presentations/documents

**Proposed solution:** Add "Export as Image" button that saves the matplotlib figure as PNG

**Alternatives:**
- Export as SVG (scalable)
- Export as interactive HTML (D3.js)

**Priority:** Medium - enhances shareability but not blocking
```

## 🎯 Priority Areas for Contribution

### High Impact, Low Effort
- [ ] Additional domain presets (Science, Education, Philosophy)
- [ ] Graph statistics panel (metrics about the knowledge graph)
- [ ] Different color schemes/themes
- [ ] Keyboard shortcuts (Enter to send, Ctrl+K to clear)

### Medium Effort
- [ ] Support for Anthropic Claude API
- [ ] Export to additional formats (GraphML, PNG, SVG)
- [ ] Interactive graph (zoom, pan, click nodes)
- [ ] Undo/redo functionality

### High Effort, High Value
- [ ] User authentication
- [ ] Database persistence (PostgreSQL + pgvector)
- [ ] Graph search and filtering
- [ ] Multi-user collaboration
- [ ] Integration with knowledge bases (Neo4j, Weaviate)

## 📝 Documentation

When adding features:

1. **Update README.md** if user-facing
2. **Update CLAUDE.md** with technical insights
3. **Add inline comments** for complex logic
4. **Write docstrings** for all functions

### Documentation Style

```python
def new_feature(param1, param2):
    """
    One-line summary of what this does.
    
    More detailed explanation if needed. Explain why certain
    decisions were made, not just what the code does.
    
    Args:
        param1: Description and type
        param2: Description and type
        
    Returns:
        Description of return value
        
    Raises:
        ErrorType: When this error occurs
    """
```

## 🧪 Testing Guidelines

Currently, testing is manual. When adding features:

1. **Test the happy path**: Does it work as intended?
2. **Test edge cases**: Empty inputs, very long inputs, etc.
3. **Test error cases**: What if API fails? Network down?
4. **Test across domains**: Works for all domain types?

**Future**: We'll add automated tests (pytest). For now, thorough manual testing is essential.

## 🔍 Code Review Process

When your PR is submitted:

1. **Automated checks**: Will run when we add CI/CD
2. **Manual review**: Maintainer will review code
3. **Feedback**: May request changes
4. **Approval**: Once approved, will be merged

**Review criteria:**
- Code quality and style
- Functionality works as described
- No breaking changes (unless discussed)
- Documentation is updated
- Performance impact is acceptable

## 🤝 Community Guidelines

- Be respectful and constructive
- Help others learn and grow
- Share knowledge and insights
- Give credit where due
- Focus on the work, not the person

## 📧 Questions?

- **Technical questions**: See CLAUDE.md or open a discussion
- **Bug reports**: Open an issue
- **Feature ideas**: Open an issue with "Feature Request" label
- **General questions**: Start a discussion on GitHub

## 🙏 Thank You

Every contribution helps make Concept Cartographer better. Whether it's:
- Reporting a bug
- Suggesting a feature
- Fixing a typo
- Adding a feature
- Improving documentation

Your effort is appreciated! 🎉

---

**Happy Contributing!** 🗺️
