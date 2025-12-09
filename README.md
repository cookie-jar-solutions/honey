# Honey 🍯

A powerful prompt templating and LLM execution framework for Python. Write your prompts in `.hny` files, import them like Python modules, and execute them with stateful LLM conversations using **jars** (runtime contexts).

## Features

- 🍯 **Import `.hny` files as Python modules** - Write prompts in simple template files with Jinja2
- 🫙 **Jar runtimes** - Execute prompts with LLMs using context managers
- 🤖 **Multi-LLM support**: OpenAI GPT, Anthropic Claude, Google Gemini
- 🔄 **Stateful conversations**: Jars maintain conversation history across multiple turns
- ⚡ **Async support**: Full async/await support for concurrent operations
- 🎯 **Reusable & composable**: Create jar instances and reuse across contexts
- 🧪 **Mock jar**: Built-in mock for testing without API calls

## Installation

```bash
# Install base package with uv (recommended)
uv pip install -e .

# Install with specific LLM providers
uv pip install -e . openai                  # For OpenAI GPT models
uv pip install -e . anthropic               # For Anthropic Claude
uv pip install -e . google-generativeai     # For Google Gemini

# Or with pip
pip install -e .
```

**Dependencies:**
- Core: `jinja2>=3.1.6` (required)
- Optional: `openai`, `anthropic`, `google-generativeai` (install as needed)

## Quick Start

### 1. Create a prompt file

Create `prompts/greetings.hny`:

```
greet
Hello, {{name}}! How can I help you today?
---
farewell
Goodbye, {{name}}! Have a great day!
```

### 2. Use it in your code

**Without an LLM (template rendering only):**

```python
import sys
sys.path.insert(0, 'prompts')

import honey  # Auto-installs the .hny loader
import greetings

# Just renders the template
result = greetings.greet(name="Alice")
print(result)  # "Hello, Alice! How can I help you today?"
```

**With an LLM (executes via API):**

```python
from honey import openai_jar
import greetings

# Create a jar with configuration
jar = openai_jar(
    model="gpt-4",
    temperature=0.7,
    api_key="your-api-key"
)

# Use context manager to activate the jar
with jar:
    # Now the prompt executes with OpenAI
    response = greetings.greet(name="Alice")
    print(response)  # AI-generated response
```

**Stateful multi-turn conversations:**

```python
from honey import openai_jar
import chat_prompts

# Create reusable jar
jar = openai_jar(model="gpt-4", api_key="your-api-key")
jar.add_system_prompt("You are a helpful coding assistant")

# Chat loop - jar maintains conversation history
for _ in range(5):
    user_msg = input("You: ")
    
    with jar:
        response = chat_prompts.respond(message=user_msg)
    
    print(f"Assistant: {response}")

# Access conversation history and stats
history = jar.get_history()
print(f"Total messages: {jar.message_count}")
print(f"Total tokens: {jar.total_tokens}")
```

**Async support:**

```python
import asyncio
from honey import openai_jar
import prompts

async def main():
    jar = openai_jar(model="gpt-4", api_key="your-api-key")
    
    # Execute prompts concurrently
    async with jar:
        results = await asyncio.gather(
            prompts.analyze(data="dataset1"),
            prompts.analyze(data="dataset2"),
            prompts.analyze(data="dataset3")
        )
    
    print(results)

asyncio.run(main())
```

**Nested jars (inner overrides outer):**

```python
from honey import openai_jar, anthropic_jar
from prompts.demo import summarize

outer = openai_jar(model="gpt-3.5-turbo")
inner = anthropic_jar(model="claude-3-5-sonnet-20241022")

with outer:
    result1 = summarize(text="...")  # Uses GPT-3.5
    
    with inner:
        result2 = summarize(text="...")  # Uses Claude
    
    result3 = summarize(text="...")  # Back to GPT-3.5
```

## .hny File Format

`.hny` files contain one or more prompt templates:

```
prompt_name
Template content with {{variables}}
---
another_prompt
Another template with {{variable1}} and {{variable2}}
```

- First line: prompt name (becomes the function name)
- Remaining lines: Jinja2 template
- `---`: Separator between prompts (3+ dashes)

**Advanced Jinja2 features:**

```
conditional_prompt
{% if premium %}
Premium feature: {{feature}}
{% else %}
Basic feature available
{% endif %}
---
loop_prompt
{% for item in items %}
- {{item}}
{% endfor %}
```

## Available Jars

### OpenAI

```python
from honey import openai_jar

jar = openai_jar(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    api_key="your-key"
)
```

### Anthropic Claude

```python
from honey import anthropic_jar

jar = anthropic_jar(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=1000,
    api_key="your-key"
)
```

### Google Gemini

```python
from honey import gemini_jar

jar = gemini_jar(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    api_key="your-key"
)
```

### Mock Jar (for testing)

```python
from honey import mock_jar

jar = mock_jar()
with jar:
    response = prompts.test(data="sample")
    # Returns "[MOCK RESPONSE]\nPrompt: ..."
```

## Jar Methods

```python
jar = openai_jar(...)

# Add system prompt
jar.add_system_prompt("You are a helpful assistant")

# Add messages manually
jar.add_message("user", "Hello")
jar.add_message("assistant", "Hi there!")

# Get conversation history
history = jar.get_history()  # List of {role, content} dicts

# Clear history and reset
jar.clear_history()

# Check stats
print(jar.message_count)  # Number of messages
print(jar.total_tokens)   # Total tokens used
```

## Development

### Setup

```bash
# Install with development dependencies
uv pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=honey --cov-report=html

# Run specific test file
uv run pytest tests/honey/test_jars.py -v

# Run specific test
uv run pytest tests/honey/test_jars.py::TestMockJar::test_execute_returns_mock_response -v
```

### Test Coverage

Current coverage: **81.89%** (84 tests passing)

- `honey/jars.py`: 80.25%
- `honey/loader.py`: 85.34%

View detailed coverage report:
```bash
uv run pytest tests/ --cov=honey --cov-report=html
open htmlcov/index.html
```

### Project Structure

```
honey/
├── src/
│   └── honey/           # Main package (import honey)
│       ├── __init__.py      # Package exports
│       ├── loader.py        # .hny file import system
│       └── jars.py          # LLM execution jars
├── examples/
│   └── prompts/         # Example .hny files
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── fixtures/
│   │   └── prompts/         # Test .hny files
│   └── honey/
│       ├── test_loader.py         # Loader tests (23)
│       ├── test_loader_async.py   # Async loader tests (4)
│       ├── test_jars.py           # Jar tests (34)
│       ├── test_jars_async.py     # Async jar tests (11)
│       └── test_integration.py    # Integration tests (12)
├── pyproject.toml           # Project configuration
└── README.md
```

## Advanced Features

### Concurrent Async Execution

```python
import asyncio
from honey import anthropic_jar
from prompts.demo import summarize

async def process_many(articles):
    jar = anthropic_jar(model="claude-3-5-sonnet-20241022")
    
    async def process(text):
        async with jar:
            return await summarize(text=text)
    
    results = await asyncio.gather(*[process(text) for text in articles])
    return results
```

### Jinja2 Advanced Templates

```hny
conditional_prompt
{% if premium %}
Premium analysis for: {{topic}}
Include advanced insights.
{% else %}
Basic analysis for: {{topic}}
{% endif %}
---

loop_prompt
Analyze the following items:
{% for item in items %}
- {{item}}
{% endfor %}
```

## How It Works

1. **Import Hook**: When you `import honey`, it automatically installs a custom import hook (`HnyFinder`) in `sys.meta_path`
2. **File Discovery**: When you import a module, the hook searches `sys.path` for matching `.hny` files
3. **Template Parsing**: Found `.hny` files are parsed and each prompt becomes a callable Python function
4. **Runtime Detection**: Each function checks for active jars using `contextvars`
5. **Smart Execution**: 
   - No jar → renders Jinja2 template and returns string
   - Sync jar → calls `jar.execute(prompt)` with LLM
   - Async jar → returns coroutine for `await jar.aexecute(prompt)`

The framework uses context variables to track active jars separately for sync and async contexts, enabling proper isolation and concurrent execution.

## API Keys

Set API keys via environment variables or pass directly to jars:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

Or pass directly:
```python
jar = openai_jar(model="gpt-4", api_key="sk-...")
```

## Examples

See the `examples/` directory for complete working code:
- `examples/prompts/` - Sample `.hny` prompt files
- Other example scripts demonstrating various features

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- All tests pass: `uv run pytest tests/`
- Coverage stays above 80%: `uv run pytest tests/ --cov=honey`
- Code follows existing patterns
