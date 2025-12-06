# Honey 🍯

A powerful prompt templating and LLM execution framework for Python. Write your prompts in `.hny` files, import them like Python modules, and execute them with stateful LLM conversations.

## Features

- 🎯 **Template-based prompts**: Write prompts in `.hny` files with Jinja2 templating
- 🔌 **Auto-importing**: Import `.hny` files as Python modules automatically
- 🤖 **Multi-LLM support**: OpenAI, Anthropic Claude, Google Gemini
- 🔄 **Stateful conversations**: Maintain conversation history across multiple turns
- ⚡ **Async support**: Full async/await support for concurrent operations
- 🧪 **Mock jar**: Built-in mock for testing without API calls

## Installation

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

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

import hive  # Auto-installs the .hny loader
import greetings

# Just renders the template
result = greetings.greet(name="Alice")
print(result)  # "Hello, Alice! How can I help you today?"
```

**With an LLM (executes via API):**

```python
from hive import openai_jar
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
from hive import openai_jar
import chat_prompts

jar = openai_jar(model="gpt-4", api_key="your-api-key")
jar.add_system_prompt("You are a helpful coding assistant")

with jar:
    response1 = chat_prompts.ask(question="What is Python?")
    response2 = chat_prompts.ask(question="Show me a function example")
    # Jar maintains conversation history across calls

# Access conversation history
history = jar.get_history()
print(f"Total messages: {jar.message_count}")
print(f"Total tokens: {jar.total_tokens}")
```

**Async support:**

```python
import asyncio
from hive import openai_jar
import prompts

async def main():
    jar = openai_jar(model="gpt-4", api_key="your-api-key")
    
    async with jar:
        # Execute prompts concurrently
        results = await asyncio.gather(
            prompts.analyze(data="dataset1"),
            prompts.analyze(data="dataset2"),
            prompts.analyze(data="dataset3")
        )
    
    print(results)

asyncio.run(main())
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
from hive import openai_jar

jar = openai_jar(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    api_key="your-key"
)
```

### Anthropic Claude

```python
from hive import anthropic_jar

jar = anthropic_jar(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=1000,
    api_key="your-key"
)
```

### Google Gemini

```python
from hive import gemini_jar

jar = gemini_jar(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    api_key="your-key"
)
```

### Mock Jar (for testing)

```python
from hive import mock_jar

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
uv run pytest tests/ --cov=experiments/hive --cov-report=html

# Run specific test file
uv run pytest tests/hive/test_jars.py -v

# Run specific test
uv run pytest tests/hive/test_jars.py::TestMockJar::test_execute_returns_mock_response -v
```

### Test Coverage

Current coverage: **81.89%** (71 tests passing)

- `experiments/hive/jars.py`: 80.25%
- `experiments/hive/loader.py`: 85.34%

View detailed coverage report:
```bash
uv run pytest tests/ --cov=experiments/hive --cov-report=html
open htmlcov/index.html
```

### Project Structure

```
honey/
├── experiments/
│   └── hive/
│       ├── __init__.py      # Package exports
│       ├── loader.py        # .hny file import system
│       └── jars.py          # LLM execution jars
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── fixtures/
│   │   └── prompts/         # Test .hny files
│   └── hive/
│       ├── test_loader.py         # Loader tests (23)
│       ├── test_loader_async.py   # Async loader tests (4)
│       ├── test_jars.py           # Jar tests (34)
│       ├── test_jars_async.py     # Async jar tests (11)
│       └── test_integration.py    # Integration tests (12)
├── pyproject.toml           # Project configuration
└── README.md
```

## How It Works

1. **Import Hook**: When you `import hive`, it installs a custom import hook (`HnyFinder`) in `sys.meta_path`
2. **File Discovery**: When you import a module, the hook searches `sys.path` for `.hny` files
3. **Template Loading**: Found `.hny` files are parsed and converted into Python functions
4. **Runtime Detection**: Each function checks if a jar is active via `contextvars`
5. **Execution**: 
   - No jar → renders template and returns string
   - Sync jar → calls `jar.execute(prompt)`
   - Async jar → returns coroutine for `jar.aexecute(prompt)`

## Examples

See `experiments/sample.py` for a complete working example.

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- All tests pass: `uv run pytest tests/`
- Coverage stays above 80%: `uv run pytest tests/ --cov=experiments/hive`
- Code follows existing patterns
