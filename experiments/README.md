# Hive - Prompt Templating & LLM Execution Framework

Hive enables you to write prompts as `.hny` files and import them like Python modules, with seamless LLM execution via **jars** (runtime contexts).

## Features

- 🍯 **Import `.hny` files as Python modules** - Write prompts in simple template files
- 🫙 **Jar runtimes** - Execute prompts with LLMs (OpenAI, Anthropic) using context managers
- 🔄 **Sync & Async support** - Works with both `with jar:` and `async with jar:`
- 📊 **Stateful conversations** - Jars maintain conversation history and state
- 🎯 **Reusable** - Create jar instances and reuse across multiple contexts

## Quick Start

### 1. Create a `.hny` prompt file

```hny
# prompts/demo.hny
summarize
Summarize the following text:
{{text}}
----
```

### 2. Import and use (without LLM)

```python
import hive
from prompts.demo import summarize

# Returns rendered template string
template = summarize(text="Long article here...")
print(template)
# Output: "Summarize the following text:\nLong article here..."
```

### 3. Execute with LLM using jars

```python
from hive import openai_jar
from prompts.demo import summarize

# Create a jar with configuration
jar = openai_jar(model="gpt-4", temperature=0.7)

# Execute with LLM
with jar:
    response = summarize(text="Long article here...")
    print(response)  # Actual GPT-4 response
```

## Jar Types

- **`mock_jar`** - Mock runtime for testing (echoes prompts back)
- **`openai_jar`** - OpenAI API (GPT-4, etc.)
- **`anthropic_jar`** - Anthropic API (Claude, etc.)
- **`gemini_jar`** - Google Gemini API (Gemini 2.0, etc.)

## Stateful Chat Example

```python
from hive import openai_jar
from prompts.chat import respond

# Create reusable jar
jar = openai_jar(model="gpt-4")

# Add system prompt
jar.add_system_prompt("You are a helpful coding assistant.")

# Chat loop - jar maintains conversation history
for _ in range(5):
    user_msg = input("You: ")
    
    with jar:
        response = respond(message=user_msg)
    
    print(f"Assistant: {response}")

# Check conversation state
print(f"Total messages: {jar.message_count}")
print(f"Tokens used: {jar.total_tokens}")
```

## Async Support

```python
import asyncio
from hive import anthropic_jar
from prompts.demo import summarize

async def main():
    jar = anthropic_jar(model="claude-3-5-sonnet-20241022")
    
    # Must await in async context
    async with jar:
        response = await summarize(text="Long article...")
    
    print(response)

asyncio.run(main())
```

## Concurrent Requests

```python
import asyncio
from hive import openai_jar
from prompts.demo import summarize

async def process_many():
    jar = openai_jar(model="gpt-4")
    
    tasks = []
    for text in articles:
        async def task():
            async with jar:
                return await summarize(text=text)
        tasks.append(task())
    
    results = await asyncio.gather(*tasks)
    return results
```

## Nested Jars

Inner jars override outer jars:

```python
from hive import openai_jar, anthropic_jar

outer = openai_jar(model="gpt-3.5-turbo")
inner = anthropic_jar(model="claude-3-5-sonnet-20241022")

with outer:
    result1 = summarize(text="...")  # Uses GPT-3.5
    
    with inner:
        result2 = summarize(text="...")  # Uses Claude
    
    result3 = summarize(text="...")  # Back to GPT-3.5
```

## Jar API

### Configuration
```python
jar = openai_jar(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    api_key="sk-..."  # Or set OPENAI_API_KEY env var
)
```

### State Management
```python
jar.add_system_prompt(content)    # Add system prompt
jar.add_message(role, content)    # Add message manually
jar.get_history()                 # Get conversation history
jar.clear_history()               # Clear history and reset counters
jar.message_count                 # Number of messages
jar.total_tokens                  # Total tokens used
```

## .hny File Format

```hny
prompt_name
Template content with {{variables}}
Use Jinja2 syntax for logic
---

another_prompt
Another template
{{variable}}
--------
```

Prompts are separated by 3 or more dashes (`---`).

## Examples

Run the examples:
```bash
uv run example_sync.py    # Synchronous examples
uv run example_async.py   # Asynchronous examples
```

## Installation

Add to your `pyproject.toml`:
```toml
dependencies = [
    "jinja2>=3.1.6",
    "openai",                  # Optional: for openai_jar
    "anthropic",               # Optional: for anthropic_jar
    "google-generativeai",     # Optional: for gemini_jar
]
```
