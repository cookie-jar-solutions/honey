"""Hive - Prompt templating and LLM execution framework.

Import hive to automatically enable .hny file loading and access to jar runtimes.

Usage:
    import hive
    from hive import openai_jar, anthropic_jar, gemini_jar, mock_jar
    from prompts.demo import summarize
    
    # Use without jar - returns rendered template
    template = summarize(text="...")
    
    # Use with jar - executes with LLM
    with openai_jar(model="gpt-4"):
        response = summarize(text="...")
"""

from . import loader
from . import jars
from .jars import OpenAIJar as openai_jar
from .jars import AnthropicJar as anthropic_jar
from .jars import GeminiJar as gemini_jar
from .jars import MockJar as mock_jar
from .jars import OpenAICompatibleJar as openai_compatible_jar

# Auto-install the loader when hive is imported
loader.install()

__all__ = [
    'loader',
    'jars',
    'openai_jar',
    'anthropic_jar',
    'gemini_jar',
    'mock_jar',
    'openai_compatible_jar',
]