"""LLM Runtime Jars for executing prompts with language models.

Jars are reusable, stateful context managers that execute prompts against LLM APIs.
They support both synchronous and asynchronous execution modes.
"""

import contextvars
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod


# Context variables to track active jars (separate for sync and async)
_sync_jar: contextvars.ContextVar[Optional['Jar']] = contextvars.ContextVar(
    '_sync_jar', default=None
)
_async_jar: contextvars.ContextVar[Optional['Jar']] = contextvars.ContextVar(
    '_async_jar', default=None
)


class Jar(ABC):
    """Base class for LLM runtime jars.
    
    Jars are reusable context managers that maintain conversation state
    and execute prompts against LLM APIs in both sync and async modes.
    """
    
    def __init__(self, **config):
        """Initialize jar with configuration.
        
        Args:
            **config: Runtime configuration (model, temperature, api_key, etc.)
        """
        self.config = config
        self.history: List[Dict[str, str]] = []
        self.total_tokens = 0
        self.message_count = 0
        self._sync_token = None
        self._async_token = None
    
    @abstractmethod
    def execute(self, prompt: str, **metadata) -> str:
        """Execute a prompt synchronously and return the LLM response.
        
        Args:
            prompt: The rendered prompt string
            **metadata: Additional metadata (template, function name, etc.)
            
        Returns:
            The LLM response string
        """
        pass
    
    @abstractmethod
    async def aexecute(self, prompt: str, **metadata) -> str:
        """Execute a prompt asynchronously and return the LLM response.
        
        Args:
            prompt: The rendered prompt string
            **metadata: Additional metadata (template, function name, etc.)
            
        Returns:
            The LLM response string
        """
        pass
    
    def __enter__(self):
        """Enter synchronous context."""
        self._sync_token = _sync_jar.set(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit synchronous context."""
        _sync_jar.reset(self._sync_token)
        return False
    
    async def __aenter__(self):
        """Enter asynchronous context."""
        self._async_token = _async_jar.set(self)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit asynchronous context."""
        _async_jar.reset(self._async_token)
        return False
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
        """
        self.history.append({"role": role, "content": content})
        self.message_count += 1
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self.history.copy()
    
    def clear_history(self):
        """Clear the conversation history and reset counters."""
        self.history.clear()
        self.message_count = 0
        self.total_tokens = 0
    
    def add_system_prompt(self, content: str):
        """Add a system prompt to the conversation.
        
        Args:
            content: System prompt content
        """
        self.add_message("system", content)


class MockJar(Jar):
    """Mock jar for testing - echoes back the prompt."""
    
    def execute(self, prompt: str, **metadata) -> str:
        """Return a mock response."""
        self.add_message("user", prompt)
        response = f"[MOCK RESPONSE]\nPrompt: {prompt[:100]}..."
        self.add_message("assistant", response)
        return response
    
    async def aexecute(self, prompt: str, **metadata) -> str:
        """Return a mock response asynchronously."""
        self.add_message("user", prompt)
        response = f"[ASYNC MOCK RESPONSE]\nPrompt: {prompt[:100]}..."
        self.add_message("assistant", response)
        return response


class OpenAIJar(Jar):
    """Jar that uses OpenAI API for LLM execution."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI jar.
        
        Args:
            model: OpenAI model to use (default: gpt-4)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            **kwargs: Additional OpenAI API parameters (temperature, max_tokens, etc.)
        """
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._client = None
        self._async_client = None
    
    def _get_client(self):
        """Lazy load OpenAI sync client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.config.get('api_key'))
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install it with: pip install openai"
                )
        return self._client
    
    def _get_async_client(self):
        """Lazy load OpenAI async client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(api_key=self.config.get('api_key'))
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install it with: pip install openai"
                )
        return self._async_client
    
    def execute(self, prompt: str, **metadata) -> str:
        """Execute prompt using OpenAI API synchronously."""
        client = self._get_client()
        
        # Add user message to history
        self.add_message("user", prompt)
        
        # Prepare API call
        api_kwargs = {k: v for k, v in self.config.items() if k not in ['api_key']}
        
        response = client.chat.completions.create(
            messages=self.history,
            **api_kwargs
        )
        
        # Extract response and update state
        assistant_message = response.choices[0].message.content
        self.add_message("assistant", assistant_message)
        self.total_tokens += response.usage.total_tokens
        
        return assistant_message
    
    async def aexecute(self, prompt: str, **metadata) -> str:
        """Execute prompt using OpenAI API asynchronously."""
        client = self._get_async_client()
        
        # Add user message to history
        self.add_message("user", prompt)
        
        # Prepare API call
        api_kwargs = {k: v for k, v in self.config.items() if k not in ['api_key']}
        
        response = await client.chat.completions.create(
            messages=self.history,
            **api_kwargs
        )
        
        # Extract response and update state
        assistant_message = response.choices[0].message.content
        self.add_message("assistant", assistant_message)
        self.total_tokens += response.usage.total_tokens
        
        return assistant_message


class AnthropicJar(Jar):
    """Jar that uses Anthropic API for LLM execution."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, **kwargs):
        """Initialize Anthropic jar.
        
        Args:
            model: Anthropic model to use (default: claude-3-5-sonnet-20241022)
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            **kwargs: Additional Anthropic API parameters (temperature, max_tokens, etc.)
        """
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._client = None
        self._async_client = None
    
    def _get_client(self):
        """Lazy load Anthropic sync client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.config.get('api_key'))
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install it with: pip install anthropic"
                )
        return self._client
    
    def _get_async_client(self):
        """Lazy load Anthropic async client."""
        if self._async_client is None:
            try:
                from anthropic import AsyncAnthropic
                self._async_client = AsyncAnthropic(api_key=self.config.get('api_key'))
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install it with: pip install anthropic"
                )
        return self._async_client
    
    def _prepare_messages(self) -> tuple[Optional[str], List[Dict[str, str]]]:
        """Prepare messages for Anthropic API (system prompt separate).
        
        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        messages = []
        
        for msg in self.history:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                messages.append(msg)
        
        return system_prompt, messages
    
    def execute(self, prompt: str, **metadata) -> str:
        """Execute prompt using Anthropic API synchronously."""
        client = self._get_client()
        
        # Add user message to history
        self.add_message("user", prompt)
        
        # Prepare messages (Anthropic separates system prompts)
        system_prompt, messages = self._prepare_messages()
        
        # Prepare API call
        api_kwargs = {k: v for k, v in self.config.items() if k not in ['api_key']}
        if 'max_tokens' not in api_kwargs:
            api_kwargs['max_tokens'] = 4096
        
        # Add system prompt if present
        if system_prompt:
            api_kwargs['system'] = system_prompt
        
        response = client.messages.create(
            messages=messages,
            **api_kwargs
        )
        
        # Extract response and update state
        assistant_message = response.content[0].text
        self.add_message("assistant", assistant_message)
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        
        return assistant_message
    
    async def aexecute(self, prompt: str, **metadata) -> str:
        """Execute prompt using Anthropic API asynchronously."""
        client = self._get_async_client()
        
        # Add user message to history
        self.add_message("user", prompt)
        
        # Prepare messages (Anthropic separates system prompts)
        system_prompt, messages = self._prepare_messages()
        
        # Prepare API call
        api_kwargs = {k: v for k, v in self.config.items() if k not in ['api_key']}
        if 'max_tokens' not in api_kwargs:
            api_kwargs['max_tokens'] = 4096
        
        # Add system prompt if present
        if system_prompt:
            api_kwargs['system'] = system_prompt
        
        response = await client.messages.create(
            messages=messages,
            **api_kwargs
        )
        
        # Extract response and update state
        assistant_message = response.content[0].text
        self.add_message("assistant", assistant_message)
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        
        return assistant_message


class GeminiJar(Jar):
    """Jar that uses Google Gemini API for LLM execution."""
    
    def __init__(self, model: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None, **kwargs):
        """Initialize Gemini jar.
        
        Args:
            model: Gemini model to use (default: gemini-2.0-flash-exp)
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            **kwargs: Additional Gemini API parameters (temperature, max_tokens, etc.)
        """
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._client = None
        self._async_client = None
    
    def _get_client(self):
        """Lazy load Gemini sync client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.get('api_key'))
                self._client = genai.GenerativeModel(self.config['model'])
            except ImportError:
                raise ImportError(
                    "Google Generative AI package not installed. Install it with: pip install google-generativeai"
                )
        return self._client
    
    def _get_async_client(self):
        """Lazy load Gemini async client."""
        if self._async_client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.get('api_key'))
                self._async_client = genai.GenerativeModel(self.config['model'])
            except ImportError:
                raise ImportError(
                    "Google Generative AI package not installed. Install it with: pip install google-generativeai"
                )
        return self._async_client
    
    def _prepare_gemini_history(self) -> tuple[Optional[str], List[Dict[str, str]]]:
        """Prepare messages for Gemini API (system instruction separate).
        
        Returns:
            Tuple of (system_instruction, chat_history)
        """
        system_instruction = None
        chat_history = []
        
        for msg in self.history:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "assistant":
                chat_history.append({"role": "model", "parts": [msg["content"]]})
            else:  # user
                chat_history.append({"role": "user", "parts": [msg["content"]]})
        
        return system_instruction, chat_history
    
    def execute(self, prompt: str, **metadata) -> str:
        """Execute prompt using Gemini API synchronously."""
        client = self._get_client()
        
        # Add user message to history
        self.add_message("user", prompt)
        
        # Prepare messages
        system_instruction, chat_history = self._prepare_gemini_history()
        
        # Start chat with history (excluding the last user message we just added)
        chat = client.start_chat(history=chat_history[:-1])
        
        # Send the current message
        response = chat.send_message(prompt)
        
        # Extract response and update state
        assistant_message = response.text
        self.add_message("assistant", assistant_message)
        
        # Gemini usage metadata
        if hasattr(response, 'usage_metadata'):
            self.total_tokens += response.usage_metadata.total_token_count
        
        return assistant_message
    
    async def aexecute(self, prompt: str, **metadata) -> str:
        """Execute prompt using Gemini API asynchronously."""
        client = self._get_async_client()
        
        # Add user message to history
        self.add_message("user", prompt)
        
        # Prepare messages
        system_instruction, chat_history = self._prepare_gemini_history()
        
        # Start chat with history (excluding the last user message we just added)
        chat = client.start_chat(history=chat_history[:-1])
        
        # Send the current message asynchronously
        response = await chat.send_message_async(prompt)
        
        # Extract response and update state
        assistant_message = response.text
        self.add_message("assistant", assistant_message)
        
        # Gemini usage metadata
        if hasattr(response, 'usage_metadata'):
            self.total_tokens += response.usage_metadata.total_token_count
        
        return assistant_message


def get_active_jar() -> Optional[Jar]:
    """Get the currently active jar (sync context).
    
    Returns:
        Active jar instance or None
    """
    return _sync_jar.get()


def get_active_async_jar() -> Optional[Jar]:
    """Get the currently active jar (async context).
    
    Returns:
        Active jar instance or None
    """
    return _async_jar.get()


# Export jar classes for user instantiation
__all__ = ['Jar', 'MockJar', 'OpenAIJar', 'AnthropicJar', 'GeminiJar', 'get_active_jar', 'get_active_async_jar']
