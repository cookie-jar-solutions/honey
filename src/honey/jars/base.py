"""Base class and utilities for LLM runtime jars.

Jars are reusable, stateful context managers that execute prompts against LLM APIs.
They support both synchronous and asynchronous execution modes.
"""

import contextvars
from typing import Optional, List, Dict
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
    
    def __init__(self, system_prompt: Optional[str] = None, **config):
        """Initialize jar with configuration.
        
        Args:
            system_prompt: Optional system prompt to set conversation context
            **config: Runtime configuration (model, temperature, api_key, etc.)
        """
        self.config = config
        self.history: List[Dict[str, str]] = []
        self.total_tokens = 0
        self.message_count = 0
        self._sync_token = None
        self._async_token = None
        
        # Add system prompt if provided
        if system_prompt:
            self.add_message("system", system_prompt)
    
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

    def add_system_prompt(self, content: str):
        """Add or update the system prompt at the start of history."""
        for msg in self.history:
            if msg["role"] == "system":
                msg["content"] = content
                return
        # Ensure system prompt appears first in the conversation
        self.history.insert(0, {"role": "system", "content": content})
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
