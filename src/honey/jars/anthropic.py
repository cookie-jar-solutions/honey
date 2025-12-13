"""Anthropic jar implementation."""

from typing import Optional, List, Dict

from .base import Jar


class AnthropicJar(Jar):
    """Jar that uses Anthropic API for LLM execution."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, system_prompt: Optional[str] = None, **kwargs):
        """Initialize Anthropic jar.
        
        Args:
            model: Anthropic model to use (default: claude-3-5-sonnet-20241022)
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            system_prompt: Optional system prompt to set conversation context
            **kwargs: Additional Anthropic API parameters (temperature, max_tokens, etc.)
        """
        super().__init__(system_prompt=system_prompt, model=model, api_key=api_key, **kwargs)
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
