"""Gemini jar implementation."""

from typing import Optional, List, Dict

from .base import Jar


class GeminiJar(Jar):
    """Jar that uses Google Gemini API for LLM execution."""
    
    def __init__(self, model: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None, system_prompt: Optional[str] = None, **kwargs):
        """Initialize Gemini jar.
        
        Args:
            model: Gemini model to use (default: gemini-2.0-flash-exp)
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            system_prompt: Optional system prompt to set conversation context
            **kwargs: Additional Gemini API parameters (temperature, max_tokens, etc.)
        """
        super().__init__(system_prompt=system_prompt, model=model, api_key=api_key, **kwargs)
        self._client = None
        self._async_client = None
    
    def _get_client(self):
        """Lazy load Gemini sync client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(
                    api_key=self.config.get("api_key")
                )
            except ImportError:
                raise ImportError(
                    "Google GenAI package not installed. Install it with: uv add google-genai"
                )
        return self._client

    def _get_async_client(self):
        """Lazy load Gemini async client."""
        if self._async_client is None:
            try:
                from google import genai
                self._async_client = genai.AsyncClient(
                    api_key=self.config.get("api_key")
                )
            except ImportError:
                raise ImportError(
                    "Google GenAI package not installed. Install it with: uv add google-genai"
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
        # Include system_instruction if present
        chat_kwargs = {"history": chat_history[:-1]}
        if system_instruction:
            chat_kwargs["system_instruction"] = system_instruction
        chat = client.start_chat(**chat_kwargs)
        
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
        # Include system_instruction if present
        chat_kwargs = {"history": chat_history[:-1]}
        if system_instruction:
            chat_kwargs["system_instruction"] = system_instruction
        chat = client.start_chat(**chat_kwargs)
        
        # Send the current message asynchronously
        response = await chat.send_message_async(prompt)
        
        # Extract response and update state
        assistant_message = response.text
        self.add_message("assistant", assistant_message)
        
        # Gemini usage metadata
        if hasattr(response, 'usage_metadata'):
            self.total_tokens += response.usage_metadata.total_token_count
        
        return assistant_message
