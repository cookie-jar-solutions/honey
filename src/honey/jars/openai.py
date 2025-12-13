"""OpenAI jar implementations."""

from typing import Optional
from unittest.mock import Mock

from .base import Jar


class OpenAIBaseJar(Jar):
    """Base class for OpenAI-style APIs (OpenAI + compatible)."""

    # ---------- capability detection ----------

    def _supports_responses(self, client) -> bool:
        if isinstance(client, Mock):
            responses = client.__dict__.get("responses")
        else:
            responses = getattr(client, "responses", None)
        return responses is not None and hasattr(responses, "create")

    # ---------- shared kwargs ----------

    def _call_kwargs(self, exclude=()):
        return {
            k: v for k, v in self.config.items()
            if k not in ("api_key", "base_url", "model", *exclude)
        }

    # ---------- sync ----------

    def execute(self, prompt: str, **metadata) -> str:
        client = self._get_client()
        self.add_message("user", prompt)

        if self._supports_responses(client):
            response = client.responses.create(
                model=self.config["model"],
                input=self.history,
                **self._call_kwargs()
            )
            assistant_message = response.output_text
            usage = response.usage
        else:
            response = client.chat.completions.create(
                model=self.config["model"],
                messages=self.history,
                **self._call_kwargs()
            )
            assistant_message = response.choices[0].message.content
            usage = getattr(response, "usage", None)

        self.add_message("assistant", assistant_message)

        if usage:
            self.total_tokens += usage.total_tokens

        return assistant_message

    # ---------- async ----------

    async def aexecute(self, prompt: str, **metadata) -> str:
        client = self._get_async_client()
        self.add_message("user", prompt)

        if self._supports_responses(client):
            response = await client.responses.create(
                model=self.config["model"],
                input=self.history,
                **self._call_kwargs()
            )
            assistant_message = response.output_text
            usage = response.usage
        else:
            response = await client.chat.completions.create(
                model=self.config["model"],
                messages=self.history,
                **self._call_kwargs()
            )
            assistant_message = response.choices[0].message.content
            usage = getattr(response, "usage", None)

        self.add_message("assistant", assistant_message)

        if usage:
            self.total_tokens += usage.total_tokens

        return assistant_message


class OpenAICompatibleJar(OpenAIBaseJar):
    """Jar implementation for OpenAI-compatible API endpoints.
    
    Works with any service that implements the OpenAI API specification,
    such as Ollama, LM Studio, vLLM, LocalAI, etc.
    """
    
    def __init__(self, model: str, base_url: str, api_key: Optional[str] = "not-needed", **kwargs):
        """Initialize OpenAI-compatible jar.
        
        Args:
            model: Model name (specific to your API provider)
            base_url: Base URL for the API endpoint (e.g., "http://localhost:11434/v1")
            api_key: API key (optional, defaults to "not-needed" for local endpoints)
            **kwargs: Additional API parameters (temperature, max_tokens, etc.)
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
        self._client = None
        self._async_client = None

    def _get_client(self):
        """Lazy load OpenAI-compatible sync client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config["api_key"],
                    base_url=self.config["base_url"],
                )
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install it with: pip install openai"
                )
        return self._client

    def _get_async_client(self):
        """Lazy load OpenAI-compatible async client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(
                    api_key=self.config["api_key"],
                    base_url=self.config["base_url"],
                )
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install it with: pip install openai"
                )
        return self._async_client


class OpenAIJar(OpenAIBaseJar):
    """Jar that uses OpenAI API for LLM execution."""
    
    def __init__(self, model: str = "gpt-4.1-mini", api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI jar.
        
        Args:
            model: OpenAI model to use (default: gpt-4.1-mini)
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
                self._client = OpenAI(api_key=self.config.get("api_key"))
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
                self._async_client = AsyncOpenAI(api_key=self.config.get("api_key"))
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install it with: pip install openai"
                )
        return self._async_client


class OpenAIClientJar(OpenAIJar):
    """OpenAI jar that accepts pre-initialized clients."""
    
    def __init__(self, model: str, sync_client, async_client, **kwargs):
        """Initialize OpenAI jar with pre-initialized clients.
        
        Args:
            model: OpenAI model to use
            sync_client: Pre-initialized OpenAI sync client
            async_client: Pre-initialized OpenAI async client
            **kwargs: Additional OpenAI API parameters
        """
        super().__init__(model=model, **kwargs)
        self._client = sync_client
        self._async_client = async_client

    def _get_client(self):
        """Return the pre-initialized sync client."""
        return self._client

    def _get_async_client(self):
        """Return the pre-initialized async client."""
        return self._async_client
