"""Mock jar implementation for testing."""

from .base import Jar


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
