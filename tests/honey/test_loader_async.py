"""Asynchronous tests for the .hny file loader runtime integration."""

import pytest
from honey import loader, mock_jar


class TestAsyncRuntimeIntegration:
    """Tests for prompt functions in async jar contexts."""
    
    @pytest.mark.asyncio
    async def test_function_returns_coroutine_in_async_context(self, clean_jar_context):
        """Test that function returns coroutine in async jar context."""
        template = "Hello, {{name}}!"
        func = loader.create_prompt_function(template)
        jar = mock_jar()
        
        async with jar:
            result = func(name="Test")
            # Should return a coroutine
            assert hasattr(result, '__await__')
    
    @pytest.mark.asyncio
    async def test_function_executes_with_async_jar(self, clean_jar_context):
        """Test that function executes prompt with async jar."""
        template = "Hello, {{name}}!"
        func = loader.create_prompt_function(template)
        jar = mock_jar()
        
        async with jar:
            result = await func(name="AsyncTest")
        
        assert "[ASYNC MOCK RESPONSE]" in result
        assert jar.message_count == 2  # user + assistant
    
    @pytest.mark.asyncio
    async def test_function_updates_jar_history(self, clean_jar_context):
        """Test that function updates jar history in async context."""
        template = "Process: {{data}}"
        func = loader.create_prompt_function(template)
        jar = mock_jar()
        
        async with jar:
            await func(data="test data")
        
        history = jar.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert "test data" in history[0]["content"]
        assert history[1]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_multiple_async_calls_maintain_state(self, clean_jar_context):
        """Test that multiple async calls maintain jar state."""
        template = "Message: {{text}}"
        func = loader.create_prompt_function(template)
        jar = mock_jar()
        
        async with jar:
            await func(text="First")
            await func(text="Second")
            await func(text="Third")
        
        assert jar.message_count == 6  # 3 user + 3 assistant
        history = jar.get_history()
        assert "First" in history[0]["content"]
        assert "Second" in history[2]["content"]
        assert "Third" in history[4]["content"]
