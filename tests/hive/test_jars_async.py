"""Asynchronous tests for jar classes."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from hive.jars import MockJar, OpenAIJar, AnthropicJar, GeminiJar, get_active_async_jar


class TestAsyncJarContextManagers:
    """Tests for async jar context manager behavior."""
    
    @pytest.mark.asyncio
    async def test_async_context_manager_sets_active_jar(self, clean_jar_context):
        """Test async context manager sets jar as active."""
        jar = MockJar()
        
        assert get_active_async_jar() is None
        
        async with jar:
            assert get_active_async_jar() is jar
        
        assert get_active_async_jar() is None
    
    @pytest.mark.asyncio
    async def test_async_context_manager_returns_jar(self):
        """Test async context manager returns jar instance."""
        jar = MockJar()
        
        async with jar as j:
            assert j is jar
    
    @pytest.mark.asyncio
    async def test_nested_async_context_managers(self, clean_jar_context):
        """Test nested async context managers - inner overrides outer."""
        outer = MockJar()
        inner = MockJar()
        
        async with outer:
            assert get_active_async_jar() is outer
            
            async with inner:
                assert get_active_async_jar() is inner
            
            assert get_active_async_jar() is outer
        
        assert get_active_async_jar() is None
    
    @pytest.mark.asyncio
    async def test_async_context_cleanup_on_exception(self, clean_jar_context):
        """Test async context cleans up even on exception."""
        jar = MockJar()
        
        try:
            async with jar:
                assert get_active_async_jar() is jar
                raise ValueError("Test error")
        except ValueError:
            pass
        
        assert get_active_async_jar() is None


class TestAsyncExecution:
    """Tests for async execution methods."""
    
    @pytest.mark.asyncio
    async def test_mock_jar_aexecute(self):
        """Test MockJar async execution."""
        jar = MockJar()
        
        result = await jar.aexecute("Test prompt")
        
        assert "[ASYNC MOCK RESPONSE]" in result
        assert "Test prompt" in result
        assert jar.message_count == 2
    
    @pytest.mark.asyncio
    async def test_openai_jar_aexecute(self):
        """Test OpenAI jar async execution with mocked client."""
        jar = OpenAIJar(model="gpt-4", api_key="test-key")
        
        # Mock async client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Async AI response"))]
        mock_response.usage = Mock(total_tokens=60)
        mock_client.chat.completions.create.return_value = mock_response
        
        jar._async_client = mock_client
        
        result = await jar.aexecute("Async test prompt")
        
        assert result == "Async AI response"
        assert jar.total_tokens == 60
        assert jar.message_count == 2
        mock_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_anthropic_jar_aexecute(self):
        """Test Anthropic jar async execution with mocked client."""
        jar = AnthropicJar(model="claude-3-5-sonnet-20241022", api_key="test-key")
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Async Claude response")]
        mock_response.usage = Mock(input_tokens=15, output_tokens=25)
        mock_client.messages.create.return_value = mock_response
        
        jar._async_client = mock_client
        
        result = await jar.aexecute("Async test")
        
        assert result == "Async Claude response"
        assert jar.total_tokens == 40
        assert jar.message_count == 2
    
    @pytest.mark.asyncio
    async def test_gemini_jar_aexecute(self):
        """Test Gemini jar async execution with mocked client."""
        jar = GeminiJar(model="gemini-2.0-flash-exp", api_key="test-key")
        
        mock_client = Mock()
        mock_chat = AsyncMock()
        mock_response = Mock()
        mock_response.text = "Async Gemini response"
        mock_response.usage_metadata = Mock(total_token_count=30)
        mock_chat.send_message_async.return_value = mock_response
        mock_client.start_chat.return_value = mock_chat
        
        # Set both sync and async clients to avoid lazy loading
        jar._client = mock_client
        jar._async_client = mock_client
        
        result = await jar.aexecute("Async Gemini test")
        
        assert result == "Async Gemini response"
        assert jar.total_tokens == 30
        assert jar.message_count == 2


class TestConcurrentAsyncExecution:
    """Tests for concurrent async jar usage."""
    
    @pytest.mark.asyncio
    async def test_concurrent_jar_execution(self):
        """Test multiple jars can execute concurrently."""
        jar1 = MockJar()
        jar2 = MockJar()
        
        async def use_jar1():
            async with jar1:
                return await jar1.aexecute("Request 1")
        
        async def use_jar2():
            async with jar2:
                return await jar2.aexecute("Request 2")
        
        results = await asyncio.gather(use_jar1(), use_jar2())
        
        assert len(results) == 2
        assert "Request 1" in results[0]
        assert "Request 2" in results[1]
        assert jar1.message_count == 2
        assert jar2.message_count == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_same_jar(self):
        """Test same jar handles concurrent requests."""
        jar = MockJar()
        
        async def make_request(prompt):
            return await jar.aexecute(prompt)
        
        results = await asyncio.gather(
            make_request("Concurrent 1"),
            make_request("Concurrent 2"),
            make_request("Concurrent 3")
        )
        
        assert len(results) == 3
        # All requests should complete
        assert all("[ASYNC MOCK RESPONSE]" in r for r in results)
        # History should have all messages
        assert jar.message_count == 6  # 3 user + 3 assistant


class TestAsyncJarStateManagement:
    """Tests for jar state management in async contexts."""
    
    @pytest.mark.asyncio
    async def test_async_jar_maintains_history(self):
        """Test async jar maintains conversation history."""
        jar = MockJar()
        
        async with jar:
            await jar.aexecute("First")
            await jar.aexecute("Second")
        
        history = jar.get_history()
        assert len(history) == 4
        assert history[0]["content"] == "First"
        assert history[2]["content"] == "Second"
    
    @pytest.mark.asyncio
    async def test_async_jar_reusable(self):
        """Test async jar is reusable across multiple async contexts."""
        jar = MockJar()
        jar.add_system_prompt("System")
        
        async with jar:
            await jar.aexecute("Message 1")
        
        async with jar:
            await jar.aexecute("Message 2")
        
        history = jar.get_history()
        assert len(history) == 5  # system + 2*(user + assistant)
        assert history[0]["role"] == "system"


class TestAsyncContextIsolation:
    """Tests for async context isolation."""
    
    @pytest.mark.asyncio
    async def test_sync_and_async_contexts_isolated(self, clean_jar_context):
        """Test sync and async jars are isolated from each other."""
        from hive.jars import get_active_jar
        
        sync_jar = MockJar()
        async_jar = MockJar()
        
        # Set sync context
        with sync_jar:
            assert get_active_jar() is sync_jar
            assert get_active_async_jar() is None
            
            # Set async context within sync
            async with async_jar:
                assert get_active_jar() is sync_jar
                assert get_active_async_jar() is async_jar
            
            assert get_active_jar() is sync_jar
            assert get_active_async_jar() is None
        
        assert get_active_jar() is None
        assert get_active_async_jar() is None
