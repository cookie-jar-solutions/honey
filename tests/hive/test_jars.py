"""Synchronous tests for jar classes."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from hive.jars import Jar, MockJar, OpenAIJar, AnthropicJar, GeminiJar, get_active_jar


class TestJarBase:
    """Tests for base Jar class."""
    
    def test_jar_initialization(self):
        """Test jar initializes with config."""
        jar = MockJar(model="test-model", temperature=0.5)
        
        assert jar.config["model"] == "test-model"
        assert jar.config["temperature"] == 0.5
        assert jar.history == []
        assert jar.total_tokens == 0
        assert jar.message_count == 0
    
    def test_add_message(self):
        """Test adding messages to history."""
        jar = MockJar()
        
        jar.add_message("user", "Hello")
        jar.add_message("assistant", "Hi there")
        
        assert len(jar.history) == 2
        assert jar.history[0] == {"role": "user", "content": "Hello"}
        assert jar.history[1] == {"role": "assistant", "content": "Hi there"}
        assert jar.message_count == 2
    
    def test_get_history_returns_copy(self):
        """Test get_history returns a copy, not reference."""
        jar = MockJar()
        jar.add_message("user", "Test")
        
        history = jar.get_history()
        history.append({"role": "user", "content": "Modified"})
        
        assert len(jar.history) == 1
        assert len(history) == 2
    
    def test_clear_history(self):
        """Test clearing history resets all counters."""
        jar = MockJar()
        jar.add_message("user", "Test")
        jar.total_tokens = 100
        
        jar.clear_history()
        
        assert jar.history == []
        assert jar.message_count == 0
        assert jar.total_tokens == 0
    
    def test_add_system_prompt(self):
        """Test adding system prompt."""
        jar = MockJar()
        
        jar.add_system_prompt("You are helpful")
        
        assert len(jar.history) == 1
        assert jar.history[0]["role"] == "system"
        assert jar.history[0]["content"] == "You are helpful"


class TestMockJar:
    """Tests for MockJar."""
    
    def test_execute_returns_mock_response(self):
        """Test execute returns formatted mock response."""
        jar = MockJar()
        
        response = jar.execute("Test prompt")
        
        assert "[MOCK RESPONSE]" in response
        assert "Test prompt" in response
        assert jar.message_count == 2
    
    def test_execute_updates_history(self):
        """Test execute updates conversation history."""
        jar = MockJar()
        
        jar.execute("Hello")
        
        assert len(jar.history) == 2
        assert jar.history[0]["role"] == "user"
        assert jar.history[0]["content"] == "Hello"
        assert jar.history[1]["role"] == "assistant"


class TestJarContextManagers:
    """Tests for jar context manager behavior."""
    
    def test_sync_context_manager_sets_active_jar(self, clean_jar_context):
        """Test sync context manager sets jar as active."""
        jar = MockJar()
        
        assert get_active_jar() is None
        
        with jar:
            assert get_active_jar() is jar
        
        assert get_active_jar() is None
    
    def test_sync_context_manager_returns_jar(self):
        """Test sync context manager returns jar instance."""
        jar = MockJar()
        
        with jar as j:
            assert j is jar
    
    def test_nested_sync_context_managers(self, clean_jar_context):
        """Test nested context managers - inner overrides outer."""
        outer = MockJar()
        inner = MockJar()
        
        with outer:
            assert get_active_jar() is outer
            
            with inner:
                assert get_active_jar() is inner
            
            assert get_active_jar() is outer
        
        assert get_active_jar() is None
    
    def test_sync_context_cleanup_on_exception(self, clean_jar_context):
        """Test sync context cleans up even on exception."""
        jar = MockJar()
        
        try:
            with jar:
                assert get_active_jar() is jar
                raise ValueError("Test error")
        except ValueError:
            pass
        
        assert get_active_jar() is None


class TestOpenAIJar:
    """Tests for OpenAIJar with mocked client."""
    
    def test_initialization(self):
        """Test OpenAI jar initialization."""
        jar = OpenAIJar(model="gpt-4", temperature=0.7, api_key="test-key")
        
        assert jar.config["model"] == "gpt-4"
        assert jar.config["temperature"] == 0.7
        assert jar.config["api_key"] == "test-key"
    
    def test_execute_with_mocked_client(self):
        """Test execute with mocked OpenAI client."""
        jar = OpenAIJar(model="gpt-4", api_key="test-key")
        
        # Mock the client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="AI response"))]
        mock_response.usage = Mock(total_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response
        
        jar._client = mock_client
        
        result = jar.execute("Test prompt")
        
        assert result == "AI response"
        assert jar.total_tokens == 50
        assert jar.message_count == 2
        mock_client.chat.completions.create.assert_called_once()
    
    def test_execute_passes_config_to_api(self):
        """Test execute passes configuration to API."""
        jar = OpenAIJar(model="gpt-4", temperature=0.7, api_key="test-key")
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_response.usage = Mock(total_tokens=10)
        mock_client.chat.completions.create.return_value = mock_response
        
        jar._client = mock_client
        jar.execute("Test")
        
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.7
        assert "api_key" not in call_kwargs  # Should be filtered out


class TestAnthropicJar:
    """Tests for AnthropicJar with mocked client."""
    
    def test_initialization(self):
        """Test Anthropic jar initialization."""
        jar = AnthropicJar(model="claude-3-5-sonnet-20241022", api_key="test-key")
        
        assert jar.config["model"] == "claude-3-5-sonnet-20241022"
        assert jar.config["api_key"] == "test-key"
    
    def test_prepare_messages_separates_system(self):
        """Test that system prompts are separated."""
        jar = AnthropicJar()
        jar.add_system_prompt("You are helpful")
        jar.add_message("user", "Hello")
        jar.add_message("assistant", "Hi")
        
        system_prompt, messages = jar._prepare_messages()
        
        assert system_prompt == "You are helpful"
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    def test_execute_with_mocked_client(self):
        """Test execute with mocked Anthropic client."""
        jar = AnthropicJar(model="claude-3-5-sonnet-20241022", api_key="test-key")
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Claude response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_client.messages.create.return_value = mock_response
        
        jar._client = mock_client
        
        result = jar.execute("Test prompt")
        
        assert result == "Claude response"
        assert jar.total_tokens == 30
        assert jar.message_count == 2
    
    def test_execute_includes_system_prompt(self):
        """Test execute includes system prompt in API call."""
        jar = AnthropicJar(api_key="test-key")
        jar.add_system_prompt("Be helpful")
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.usage = Mock(input_tokens=5, output_tokens=5)
        mock_client.messages.create.return_value = mock_response
        
        jar._client = mock_client
        jar.execute("Test")
        
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Be helpful"


class TestGeminiJar:
    """Tests for GeminiJar with mocked client."""
    
    def test_initialization(self):
        """Test Gemini jar initialization."""
        jar = GeminiJar(model="gemini-2.0-flash-exp", api_key="test-key")
        
        assert jar.config["model"] == "gemini-2.0-flash-exp"
        assert jar.config["api_key"] == "test-key"
    
    def test_prepare_gemini_history_converts_roles(self):
        """Test that assistant role is converted to model."""
        jar = GeminiJar()
        jar.add_message("user", "Hello")
        jar.add_message("assistant", "Hi")
        
        system_instruction, chat_history = jar._prepare_gemini_history()
        
        assert len(chat_history) == 2
        assert chat_history[0]["role"] == "user"
        assert chat_history[1]["role"] == "model"
        assert chat_history[1]["parts"] == ["Hi"]
    
    def test_execute_with_mocked_client(self):
        """Test execute with mocked Gemini client."""
        jar = GeminiJar(model="gemini-2.0-flash-exp", api_key="test-key")
        
        mock_client = Mock()
        mock_chat = Mock()
        mock_response = Mock()
        mock_response.text = "Gemini response"
        mock_response.usage_metadata = Mock(total_token_count=25)
        mock_chat.send_message.return_value = mock_response
        mock_client.start_chat.return_value = mock_chat
        
        jar._client = mock_client
        
        result = jar.execute("Test prompt")
        
        assert result == "Gemini response"
        assert jar.total_tokens == 25
        assert jar.message_count == 2


class TestJarReusability:
    """Tests for jar reusability across contexts."""
    
    def test_jar_reusable_across_contexts(self):
        """Test jar maintains state across multiple context uses."""
        jar = MockJar()
        
        with jar:
            jar.execute("First message")
        
        assert jar.message_count == 2
        
        with jar:
            jar.execute("Second message")
        
        assert jar.message_count == 4
        assert len(jar.history) == 4
    
    def test_jar_state_persists_between_contexts(self):
        """Test jar state persists between context uses."""
        jar = MockJar()
        jar.add_system_prompt("System instruction")
        
        with jar:
            jar.execute("Message 1")
        
        with jar:
            jar.execute("Message 2")
        
        history = jar.get_history()
        assert history[0]["role"] == "system"
        assert len(history) == 5  # system + 2*(user + assistant)
