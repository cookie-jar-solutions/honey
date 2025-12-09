"""Integration tests for complete workflows."""

import pytest
import sys
import importlib
from pathlib import Path
from unittest.mock import Mock, AsyncMock


class TestLoaderJarIntegration:
    """Integration tests combining loader and jar functionality."""
    
    def test_prompt_without_jar_returns_string(self, temp_hny_file, isolated_sys_path):
        """Test prompt function returns string when no jar is active."""
        content = "greet\nHello {{name}}!"
        hny_path = temp_hny_file(content, "test_prompts.hny")
        
        test_prompts = importlib.import_module("test_prompts")
        result = test_prompts.greet(name="Alice")
        
        assert result == "Hello Alice!"
        assert isinstance(result, str)
    
    def test_prompt_with_jar_executes_llm(self, temp_hny_file, isolated_sys_path, clean_jar_context):
        """Test prompt function executes with jar when active."""
        from honey.jars import MockJar
        
        content = "summarize\nSummarize: {{text}}"
        hny_path = temp_hny_file(content, "test_prompts.hny")
        
        test_prompts = importlib.import_module("test_prompts")
        jar = MockJar()
        
        with jar:
            result = test_prompts.summarize(text="Long document")
        
        assert "[MOCK RESPONSE]" in result
        assert "Summarize: Long document" in result
        assert jar.message_count == 2
    
    def test_multi_turn_conversation(self, temp_hny_file, isolated_sys_path, clean_jar_context):
        """Test multi-turn conversation with stateful jar."""
        from honey.jars import MockJar
        
        content = "chat\n{{message}}"
        hny_path = temp_hny_file(content, "conversation.hny")
        
        conversation = importlib.import_module("conversation")
        jar = MockJar()
        jar.add_system_prompt("You are a helpful assistant")
        
        with jar:
            response1 = conversation.chat(message="Hello")
            response2 = conversation.chat(message="How are you?")
        
        history = jar.get_history()
        assert len(history) == 5  # system + 2*(user + assistant)
        assert history[0]["role"] == "system"
        assert history[1]["content"] == "Hello"
        assert history[3]["content"] == "How are you?"
    
    def test_different_jars_isolated(self, temp_hny_file, isolated_sys_path, clean_jar_context):
        """Test different jar instances maintain separate state."""
        from honey.jars import MockJar
        
        content = "process\n{{input}}"
        hny_path = temp_hny_file(content, "processor.hny")
        
        processor = importlib.import_module("processor")
        jar1 = MockJar()
        jar2 = MockJar()
        
        with jar1:
            processor.process(input="Request 1")
        
        with jar2:
            processor.process(input="Request 2")
        
        assert jar1.message_count == 2
        assert jar2.message_count == 2
        assert jar1.history[0]["content"] == "Request 1"
        assert jar2.history[0]["content"] == "Request 2"


class TestAsyncIntegration:
    """Integration tests for async workflows."""
    
    @pytest.mark.asyncio
    async def test_async_prompt_execution(self, temp_hny_file, isolated_sys_path, clean_jar_context):
        """Test async prompt execution with jar."""
        from honey.jars import MockJar
        
        content = "analyze\nAnalyze: {{data}}"
        hny_path = temp_hny_file(content, "async_prompts.hny")
        
        async_prompts = importlib.import_module("async_prompts")
        jar = MockJar()
        
        async with jar:
            result = await async_prompts.analyze(data="Sample data")
        
        assert "[ASYNC MOCK RESPONSE]" in result
        assert "Analyze: Sample data" in result
        assert jar.message_count == 2
    
    @pytest.mark.asyncio
    async def test_async_multi_turn(self, temp_hny_file, isolated_sys_path, clean_jar_context):
        """Test async multi-turn conversation."""
        from honey.jars import MockJar
        
        content = "discuss\n{{topic}}"
        hny_path = temp_hny_file(content, "async_chat.hny")
        
        async_chat = importlib.import_module("async_chat")
        jar = MockJar()
        
        async with jar:
            await async_chat.discuss(topic="Python")
            await async_chat.discuss(topic="JavaScript")
        
        assert jar.message_count == 4
        assert jar.history[0]["content"] == "Python"
        assert jar.history[2]["content"] == "JavaScript"
    
    @pytest.mark.asyncio
    async def test_concurrent_async_requests(self, temp_hny_file, isolated_sys_path, clean_jar_context):
        """Test concurrent async requests with different jars."""
        import asyncio
        from honey.jars import MockJar
        
        content = "task\n{{work}}"
        hny_path = temp_hny_file(content, "tasks.hny")
        
        tasks = importlib.import_module("tasks")
        
        async def use_jar(jar, work_item):
            async with jar:
                return await tasks.task(work=work_item)
        
        jar1 = MockJar()
        jar2 = MockJar()
        jar3 = MockJar()
        
        results = await asyncio.gather(
            use_jar(jar1, "Task 1"),
            use_jar(jar2, "Task 2"),
            use_jar(jar3, "Task 3")
        )
        
        assert len(results) == 3
        assert all("[ASYNC MOCK RESPONSE]" in r for r in results)
        assert jar1.history[0]["content"] == "Task 1"
        assert jar2.history[0]["content"] == "Task 2"
        assert jar3.history[0]["content"] == "Task 3"


class TestRealWorldScenarios:
    """Tests simulating real-world usage patterns."""
    
    def test_template_with_conditionals(self, temp_hny_file, isolated_sys_path, clean_jar_context):
        """Test template with Jinja2 conditionals."""
        from honey.jars import MockJar
        
        content = """format
Format this text:
{% if style %}Style: {{style}}{% endif %}
Content: {{content}}"""
        
        hny_path = temp_hny_file(content, "formatter.hny")
        
        formatter = importlib.import_module("formatter")
        jar = MockJar()
        
        with jar:
            result1 = formatter.format(content="Hello", style="bold")
            result2 = formatter.format(content="World")
        
        assert jar.message_count == 4
        assert "Style: bold" in jar.history[0]["content"]
        assert "Style:" not in jar.history[2]["content"]
    
    def test_multiple_prompts_same_jar(self, temp_hny_file, isolated_sys_path, clean_jar_context):
        """Test using multiple different prompts with same jar."""
        from honey.jars import MockJar
        
        content1 = "summarize\nSummarize: {{text}}"
        content2 = "translate\nTranslate to {{lang}}: {{text}}"
        
        hny_path1 = temp_hny_file(content1, "tools1.hny")
        hny_path2 = temp_hny_file(content2, "tools2.hny")
        
        tools1 = importlib.import_module("tools1")
        tools2 = importlib.import_module("tools2")
        
        jar = MockJar()
        jar.add_system_prompt("You are helpful")
        
        with jar:
            tools1.summarize(text="Long doc")
            tools2.translate(lang="Spanish", text="Hello")
        
        history = jar.get_history()
        assert len(history) == 5  # system + 2*(user + assistant)
        assert "Summarize" in history[1]["content"]
        assert "Translate to Spanish" in history[3]["content"]
    
    def test_jar_with_custom_config(self, temp_hny_file, isolated_sys_path, clean_jar_context):
        """Test jar with custom configuration."""
        from honey.jars import OpenAIJar
        
        content = "generate\nGenerate: {{prompt}}"
        
        hny_path = temp_hny_file(content, "generator.hny")
        
        generator = importlib.import_module("generator")
        
        jar = OpenAIJar(
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            api_key="test-key"
        )
        
        # Mock the client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated"))]
        mock_response.usage = Mock(total_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response
        jar._client = mock_client
        
        with jar:
            result = generator.generate(prompt="Test")
        
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100


class TestErrorHandling:
    """Integration tests for error scenarios."""
    
    def test_missing_template_variable(self, temp_hny_file, isolated_sys_path):
        """Test handling of missing template variable."""
        content = "greet\nHello {{name}}!"
        hny_path = temp_hny_file(content, "missing_var.hny")
        
        missing_var = importlib.import_module("missing_var")
        
        # Jinja2 default behavior: renders undefined variables as empty string
        result = missing_var.greet()  # Missing 'name' parameter
        assert result == "Hello !"  # {{name}} becomes empty string
    
    def test_jar_state_after_error(self, temp_hny_file, isolated_sys_path, clean_jar_context):
        """Test jar state is maintained even after execution error."""
        from honey.jars import MockJar
        
        content = "process\n{{data}}"
        hny_path = temp_hny_file(content, "error_test.hny")
        
        error_test = importlib.import_module("error_test")
        jar = MockJar()
        
        with jar:
            jar.execute("First message")
            
            # Simulate error in execution
            try:
                error_test.process()  # Missing 'data' parameter
            except Exception:
                pass
            
            # Jar should still work after error
            result = jar.execute("Second message")
        
        assert "[MOCK RESPONSE]" in result
        # First message should still be in history
        assert jar.history[0]["content"] == "First message"
