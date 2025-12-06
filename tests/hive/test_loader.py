"""Synchronous tests for the .hny file loader."""

import sys
import pytest
from pathlib import Path
from hive import loader


class TestParseHnyFile:
    """Tests for parse_hny_file function."""
    
    def test_parse_single_prompt(self, temp_hny_file):
        """Test parsing a single prompt from .hny file."""
        content = "greet\nHello, {{name}}!"
        hny_path = temp_hny_file(content, "test.hny")
        
        prompts = loader.parse_hny_file(hny_path)
        
        assert len(prompts) == 1
        assert "greet" in prompts
        assert prompts["greet"] == "Hello, {{name}}!"
    
    def test_parse_multiple_prompts(self, prompts_dir):
        """Test parsing multiple prompts from .hny file."""
        multi_hny = prompts_dir / "multi.hny"
        
        prompts = loader.parse_hny_file(multi_hny)
        
        assert len(prompts) == 3
        assert "summarize" in prompts
        assert "translate" in prompts
        assert "analyze" in prompts
        assert "{{text}}" in prompts["summarize"]
        assert "{{language}}" in prompts["translate"]
    
    def test_parse_empty_file(self, temp_hny_file):
        """Test parsing an empty .hny file."""
        hny_path = temp_hny_file("", "empty.hny")
        
        prompts = loader.parse_hny_file(hny_path)
        
        assert len(prompts) == 0
    
    def test_parse_with_extra_dashes(self, temp_hny_file):
        """Test parsing with varying separator lengths."""
        content = "prompt1\nContent 1\n---\nprompt2\nContent 2\n--------"
        hny_path = temp_hny_file(content, "dashes.hny")
        
        prompts = loader.parse_hny_file(hny_path)
        
        assert len(prompts) == 2
        assert "prompt1" in prompts
        assert "prompt2" in prompts
    
    def test_parse_complex_jinja(self, prompts_dir):
        """Test parsing prompts with Jinja2 logic."""
        complex_hny = prompts_dir / "complex.hny"
        
        prompts = loader.parse_hny_file(complex_hny)
        
        assert "chat" in prompts
        assert "{% if context %}" in prompts["chat"]
        assert "{{ message }}" in prompts["chat"]


class TestPromptFunctionCreation:
    """Tests for create_prompt_function."""
    
    def test_function_basic_rendering(self):
        """Test that created function renders templates correctly."""
        template = "Hello, {{name}}!"
        func = loader.create_prompt_function(template)
        
        result = func(name="World")
        
        assert result == "Hello, World!"
    
    def test_function_with_multiple_vars(self):
        """Test rendering with multiple variables."""
        template = "{{greeting}} {{name}}, welcome to {{place}}!"
        func = loader.create_prompt_function(template)
        
        result = func(greeting="Hi", name="Alice", place="Wonderland")
        
        assert result == "Hi Alice, welcome to Wonderland!"
    
    def test_function_has_metadata(self):
        """Test that function has template metadata."""
        template = "Test template"
        func = loader.create_prompt_function(template)
        
        assert hasattr(func, "__template__")
        assert func.__template__ == template
        assert hasattr(func, "__doc__")
    
    def test_function_without_jar_context(self, clean_jar_context):
        """Test function returns string when no jar context active."""
        template = "Hello, {{name}}!"
        func = loader.create_prompt_function(template)
        
        result = func(name="Test")
        
        assert isinstance(result, str)
        assert result == "Hello, Test!"
    
    def test_function_with_jinja_conditionals(self):
        """Test Jinja2 conditional rendering."""
        template = "{% if premium %}Premium{% else %}Basic{% endif %} user"
        func = loader.create_prompt_function(template)
        
        result1 = func(premium=True)
        result2 = func(premium=False)
        
        assert result1 == "Premium user"
        assert result2 == "Basic user"


class TestHnyFinder:
    """Tests for HnyFinder class."""
    
    def test_find_spec_finds_hny_file(self, prompts_dir, isolated_sys_path):
        """Test that finder locates .hny files on path."""
        sys.path.insert(0, str(prompts_dir))
        finder = loader.HnyFinder()
        
        spec = finder.find_spec("simple")
        
        assert spec is not None
        assert spec.name == "simple"
        assert spec.origin.endswith("simple.hny")
    
    def test_find_spec_returns_none_for_missing(self, isolated_sys_path):
        """Test that finder returns None for missing files."""
        finder = loader.HnyFinder()
        
        spec = finder.find_spec("nonexistent")
        
        assert spec is None
    
    def test_find_spec_returns_none_for_non_hny(self, tmp_path, isolated_sys_path):
        """Test that finder ignores non-.hny files."""
        # Create a .py file
        py_file = tmp_path / "test.py"
        py_file.write_text("# python file")
        sys.path.insert(0, str(tmp_path))
        
        finder = loader.HnyFinder()
        spec = finder.find_spec("test")
        
        assert spec is None


class TestHnyLoader:
    """Tests for HnyLoader class."""
    
    def test_exec_module_creates_functions(self, prompts_dir, isolated_sys_path):
        """Test that loader creates prompt functions in module."""
        sys.path.insert(0, str(prompts_dir))
        
        # Import the module (this triggers the loader)
        import simple
        
        assert hasattr(simple, "greet")
        assert callable(simple.greet)
        assert simple.greet(name="Test") == "Hello, Test!"
    
    def test_exec_module_sets_metadata(self, prompts_dir, isolated_sys_path):
        """Test that loader sets module metadata."""
        sys.path.insert(0, str(prompts_dir))
        
        import multi
        
        assert hasattr(multi, "__prompts__")
        assert "summarize" in multi.__prompts__
        assert "translate" in multi.__prompts__
        assert "analyze" in multi.__prompts__
        assert hasattr(multi, "__file__")


class TestLoaderIntegration:
    """Integration tests for loader install/uninstall."""
    
    def test_loader_auto_installs(self, isolated_meta_path):
        """Test that loader is auto-installed on import."""
        # Reimport to trigger installation
        import importlib
        importlib.reload(loader)
        
        assert loader._finder in sys.meta_path
    
    def test_install_function(self, isolated_meta_path):
        """Test install function adds finder to meta_path."""
        # Remove first
        if loader._finder in sys.meta_path:
            sys.meta_path.remove(loader._finder)
        
        loader.install()
        
        assert loader._finder in sys.meta_path
    
    def test_uninstall_function(self, isolated_meta_path):
        """Test uninstall function removes finder."""
        loader.install()
        assert loader._finder in sys.meta_path
        
        loader.uninstall()
        
        assert loader._finder not in sys.meta_path
    
    def test_install_is_idempotent(self, isolated_meta_path):
        """Test that calling install multiple times is safe."""
        loader.install()
        count_before = sys.meta_path.count(loader._finder)
        
        loader.install()
        count_after = sys.meta_path.count(loader._finder)
        
        assert count_before == count_after == 1
