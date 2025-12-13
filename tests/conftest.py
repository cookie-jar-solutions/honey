"""Shared pytest fixtures for honey tests."""

import sys
import pytest
from pathlib import Path
from typing import Generator


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def prompts_dir(fixtures_dir) -> Path:
    """Path to test prompt fixtures."""
    return fixtures_dir / "prompts"


@pytest.fixture
def temp_hny_file(tmp_path):
    """Create a temporary .hny file with custom content.
    
    Automatically adds tmp_path to sys.path and invalidates import caches.
    
    Usage:
        def test_something(temp_hny_file):
            hny_path = temp_hny_file("content", "module_name.hny")
    """
    import importlib
    
    # Add tmp_path to sys.path so .hny files can be found
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    
    def _create_hny(content: str, filename: str) -> Path:
        hny_file = tmp_path / filename
        hny_file.write_text(content)
        
        # Invalidate import caches so the new module can be found
        importlib.invalidate_caches()
        
        return hny_file
    
    yield _create_hny
    
    # Clean up: remove tmp_path from sys.path
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


@pytest.fixture
def isolated_sys_path(tmp_path) -> Generator[None, None, None]:
    """Ensure temp path is in sys.path and clean up sys.modules.
    
    This fixture ensures that dynamically imported modules are cleaned up.
    """
    import importlib
    
    # Ensure tmp_path is in sys.path for the test
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    
    # Track modules before test
    original_modules = set(sys.modules.keys())
    
    yield
    
    # Clean up any modules imported during test
    new_modules = set(sys.modules.keys()) - original_modules
    for module_name in new_modules:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    # Invalidate import caches
    importlib.invalidate_caches()


@pytest.fixture
def isolated_meta_path() -> Generator[None, None, None]:
    """Isolate sys.meta_path changes during test."""
    original_meta_path = sys.meta_path.copy()
    yield
    sys.meta_path.clear()
    sys.meta_path.extend(original_meta_path)


@pytest.fixture
def mock_jar():
    """Create a mock jar instance for testing."""
    from honey import mock_jar as MockJarClass
    return MockJarClass()


@pytest.fixture
def clean_jar_context():
    """Ensure no jar contexts are active before/after test."""
    from honey.jars import base
    
    # Clear any existing contexts
    base._sync_jar.set(None)
    base._async_jar.set(None)
    
    yield
    
    # Clean up after test
    base._sync_jar.set(None)
    base._async_jar.set(None)


@pytest.fixture
def sample_prompts():
    """Sample prompt templates for testing."""
    return {
        "simple": "Hello, {{name}}!",
        "multi_var": "{{greeting}} {{name}}, welcome to {{place}}!",
        "conditional": "{% if premium %}Premium{% else %}Basic{% endif %} user: {{name}}",
        "loop": "{% for item in items %}{{item}}\n{% endfor %}",
    }


@pytest.fixture(autouse=True)
def ensure_loader_installed():
    """Ensure the .hny loader is installed for all tests."""
    from honey import loader
    loader.install()
