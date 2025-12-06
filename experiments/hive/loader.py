"""Custom import loader for .hny (honey) prompt template files.

This module provides a custom import mechanism that allows .hny files to be imported
as Python modules. Each prompt definition in a .hny file becomes a callable function
that renders the Jinja2 template with provided keyword arguments.

Usage:
    # Install the loader (do this once at the start of your program)
    import prompt_loader
    prompt_loader.install()
    
    # Then import .hny files like normal Python modules
    from summaries import summarize
    result = summarize(text="This is long text that needs to be summarized")
"""

import sys
import re
from pathlib import Path
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Optional, Dict, Any

try:
    from jinja2 import Template
except ImportError:
    # Fallback to simple string formatting if jinja2 is not available
    class Template:
        def __init__(self, template_str):
            self.template_str = template_str
        
        def render(self, **kwargs):
            # Simple {{variable}} replacement
            result = self.template_str
            for key, value in kwargs.items():
                result = result.replace("{{" + key + "}}", str(value))
                result = result.replace("{{ " + key + " }}", str(value))
            return result


def parse_hny_file(filepath: Path) -> Dict[str, str]:
    """Parse a .hny file into a dictionary of prompt name -> template content.
    
    Format:
        prompt_name
        template content with {{variables}}
        --- (separator: 3 or more dashes)
        
        another_prompt
        more template content
        --------
    
    Args:
        filepath: Path to the .hny file
        
    Returns:
        Dictionary mapping prompt names to their template strings
    """
    content = filepath.read_text(encoding='utf-8')
    
    # Split by separator lines (3 or more dashes)
    sections = re.split(r'\n-{3,}\n', content)
    
    prompts = {}
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Split into first line (name) and rest (template)
        lines = section.split('\n', 1)
        if len(lines) < 2:
            # If only one line, treat it as name with empty template
            name = lines[0].strip()
            template = ""
        else:
            name = lines[0].strip()
            template = lines[1].strip()
        
        if name:
            prompts[name] = template
    
    return prompts


def create_prompt_function(template_str: str):
    """Create a callable function that renders a Jinja2 template.
    
    Args:
        template_str: The Jinja2 template string
        
    Returns:
        A function that accepts keyword arguments and returns rendered template
    """
    template = Template(template_str)
    
    def prompt_function(**kwargs) -> str:
        """Render the prompt template with the provided variables.
        
        Args:
            **kwargs: Template variables to render
            
        Returns:
            Rendered prompt string
        """
        return template.render(**kwargs)
    
    # Store the template string as an attribute for inspection
    prompt_function.__template__ = template_str
    prompt_function.__doc__ = f"Render prompt template:\n\n{template_str[:200]}{'...' if len(template_str) > 200 else ''}"
    
    return prompt_function


class HnyLoader(Loader):
    """Loader for .hny prompt template files."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
    
    def create_module(self, spec: ModuleSpec) -> Optional[ModuleType]:
        """Return None to use default module creation."""
        return None
    
    def exec_module(self, module: ModuleType) -> None:
        """Execute the module by parsing .hny file and creating prompt functions."""
        # Parse the .hny file
        prompts = parse_hny_file(self.filepath)
        
        # Create a function for each prompt and add to module
        for name, template_str in prompts.items():
            func = create_prompt_function(template_str)
            setattr(module, name, func)
        
        # Add metadata
        module.__file__ = str(self.filepath)
        module.__prompts__ = list(prompts.keys())


class HnyFinder(MetaPathFinder):
    """Finder for .hny files on sys.path."""
    
    def find_spec(self, fullname: str, path: Optional[list] = None, target: Optional[ModuleType] = None) -> Optional[ModuleSpec]:
        """Try to find a .hny file matching the module name.
        
        Args:
            fullname: Full module name (e.g., 'summaries')
            path: Package search path (None for top-level modules)
            target: Module object (usually None)
            
        Returns:
            ModuleSpec if found, None otherwise
        """
        # Split module name to handle packages
        parts = fullname.split('.')
        module_name = parts[-1]
        
        # Determine search paths
        if path is None:
            search_paths = sys.path
        else:
            search_paths = path
        
        # Look for .hny file
        for search_path in search_paths:
            search_dir = Path(search_path)
            if not search_dir.exists():
                continue
                
            hny_file = search_dir / f"{module_name}.hny"
            if hny_file.exists() and hny_file.is_file():
                return ModuleSpec(
                    name=fullname,
                    loader=HnyLoader(hny_file),
                    origin=str(hny_file),
                    is_package=False
                )
        
        return None


# Global finder instance
_finder = HnyFinder()


def install():
    """Install the .hny file loader into Python's import system.
    
    This should be called once at the start of your program, before importing
    any .hny files.
    """
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)


def uninstall():
    """Remove the .hny file loader from Python's import system."""
    if _finder in sys.meta_path:
        sys.meta_path.remove(_finder)


# Auto-install when this module is imported
install()
