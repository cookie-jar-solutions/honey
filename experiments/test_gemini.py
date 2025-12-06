"""Test Gemini jar integration."""

import sys
from pathlib import Path

# Add prompts directory to path
prompts_dir = Path(__file__).parent / "prompts"
sys.path.insert(0, str(prompts_dir))

import hive
from hive import gemini_jar
from prompts.demo import summarize

def test_gemini_jar():
    """Test that GeminiJar is properly configured."""
    print("Testing GeminiJar...")
    
    # Create a gemini jar instance
    jar = gemini_jar(model="gemini-2.0-flash-exp", temperature=0.7)
    
    print(f"✅ GeminiJar created successfully")
    print(f"   Model: {jar.config['model']}")
    print(f"   Temperature: {jar.config.get('temperature', 'not set')}")
    
    # Test that it has the required methods
    assert hasattr(jar, 'execute'), "Missing execute method"
    assert hasattr(jar, 'aexecute'), "Missing aexecute method"
    assert hasattr(jar, '__enter__'), "Missing sync context manager"
    assert hasattr(jar, '__aenter__'), "Missing async context manager"
    
    print(f"✅ All required methods present")
    print(f"✅ GeminiJar is ready to use!")
    
    print("\nTo use with real Gemini API:")
    print("  jar = gemini_jar(model='gemini-2.0-flash-exp', api_key='your-key')")
    print("  with jar:")
    print("      response = summarize(text='...')")

if __name__ == "__main__":
    test_gemini_jar()
