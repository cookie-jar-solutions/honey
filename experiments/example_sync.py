"""Synchronous example of using hive jars for LLM execution."""

import sys
from pathlib import Path

# Add prompts directory to path
prompts_dir = Path(__file__).parent / "prompts"
sys.path.insert(0, str(prompts_dir))

import hive
from hive import mock_jar, openai_jar
from prompts.demo import summarize
from prompts.system import generic

def example_basic():
    """Basic example: with and without jar."""
    print("=" * 60)
    print("BASIC EXAMPLE: With and Without Jar")
    print("=" * 60)
    
    # Without jar - just returns rendered template
    template = summarize(text="This is a long article about AI.")
    print("\nWithout jar (rendered template):")
    print(template)
    
    # With jar - executes with mock LLM
    with mock_jar():
        response = summarize(text="This is a long article about AI.")
    print("\nWith mock jar (LLM response):")
    print(response)


def example_stateful_chat():
    """Example: Stateful chat loop with jar."""
    print("\n" + "=" * 60)
    print("STATEFUL CHAT EXAMPLE")
    print("=" * 60)
    
    # Create a reusable jar
    jar = mock_jar()
    
    messages = [
        "What is Python?",
        "How do I use async/await?",
        "Thanks!"
    ]
    
    for i, msg in enumerate(messages):
        print(f"\nYou [{i+1}]: {msg}")
        
        with jar:
            response = summarize(text=msg)
        
        print(f"Assistant [{i+1}]: {response}")
    
    # Check jar state
    print(f"\n📊 Jar Stats:")
    print(f"   Messages: {jar.message_count}")
    print(f"   History length: {len(jar.get_history())}")


def example_nested_jars():
    """Example: Nested jars (inner overrides outer)."""
    print("\n" + "=" * 60)
    print("NESTED JARS EXAMPLE")
    print("=" * 60)
    
    outer_jar = mock_jar()
    inner_jar = mock_jar()
    
    text = "Explain machine learning"
    
    with outer_jar:
        print("\n🔵 Using outer jar:")
        response1 = summarize(text=text)
        print(response1[:80] + "...")
        
        with inner_jar:
            print("\n🔴 Using inner jar (overrides):")
            response2 = summarize(text=text)
            print(response2[:80] + "...")
        
        print("\n🔵 Back to outer jar:")
        response3 = summarize(text="Follow up question")
        print(response3[:80] + "...")
    
    print(f"\n📊 Outer jar messages: {outer_jar.message_count}")
    print(f"📊 Inner jar messages: {inner_jar.message_count}")


def example_system_prompt():
    """Example: Using system prompts with jar."""
    print("\n" + "=" * 60)
    print("SYSTEM PROMPT EXAMPLE")
    print("=" * 60)
    
    jar = mock_jar()
    
    # Add system prompt to jar
    system_prompt = generic()
    jar.add_system_prompt(system_prompt)
    
    print(f"\nAdded system prompt to jar")
    print(f"System prompt: {system_prompt[:100]}...")
    
    # Now use jar for completion
    with jar:
        response = summarize(text="Explain quantum computing")
    
    print(f"\nResponse: {response[:100]}...")
    print(f"\n📊 History has {len(jar.get_history())} messages")


if __name__ == "__main__":
    example_basic()
    example_stateful_chat()
    example_nested_jars()
    example_system_prompt()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    
    # Uncomment to try with real OpenAI API:
    # jar = openai_jar(model="gpt-4", temperature=0.7)
    # with jar:
    #     response = summarize(text="Your text here")
    #     print(response)
