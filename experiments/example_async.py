"""Asynchronous example of using hive jars for LLM execution."""

import sys
import asyncio
from pathlib import Path

# Add prompts directory to path
prompts_dir = Path(__file__).parent / "prompts"
sys.path.insert(0, str(prompts_dir))

import hive
from hive import mock_jar, anthropic_jar
from prompts.demo import summarize
from prompts.system import generic


async def example_basic_async():
    """Basic async example with jar."""
    print("=" * 60)
    print("ASYNC BASIC EXAMPLE")
    print("=" * 60)
    
    jar = mock_jar()
    
    async with jar:
        # Must await in async context
        response = await summarize(text="This is a long article about AI.")
    
    print(f"\nAsync response: {response}")


async def example_async_chat():
    """Async chat loop with stateful jar."""
    print("\n" + "=" * 60)
    print("ASYNC STATEFUL CHAT EXAMPLE")
    print("=" * 60)
    
    # Create reusable jar
    jar = mock_jar()
    
    messages = [
        "What is Python?",
        "How do I use async/await?",
        "Can you give me an example?",
        "Thanks!"
    ]
    
    for i, msg in enumerate(messages):
        print(f"\nYou [{i+1}]: {msg}")
        
        async with jar:
            # Must await in async context
            response = await summarize(text=msg)
        
        print(f"Assistant [{i+1}]: {response[:80]}...")
    
    # Access jar state
    print(f"\n📊 Jar Stats:")
    print(f"   Total messages: {jar.message_count}")
    print(f"   History length: {len(jar.get_history())}")


async def example_concurrent_requests():
    """Example: Concurrent async requests with multiple jars."""
    print("\n" + "=" * 60)
    print("CONCURRENT REQUESTS EXAMPLE")
    print("=" * 60)
    
    jar1 = mock_jar()
    jar2 = mock_jar()
    jar3 = mock_jar()
    
    # Define tasks
    async def task1():
        async with jar1:
            return await summarize(text="Explain Python")
    
    async def task2():
        async with jar2:
            return await summarize(text="Explain JavaScript")
    
    async def task3():
        async with jar3:
            return await summarize(text="Explain Rust")
    
    # Run concurrently
    print("\n🚀 Running 3 requests concurrently...")
    results = await asyncio.gather(task1(), task2(), task3())
    
    for i, result in enumerate(results, 1):
        print(f"\nResponse {i}: {result[:60]}...")


async def example_async_nested_jars():
    """Example: Nested async jar contexts."""
    print("\n" + "=" * 60)
    print("ASYNC NESTED JARS EXAMPLE")
    print("=" * 60)
    
    outer_jar = mock_jar()
    inner_jar = mock_jar()
    
    async with outer_jar:
        print("\n🔵 Using outer jar:")
        response1 = await summarize(text="What is AI?")
        print(response1[:80] + "...")
        
        async with inner_jar:
            print("\n🔴 Using inner jar (overrides):")
            response2 = await summarize(text="What is ML?")
            print(response2[:80] + "...")
        
        print("\n🔵 Back to outer jar:")
        response3 = await summarize(text="Follow up")
        print(response3[:80] + "...")
    
    print(f"\n📊 Outer jar: {outer_jar.message_count} messages")
    print(f"📊 Inner jar: {inner_jar.message_count} messages")


async def example_with_system_prompt():
    """Example: Async jar with system prompt."""
    print("\n" + "=" * 60)
    print("ASYNC SYSTEM PROMPT EXAMPLE")
    print("=" * 60)
    
    jar = mock_jar()
    
    # Add system prompt
    system = generic()
    jar.add_system_prompt(system)
    
    print(f"\nAdded system prompt: {system[:80]}...")
    
    # Use jar
    async with jar:
        response = await summarize(text="Explain quantum computing")
    
    print(f"\nResponse: {response[:100]}...")
    print(f"📊 History: {len(jar.get_history())} messages")


async def main():
    """Run all async examples."""
    await example_basic_async()
    await example_async_chat()
    await example_concurrent_requests()
    await example_async_nested_jars()
    await example_with_system_prompt()
    
    print("\n" + "=" * 60)
    print("✅ All async examples completed!")
    print("=" * 60)
    
    # Uncomment to try with real Anthropic API:
    # jar = anthropic_jar(model="claude-3-5-sonnet-20241022", temperature=0.7)
    # async with jar:
    #     response = await summarize(text="Your text here")
    #     print(response)


if __name__ == "__main__":
    asyncio.run(main())
