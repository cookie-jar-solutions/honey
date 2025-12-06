"""Sample usage of hive with jars."""

import hive
from hive import mock_jar
from prompts import demo, system

# Without jar - returns rendered templates
system_prompt = system.generic()
summary_prompt = demo.summarize(text="This is a sample text to be summarized.")

print("=" * 60)
print("WITHOUT JAR (Rendered Templates)")
print("=" * 60)
print("\nSystem Prompt:")
print(system_prompt)
print("\nSummary Prompt:")
print(summary_prompt)

# With jar - executes with mock LLM
print("\n" + "=" * 60)
print("WITH MOCK JAR (LLM Execution)")
print("=" * 60)

jar = mock_jar()
jar.add_system_prompt(system_prompt)

with jar:
    response = demo.summarize(text="This is a sample text to be summarized.")

print("\nLLM Response:")
print(response)
print(f"\n📊 Jar Stats: {jar.message_count} messages, {jar.total_tokens} tokens")