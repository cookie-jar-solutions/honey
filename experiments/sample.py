from hive import loader
from prompts import demo, system

system_prompt = system.generic()
summary_prompt = demo.summarize(text="This is a sample text to be summarized.")

print("System Prompt:")
print(system_prompt)
print("\nSummary Prompt:")
print(summary_prompt)