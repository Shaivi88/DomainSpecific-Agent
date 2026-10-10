import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

# "groq/" prefix routes via LiteLLM, which preserves "openai/gpt-oss-120b"
# as the literal model name sent to Groq's API — needed because custom_openai=True
# strips the "openai/" prefix before the API call, causing a 404.
llm = LLM(
    model="groq/openai/gpt-oss-120b",
    temperature=0.7,
)
