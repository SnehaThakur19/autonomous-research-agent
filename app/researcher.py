from tavily import TavilyClient
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize clients
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


# 1. Get user's question
query = input("What do you want to research? ")


# 2. Search the web using Tavily
response = tavily.search(
    query=query,
    max_results=5
)


# 3. Convert search results into context
context = ""

for i, result in enumerate(response["results"], start=1):
    context += f"""
SOURCE {i}
Title: {result["title"]}
URL: {result["url"]}
Content: {result["content"]}

"""


# 4. Send the question + sources to Groq
prompt = f"""
You are a research assistant.

Answer the user's question using ONLY the information provided
in the sources below.

If the sources do not contain enough information, say:
"Insufficient information in the provided sources."

User Question:
{query}

Sources:
{context}

Give a clear and concise answer.
Mention the relevant source numbers in your answer.
"""


completion = groq.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
)


# 5. Print the final answer
print("\n" + "=" * 60)
print("RESEARCH ANSWER")
print("=" * 60)

print(completion.choices[0].message.content)


print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for i, result in enumerate(response["results"], start=1):
    print(f"[{i}] {result['title']}")
    print(result["url"])
    print()