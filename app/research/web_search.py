import os

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


def web_search(query, max_results=5):
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in .env")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        max_results=max_results
    )

    results = []

    for result in response.get("results", []):
        results.append({
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content")
        })

    return results