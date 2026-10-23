from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

query = input("What do you want to search? ")

response = client.search(query)

for result in response["results"]:
    print("\nTitle:", result["title"])
    print("URL:", result["url"])
    print("Content:", result["content"])