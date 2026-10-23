from app.research.web_search import web_search


query = "Siamese-PTM clinical trial matching"

results = web_search(
    query,
    max_results=5
)

print("Query:", query)
print("Results:", len(results))

for i, result in enumerate(results, start=1):
    print(f"\n--- RESULT {i} ---")
    print("Title:", result["title"])
    print("URL:", result["url"])
    print("Content:", result["content"][:300])