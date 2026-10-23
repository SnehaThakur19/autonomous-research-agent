from app.graph.nodes import (
    retrieve_node,
    claim_extraction_node,
    verification_node,
    critic_node,
    web_research_node,
    web_reverification_node
)

state = {
    "query": "What performance improvement did Siamese-PTM achieve compared to the baseline?",
    "iteration": 0,
    "max_iterations": 3
}


# Node 1: Retrieve + Rerank
result = retrieve_node(state)
state.update(result)

print("Retrieved Evidence:")
print("-------------------")

for i, item in enumerate(state["evidence"], start=1):
    print(f"\nEvidence {i}")
    print("Page:", item["metadata"].get("page"))
    print("Chunk:", item["metadata"].get("chunk_id"))
    print("Score:", item.get("rerank_score"))
    print("Text:", item["text"][:200])


# Node 2: Claim Extraction
result = claim_extraction_node(state)
state.update(result)

print("\n\nExtracted Claims:")
print("----------------")
print(state["claims"])


# Node 3: Verification
result = verification_node(state)
state.update(result)

print("\n\nVerification Results:")
print("--------------------")
print(state["verification_results"])


# Node 4: Critic
result = critic_node(state)
state.update(result)

print("\n\nCritic Result:")
print("-------------")
print(state["critic_result"])


# Node 5: Web Research
result = web_research_node(state)
state.update(result)

print("\n\nResearch Query:")
print("--------------")
print(state["research_query"])


print("\n\nWeb Research Results:")
print("--------------------")

for i, item in enumerate(state["web_results"], start=1):
    print(f"\nResult {i}")
    print("Title:", item.get("title"))
    print("URL:", item.get("url"))
    print("Content:", item.get("content", "")[:300])

# Node 6: Web Re-verification
result = web_reverification_node(state)
state.update(result)

print("\n\nWeb Re-verification Results:")
print("---------------------------")
print(state["web_verification_results"])


# Node 7: Critic AGAIN
result = critic_node(state)
state.update(result)

print("\n\nFinal Critic Result:")
print("-------------------")
print(state["critic_result"])