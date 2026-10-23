from app.graph.state import ResearchState


state: ResearchState = {
    "query": "What performance improvement did Siamese-PTM achieve?",
    "evidence": [],
    "claims": "",
    "verification_results": "",
    "critic_result": "",
    "research_query": "",
    "web_results": [],
    "iteration": 0,
    "max_iterations": 3,
    "final_report": ""
}


print("Research State Created")
print("----------------------")

for key, value in state.items():
    print(f"{key}: {value}")