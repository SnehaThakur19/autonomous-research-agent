from app.graph.workflow import research_graph


state = {
    "query": "Verify whether Siamese-PTM specifically achieved an F1 score of 0.92 versus Random Forest 0.66 with a statistically significant 40% improvement (p=0.006).",
    "iteration": 0,
    "max_iterations": 3
}


result = research_graph.invoke(state)


print("\nFINAL RESEARCH REPORT")
print("=====================")

print(result.get("final_report"))


print("\n\nITERATION COUNT")
print("================")

print(result.get("iteration"))