from app.research.query_generator import generate_research_query


claim = """
Siamese-PTM achieved a statistically significant
40% performance gain over the Random Forest baseline.
"""

critic_feedback = """
The evidence confirms a 40% improvement, but it is unclear
whether the improvement was specifically over Random Forest.
Find direct evidence confirming the baseline and comparison.
"""


query = generate_research_query(
    claim,
    critic_feedback
)

print("Generated Research Query:")
print(query)