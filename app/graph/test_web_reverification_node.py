from app.graph.nodes import web_reverification_node


state = {
    "claims": """
1. Siamese-PTM achieved a statistically significant 40% performance
gain over the rule-based Random Forest baseline with F1 improving
from 0.66 to 0.92, p=.006.

2. Siamese-PTM outperformed 7 of 10 single-encoder models.
""",

    "web_results": [
        {
            "title": "Test Web Result",
            "url": "https://example.com",
            "content": """
The retrieved web evidence discusses Siamese-PTM and clinical
trial matching. It mentions that Siamese-PTM outperformed several
single-encoder approaches, but it does not clearly confirm that
the F1 score of 0.92 belonged to Siamese-PTM or that the 40%
improvement was against Random Forest.
"""
        }
    ]
}


result = web_reverification_node(state)


print("\nWeb Re-verification Node:")
print("------------------------")
print(result["web_verification_results"])