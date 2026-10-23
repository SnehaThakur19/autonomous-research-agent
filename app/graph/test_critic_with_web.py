from app.graph.nodes import critic_node


state = {
    "query": "What performance improvement did Siamese-PTM achieve compared to the baseline?",

    "claims": """
1. Siamese-PTM achieved a statistically significant 40% performance
gain over the rule-based Random Forest baseline with F1 improving
from 0.66 to 0.92, p=.006.

2. Siamese-PTM outperformed 7 of 10 single-encoder models.
""",

    "verification_results": """
Claim 1:
Status: SUPPORTED
Reason: The PDF evidence appears to support the reported numbers.

Claim 2:
Status: SUPPORTED
Reason: The PDF reports that Siamese-PTM outperformed 7 of 10 models.
""",

    "web_verification_results": """
Claim 1:
Status: NOT SUPPORTED
Reason: The web evidence does not explicitly confirm that Siamese-PTM
achieved the 40% gain, F1 0.92, F1 0.66, or p=.006 against the stated baseline.

Claim 2:
Status: NOT SUPPORTED
Reason: The web evidence only says Siamese-PTM outperformed several
single-encoder approaches and does not confirm exactly 7 of 10.
"""
}


result = critic_node(state)


print("\nUpdated Critic Result:")
print("----------------------")
print(result["critic_result"])