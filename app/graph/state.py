from typing import TypedDict, List, Dict, Any


class ResearchState(TypedDict, total=False):
    # User's research question
    query: str

    # Subtasks produced by the planner node
    subtasks: List[str]

    # Evidence retrieved from PDF / vector database
    evidence: List[Dict[str, Any]]

    # Claims extracted from the evidence
    claims: str

    # Verification results for the claims
    verification_results: str

    # Web re-verification results
    web_verification_results: str

    # Critic's decision and reasoning
    critic_result: str

    # Search query generated when more research is needed
    research_query: str

    # Tool selected by the critic for the next iteration
    # (PDF / ACADEMIC / WEB / NONE)
    selected_tool: str

    # External research results (web or academic)
    web_results: List[Dict[str, Any]]

    # Structured citation validation report
    citation_report: Dict[str, Any]

    # Current self-correction iteration
    iteration: int

    # Maximum allowed iterations
    max_iterations: int

    # Final answer/report
    final_report: str
