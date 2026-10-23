import re

from langgraph.graph import StateGraph, START, END

from app.graph.state import ResearchState

from app.graph.nodes import (
    planner_node,
    retrieve_node,
    claim_extraction_node,
    verification_node,
    citation_validation_node,
    critic_node,
    pdf_re_research_node,
    web_research_node,
    academic_research_node,
    web_reverification_node,
    report_node,
)


def route_after_critic(state: ResearchState):

    critic_result = state.get("critic_result", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if "NEEDS_RESEARCH" in critic_result:

        if iteration < max_iterations:
            return "research"

        # Maximum iterations reached
        return "report"

    # Research is good
    return "report"


def route_after_increment(state: ResearchState):
    """Alternative tool selection: pick the next retrieval tool
    based on the critic's recommendation."""
    critic_result = state.get("critic_result", "")

    match = re.search(
        r"Tool\s*:\s*(WEB|ACADEMIC|PDF|NONE)",
        critic_result,
        re.IGNORECASE
    )

    tool = match.group(1).upper() if match else "WEB"

    if tool == "PDF":
        return "pdf_re_research"

    if tool == "ACADEMIC":
        return "academic_research"

    # Default and unknown tools fall back to web search
    return "web_research"


def increment_iteration(state: ResearchState):
    current_iteration = state.get("iteration", 0)

    return {
        "iteration": current_iteration + 1
    }


# Create the research graph
builder = StateGraph(ResearchState)


# Add nodes
builder.add_node("plan", planner_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("extract_claims", claim_extraction_node)
builder.add_node("verify", verification_node)
builder.add_node("validate_citations", citation_validation_node)
builder.add_node("critic", critic_node)
builder.add_node("pdf_re_research", pdf_re_research_node)
builder.add_node("web_research", web_research_node)
builder.add_node("academic_research", academic_research_node)
builder.add_node(
    "increment_iteration",
    increment_iteration
)
builder.add_node("web_reverify", web_reverification_node)
builder.add_node("report", report_node)

# Normal flow
builder.add_edge(START, "plan")
builder.add_edge("plan", "retrieve")
builder.add_edge("retrieve", "extract_claims")
builder.add_edge("extract_claims", "verify")
builder.add_edge("verify", "validate_citations")
builder.add_edge("validate_citations", "critic")


# Conditional routing after critic: keep researching or report
builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "research": "increment_iteration",
        "report": "report",
    }
)

# Conditional routing after increment: choose an alternative tool
builder.add_conditional_edges(
    "increment_iteration",
    route_after_increment,
    {
        "pdf_re_research": "pdf_re_research",
        "web_research": "web_research",
        "academic_research": "academic_research",
    }
)

builder.add_edge(
    "report",
    END
)

# PDF re-research expands local evidence → re-verify against it
builder.add_edge(
    "pdf_re_research",
    "verify"
)

# Web / academic research → re-verify against external evidence
builder.add_edge(
    "web_research",
    "web_reverify"
)

builder.add_edge(
    "academic_research",
    "web_reverify"
)

builder.add_edge(
    "web_reverify",
    "validate_citations"
)


# Compile the graph
research_graph = builder.compile()
