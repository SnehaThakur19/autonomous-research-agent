import re

from app.report_generator import generate_report
from app.verification.verifier import verify_claims
from app.graph.web_verification import verify_claims_with_web

from app.rag.vectorstore import get_vector_store
from app.rag.retriever import retrieve
from app.rag.reranker import rerank

from app.verification.claim_extractor import extract_claims
from app.verification.critic import critique_research
from app.verification.citation_validator import validate_research_citations

from app.research.task_decomposer import decompose_query
from app.research.query_generator import generate_research_query
from app.research.web_search import web_search
from app.research.academic_search import academic_search

from app.graph.state import ResearchState


def merge_evidence(*evidence_lists):
    """Merge evidence lists, deduplicating by chunk_id and
    keeping the best (lowest) distance for each chunk."""
    merged = {}

    for evidence_list in evidence_lists:
        for item in evidence_list:
            chunk_id = item.get("metadata", {}).get("chunk_id")

            if chunk_id is None:
                continue

            if chunk_id not in merged:
                merged[chunk_id] = item
            else:
                best_distance = merged[chunk_id].get("distance")
                new_distance = item.get("distance")

                if best_distance is None or (
                    new_distance is not None
                    and new_distance < best_distance
                ):
                    merged[chunk_id] = item

    return list(merged.values())


def selected_tool(state: ResearchState):
    """Extract the critic's tool recommendation from its output."""
    critic_result = state.get("critic_result", "")

    match = re.search(
        r"Tool\s*:\s*(WEB|ACADEMIC|PDF|NONE)",
        critic_result,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    # Default corrective tool
    return "WEB"


# Node 1: Planner (query decomposition)
def planner_node(state: ResearchState):
    # Reuse a plan generated earlier (e.g. by the web UI's
    # plan step) instead of re-decomposing
    if state.get("subtasks"):
        return {}

    query = state["query"]

    subtasks = decompose_query(query)

    return {
        "subtasks": subtasks
    }


# Node 2: Retrieve + Rerank (per subtask, merged)
def retrieve_node(state: ResearchState):
    query = state["query"]

    subtasks = state.get("subtasks", [query])

    collection = get_vector_store()

    all_retrieved = []

    for subtask in subtasks:
        retrieved = retrieve(
            collection,
            subtask,
            top_k=10
        )

        all_retrieved.append(retrieved)

    merged = merge_evidence(*all_retrieved)

    # Rerank the merged pool against the original question
    evidence = rerank(
        query,
        merged,
        top_k=5
    )

    return {
        "evidence": evidence
    }


# Node 3: Claim Extraction
def claim_extraction_node(state: ResearchState):
    query = state["query"]
    evidence = state["evidence"]

    claims = extract_claims(
        query,
        evidence
    )

    return {
        "claims": claims
    }


# Node 4: Verification
def verification_node(state: ResearchState):
    claims = state["claims"]
    evidence = state["evidence"]

    verification_results = verify_claims(
        claims,
        evidence
    )

    return {
        "verification_results": verification_results
    }


# Node 5: Citation Validation
def citation_validation_node(state: ResearchState):
    verification_results = state.get("verification_results", "")
    evidence = state.get("evidence", [])

    citation_report = validate_research_citations(
        verification_results,
        evidence
    )

    return {
        "citation_report": citation_report
    }


# Node 6: Critic
def critic_node(state: ResearchState):

    query = state["query"]
    claims = state["claims"]
    verification_results = state["verification_results"]

    web_verification_results = state.get(
        "web_verification_results",
        ""
    )

    critic_result = critique_research(
        query,
        claims,
        verification_results,
        web_verification_results
    )

    return {
        "critic_result": critic_result,
        "selected_tool": selected_tool({
            "critic_result": critic_result
        })
    }


# Node 7: PDF Re-research (alternative tool: better local retrieval)
def pdf_re_research_node(state: ResearchState):
    claims = state["claims"]
    critic_result = state["critic_result"]

    existing_evidence = state.get("evidence", [])

    research_query = generate_research_query(
        claims,
        critic_result,
        search_type="local document"
    )

    collection = get_vector_store()

    retrieved = retrieve(
        collection,
        research_query,
        top_k=10
    )

    new_evidence = rerank(
        research_query,
        retrieved,
        top_k=5
    )

    evidence = merge_evidence(existing_evidence, new_evidence)

    return {
        "research_query": research_query,
        "evidence": evidence
    }


# Node 8: Web Research (alternative tool: general web search)
def web_research_node(state: ResearchState):
    claims = state["claims"]
    critic_result = state["critic_result"]

    research_query = generate_research_query(
        claims,
        critic_result,
        search_type="web"
    )

    web_results = web_search(
        research_query,
        max_results=5
    )

    return {
        "research_query": research_query,
        "web_results": web_results
    }


# Node 9: Academic Research (alternative tool: arXiv paper search)
def academic_research_node(state: ResearchState):
    claims = state["claims"]
    critic_result = state["critic_result"]

    research_query = generate_research_query(
        claims,
        critic_result,
        search_type="academic paper"
    )

    academic_results = academic_search(
        research_query,
        max_results=5
    )

    return {
        "research_query": research_query,
        "web_results": academic_results
    }


# Node 10: External Re-verification (web or academic evidence)
def web_reverification_node(state: ResearchState):

    claims = state["claims"]
    web_results = state["web_results"]

    web_verification_results = verify_claims_with_web(
        claims,
        web_results
    )

    return {
        "web_verification_results": web_verification_results
    }


# Node 11: Final Report Generator
def report_node(state: ResearchState):

    report = generate_report(
        query=state["query"],
        claims=state["claims"],
        verification_results=state["verification_results"],
        critic_result=state["critic_result"],
        evidence=state["evidence"],
        web_results=state.get("web_results", []),
        citation_report=state.get("citation_report"),
        subtasks=state.get("subtasks", [])
    )

    return {
        "final_report": report
    }
