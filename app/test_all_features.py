"""Feature test suite for the autonomous research agent.

Runs every component once and prints PASS/FAIL per feature.

Usage (from the project root):
    venv\\Scripts\\python.exe -m app.test_all_features
"""

import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

RESULTS = []


def test(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, "PASS", ""))
        print(f"[PASS] {name} :: {detail}", flush=True)
        return detail
    except Exception as exc:
        RESULTS.append((name, "FAIL", str(exc)))
        print(f"[FAIL] {name} :: {exc}", flush=True)
        traceback.print_exc()
        return None


QUERY = (
    "What architecture is proposed in the paper "
    "and what results does it achieve?"
)

PDF_PATH = "data/research_paper.pdf"


def main():
    print("=" * 70, flush=True)
    print("FEATURE TEST SUITE", flush=True)
    print("=" * 70, flush=True)

    # 1. Environment
    def t_env():
        missing = [
            key for key in ("GROQ_API_KEY", "TAVILY_API_KEY")
            if not os.getenv(key)
        ]
        if missing:
            raise ValueError(f"Missing keys in .env: {missing}")
        return "GROQ_API_KEY and TAVILY_API_KEY present"

    test("env_keys", t_env)

    # 2. PDF loading
    def t_loader():
        from app.rag.loader import load_pdf

        pages = load_pdf(PDF_PATH)

        if not pages:
            raise ValueError("No pages extracted")

        return f"{len(pages)} pages from {PDF_PATH}"

    pages_result = test("pdf_loader", t_loader)

    # 3. Chunking
    def t_chunker():
        from app.rag.loader import load_pdf
        from app.rag.chunker import chunk_pages

        chunks = chunk_pages(load_pdf(PDF_PATH))

        if not chunks:
            raise ValueError("No chunks produced")

        sample = chunks[0]

        for field in ("text", "metadata"):
            if field not in sample:
                raise ValueError(f"Chunk missing '{field}'")

        for field in ("source", "page", "chunk_id"):
            if field not in sample["metadata"]:
                raise ValueError(f"Metadata missing '{field}'")

        return f"{len(chunks)} chunks, sample id={sample['metadata']['chunk_id']}"

    test("pdf_chunker", t_chunker)

    # 4. Embeddings
    def t_embeddings():
        from app.rag.loader import load_pdf
        from app.rag.chunker import chunk_pages
        from app.rag.embeddings import create_embeddings

        chunks = chunk_pages(load_pdf(PDF_PATH))[:10]

        embeddings = create_embeddings(chunks)

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding count mismatch")

        return f"{embeddings.shape[0]}x{embeddings.shape[1]} embeddings"

    test("embeddings", t_embeddings)

    # 5. Vector store
    def t_vectorstore():
        from app.rag.vectorstore import get_vector_store

        collection = get_vector_store()

        count = collection.count()

        if count == 0:
            raise ValueError("Vector store is empty - run app.ingest first")

        return f"collection 'research_documents' has {count} chunks"

    test("vectorstore", t_vectorstore)

    # 6. Retrieval
    retrieved_chunks = []

    def t_retriever():
        from app.rag.vectorstore import get_vector_store
        from app.rag.retriever import retrieve

        collection = get_vector_store()

        chunks = retrieve(collection, QUERY, top_k=10)

        if not chunks:
            raise ValueError("No chunks retrieved")

        if "distance" not in chunks[0]:
            raise ValueError("Retrieved chunk missing distance")

        retrieved_chunks.extend(chunks)

        return f"retrieved {len(chunks)} chunks"

    test("retriever", t_retriever)

    # 7. Reranker
    evidence = []

    def t_reranker():
        from app.rag.reranker import rerank

        if not retrieved_chunks:
            raise ValueError("No retrieved chunks to rerank")

        ranked = rerank(QUERY, retrieved_chunks, top_k=5)

        if not ranked:
            raise ValueError("Rerank returned nothing")

        if "rerank_score" not in ranked[0]:
            raise ValueError("Rerank output missing rerank_score")

        evidence.extend(ranked)

        return f"reranked to top {len(ranked)}, best score {ranked[0]['rerank_score']:.4f}"

    test("reranker", t_reranker)

    # 8. Query decomposition
    subtasks = []

    def t_decomposer():
        from app.research.task_decomposer import decompose_query

        result = decompose_query(QUERY)

        if not result:
            raise ValueError("No subtasks returned")

        subtasks.extend(result)

        return f"{len(result)} subtasks: {result}"

    test("task_decomposer", t_decomposer)

    # 9. Academic search (arXiv)
    def t_academic():
        from app.research.academic_search import academic_search

        results = academic_search(
            "Siamese network protein language model",
            max_results=3
        )

        if not results:
            raise ValueError("No arXiv results")

        for field in ("title", "url", "content"):
            if not results[0].get(field):
                raise ValueError(f"arXiv result missing '{field}'")

        return f"{len(results)} papers, first: {results[0]['title'][:60]}"

    test("academic_search", t_academic)

    # 10. Web search (Tavily)
    def t_web():
        from app.research.web_search import web_search

        results = web_search("protein language models fine-tuning", max_results=3)

        if not results:
            raise ValueError("No web results")

        return f"{len(results)} results, first: {results[0]['title'][:60]}"

    test("web_search", t_web)

    # 11. Claim extraction
    claims_text = ""

    def t_claims():
        from app.verification.claim_extractor import extract_claims

        if not evidence:
            raise ValueError("No evidence available")

        result = extract_claims(QUERY, evidence)

        if not result:
            raise ValueError("No claims extracted")

        claims_text_holder.append(result)

        return f"{len(result.splitlines())} lines of claims"

    claims_text_holder = []
    test("claim_extraction", t_claims)
    claims_text = claims_text_holder[0] if claims_text_holder else ""

    # 12. Claim verification
    verification_text = ""

    def t_verify():
        from app.verification.verifier import verify_claims

        if not claims_text:
            raise ValueError("No claims available")

        result = verify_claims(claims_text, evidence)

        for label in ("SUPPORTED", "Status:", "Claim"):
            if label not in result:
                raise ValueError(f"Verification output missing '{label}'")

        verification_text_holder.append(result)

        return f"verification output: {len(result)} chars"

    verification_text_holder = []
    test("claim_verification", t_verify)
    verification_text = (
        verification_text_holder[0] if verification_text_holder else ""
    )

    # 13. Citation validation (offline)
    def t_citations():
        from app.verification.citation_validator import (
            validate_research_citations,
        )

        if not verification_text:
            raise ValueError("No verification output to validate")

        report = validate_research_citations(verification_text, evidence)

        summary = report["summary"]

        if summary["total_claims"] == 0:
            raise ValueError("No claims parsed from verification output")

        return (
            f"parsed {summary['total_claims']} claims, "
            f"{summary['supported']} supported, "
            f"{summary['claims_with_invalid_citations']} with invalid citations"
        )

    test("citation_validation", t_citations)

    # 14. Critic
    critic_text = ""

    def t_critic():
        from app.verification.critic import critique_research

        if not claims_text or not verification_text:
            raise ValueError("Missing claims/verification for critic")

        result = critique_research(
            QUERY, claims_text, verification_text, ""
        )

        if "Decision:" not in result:
            raise ValueError("Critic output missing 'Decision:'")

        if "Tool:" not in result:
            raise ValueError("Critic output missing 'Tool:'")

        critic_text_holder.append(result)

        return f"critic: {result[:80].replace(chr(10), ' ')}..."

    critic_text_holder = []
    test("critic", t_critic)
    critic_text = critic_text_holder[0] if critic_text_holder else ""

    # 15. Query rewriting
    def t_rewrite():
        from app.research.query_generator import generate_research_query

        result = generate_research_query(
            claims_text or "a sample claim",
            critic_text or "needs more evidence",
            search_type="web"
        )

        if not result:
            raise ValueError("Empty rewritten query")

        return f"rewritten query: {result[:70]}"

    test("query_rewrite", t_rewrite)

    # 16. External (web) re-verification
    def t_web_verify():
        from app.graph.web_verification import verify_claims_with_web
        from app.research.web_search import web_search

        if not claims_text:
            raise ValueError("No claims available")

        results = web_search("protein language model fine-tuning results", max_results=3)

        result = verify_claims_with_web(claims_text, results)

        if "Status:" not in result:
            raise ValueError("Web verification output malformed")

        return f"re-verified against {len(results)} web sources"

    test("web_reverification", t_web_verify)

    # 17. Full end-to-end graph
    def t_graph():
        from app.graph.workflow import research_graph

        result = research_graph.invoke({
            "query": QUERY,
            "iteration": 0,
            "max_iterations": 2
        })

        if not result.get("final_report"):
            raise ValueError("No final report produced")

        if not result.get("subtasks"):
            raise ValueError("No subtasks in state")

        if not result.get("citation_report"):
            raise ValueError("No citation report in state")

        report = result["final_report"]

        with open("final_report.txt", "w", encoding="utf-8") as handle:
            handle.write(report)

            handle.write("\n\n--- CITATION VALIDATION REPORT ---\n\n")
            handle.write(
                result.get("citation_report", {}).get("text", "")
            )

        return (
            f"report {len(report)} chars, "
            f"{result.get('iteration', 0)} iterations, "
            f"saved to final_report.txt"
        )

    test("full_graph", t_graph)

    # Summary
    print("", flush=True)
    print("=" * 70, flush=True)
    passed = sum(1 for _, status, _ in RESULTS if status == "PASS")
    failed = sum(1 for _, status, _ in RESULTS if status == "FAIL")
    print(f"RESULT: {passed} passed, {failed} failed out of {len(RESULTS)}", flush=True)

    for name, status, error in RESULTS:
        if status == "FAIL":
            print(f"  FAILED: {name} - {error}", flush=True)

    print("=" * 70, flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
