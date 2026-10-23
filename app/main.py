"""Autonomous research agent — command line entrypoint.

Usage:
    python -m app.main "Your complex research question"
    python -m app.main "Your question" --pdf data/research_paper.pdf
    python -m app.main "Your question" --max-iterations 3 --verbose
"""

import argparse

from app.graph.workflow import research_graph


def run_research(query, max_iterations=3, verbose=False):
    """Run the full research graph and return the result state."""
    initial_state = {
        "query": query,
        "iteration": 0,
        "max_iterations": max_iterations
    }

    result = research_graph.invoke(initial_state)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous AI research agent"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Research question (prompted if omitted)"
    )
    parser.add_argument(
        "--pdf",
        nargs="+",
        help="PDF path(s) to ingest before researching"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector store when ingesting PDFs"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum self-correction iterations (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print subtasks, critic result, and citation report"
    )

    args = parser.parse_args()

    if args.pdf:
        from app.ingest import ingest_pdfs

        ingest_pdfs(args.pdf, rebuild=args.rebuild)

    query = args.query

    if not query:
        query = input("Enter your research question: ")

    result = run_research(
        query,
        max_iterations=args.max_iterations,
        verbose=args.verbose
    )

    if args.verbose:
        subtasks = result.get("subtasks", [])

        if subtasks:
            print("\nResearch subtasks:")
            for i, subtask in enumerate(subtasks, start=1):
                print(f"  {i}. {subtask}")

        print(f"\nIterations used: {result.get('iteration', 0)}")

        print(f"\nCritic:\n{result.get('critic_result', '')}")

        citation_report = result.get("citation_report", {})

        if citation_report:
            print(f"\n{citation_report.get('text', '')}")

    print("\n" + "=" * 70 + "\n")

    print(result.get("final_report", "No report generated."))


if __name__ == "__main__":
    main()
