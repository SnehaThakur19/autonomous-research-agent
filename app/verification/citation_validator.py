import re


def parse_verification_results(verification_results):
    """Parse free-text verification output into structured records.

    Expects blocks in the format produced by the verifier:

        Claim 1:
        Status: SUPPORTED
        Reason: ...
        Evidence: Evidence 1, Evidence 3

    Returns a list of dicts:
        {claim_number, status, reason, cited_evidence}
    """
    records = []

    if not verification_results:
        return records

    # Split on "Claim N:" headers, keeping the numbers
    # (tolerates markdown decoration like "**Claim 1:**")
    parts = re.split(r"Claim\s+(\d+)[^:\n]*:", verification_results)

    # parts[0] is the preamble, then (number, body) pairs
    for i in range(1, len(parts) - 1, 2):
        claim_number = int(parts[i])
        body = parts[i + 1]

        status_match = re.search(r"Status[^:\n]*:\s*(.+)", body)
        reason_match = re.search(r"Reason[^:\n]*:\s*(.+)", body)
        evidence_match = re.search(r"Evidence[^:\n]*:\s*(.+)", body)

        # Strip markdown decoration (e.g. "**SUPPORTED**")
        status = (
            re.sub(r"[*_`#]", "", status_match.group(1)).strip().upper()
            if status_match
            else ""
        )
        reason = (
            re.sub(r"[*_`#]", "", reason_match.group(1)).strip()
            if reason_match
            else ""
        )

        cited_evidence = []

        if evidence_match:
            cited_evidence = [
                int(number)
                for number in re.findall(r"\d+", evidence_match.group(1))
            ]

        records.append({
            "claim_number": claim_number,
            "status": status.upper(),
            "reason": reason,
            "cited_evidence": cited_evidence
        })

    return records


def validate_citations(records, evidence):
    """Check that every cited evidence number actually exists.

    Adds to each record:
        valid_citations: cited numbers within 1..len(evidence)
        invalid_citations: cited numbers out of range
        sources: traceable source info for each valid citation
    """
    total = len(evidence)

    for record in records:
        valid = []
        invalid = []
        sources = []

        for number in record["cited_evidence"]:
            if 1 <= number <= total:
                valid.append(number)

                item = evidence[number - 1]
                metadata = item.get("metadata", {})

                sources.append({
                    "evidence_number": number,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id")
                })
            else:
                invalid.append(number)

        record["valid_citations"] = valid
        record["invalid_citations"] = invalid
        record["sources"] = sources

    return records


def build_citation_report(records):
    """Build a structured citation validation report.

    Returns a dict with per-claim records, a summary, and a
    human-readable text version for the critic/report prompts.
    """
    supported = [
        r for r in records if r["status"] == "SUPPORTED"
    ]
    partially = [
        r for r in records if r["status"].startswith("PARTIALLY")
    ]
    not_supported = [
        r for r in records if r["status"].startswith("NOT SUPPORTED")
    ]
    invalid = [
        r for r in records if r["invalid_citations"]
    ]
    uncited = [
        r for r in records if not r["cited_evidence"]
    ]

    summary = {
        "total_claims": len(records),
        "supported": len(supported),
        "partially_supported": len(partially),
        "not_supported": len(not_supported),
        "claims_with_invalid_citations": len(invalid),
        "claims_without_citations": len(uncited)
    }

    lines = ["Citation Validation Report", ""]

    lines.append(
        f"Total claims: {summary['total_claims']} | "
        f"SUPPORTED: {summary['supported']} | "
        f"PARTIALLY SUPPORTED: {summary['partially_supported']} | "
        f"NOT SUPPORTED: {summary['not_supported']} | "
        f"Invalid citations: {summary['claims_with_invalid_citations']} | "
        f"Missing citations: {summary['claims_without_citations']}"
    )
    lines.append("")

    for record in records:
        lines.append(f"Claim {record['claim_number']}: {record['status']}")

        if record["valid_citations"]:
            sources_text = "; ".join(
                f"Evidence {s['evidence_number']} "
                f"({s['source']}, page {s['page']}, {s['chunk_id']})"
                for s in record["sources"]
            )
            lines.append(f"  Valid citations: {sources_text}")
        else:
            lines.append("  Valid citations: NONE")

        if record["invalid_citations"]:
            lines.append(
                f"  INVALID citations (evidence does not exist): "
                f"{record['invalid_citations']}"
            )

        if record["status"].startswith("NOT SUPPORTED"):
            lines.append(
                "  WARNING: unsupported claim - must not be "
                "presented as a finding."
            )

        lines.append("")

    return {
        "claims": records,
        "summary": summary,
        "text": "\n".join(lines)
    }


def validate_research_citations(verification_results, evidence):
    """Full pipeline: parse → validate → report.

    Returns the structured citation report dict.
    """
    records = parse_verification_results(verification_results)

    records = validate_citations(records, evidence)

    return build_citation_report(records)
