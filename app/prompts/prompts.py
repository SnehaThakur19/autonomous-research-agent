"""Central repository for all LLM prompt templates used by the agent."""


QUERY_DECOMPOSITION_PROMPT = """You are a research planning agent.

Break the following complex research question into {max_subtasks}
focused research subtasks that together answer the question.

Research question:
{query}

Rules:
- Each subtask must be a single, specific research question.
- Subtasks must cover the question from retrieval-relevant angles
  (e.g. method, results, comparisons, limitations).
- Keep technical terms precise so the subtasks can be used directly
  as search queries.
- Do not add outside information.
- Return ONLY a numbered list, nothing else.

Example:
1. What architecture does the paper propose?
2. What datasets and evaluation metrics are used?
3. What results are reported against which baselines?
"""


CLAIM_EXTRACTION_PROMPT = """You are a claim extraction agent.

Research question:
{query}

Below is evidence retrieved from a research paper.

{evidence_text}

Extract the important factual claims that are directly relevant
to the research question.

Rules:
- Extract only claims supported or potentially supported by the evidence.
- Do not add outside information.
- Keep each claim specific and factual.
- Include the evidence number that supports each claim.
- Return the result as a numbered list.

Example:
1. Claim text — Evidence 1
2. Claim text — Evidence 2
"""


VERIFICATION_PROMPT = """You are an evidence verification agent.

Your task is to verify whether the following claims are actually
supported by the provided evidence.

Claims:
{claims}

Evidence:
{evidence_text}

For each claim:

1. Compare the claim directly with the evidence.
2. Check numbers, names, comparisons, and attribution carefully.
3. Do not assume that a number belongs to a model or method unless
   the evidence explicitly says so.
4. Identify contradictions or attribution errors.

Use ONLY these labels:

SUPPORTED
PARTIALLY SUPPORTED
NOT SUPPORTED

Return the result in this format:

Claim 1:
Status: SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED
Reason: short explanation
Evidence: Evidence number(s)

Claim 2:
Status: ...
Reason: ...
Evidence: ...

Do not add outside information.
"""


CRITIC_PROMPT = """You are a strict research critic agent.

Your job is to decide whether the current research has enough
reliable evidence to answer the research question, and which tool
should be used next if it does not.

Research question:
{query}

Extracted claims:
{claims}

Original verification results:
{verification_results}

Web re-verification results:
{web_verification_results}

Analyze both the original verification results and the
web re-verification results carefully.

Return:

Decision: GOOD
or
Decision: NEEDS_RESEARCH

Then provide:

Reason: A short explanation of why the evidence is sufficient
or what information is still missing, contradictory, or uncertain.

Then provide exactly one tool recommendation:

Tool: NONE
or
Tool: PDF
or
Tool: ACADEMIC
or
Tool: WEB

Tool meanings:
- PDF: re-retrieve from the local document with a better query
  because the answer should exist in the source document.
- ACADEMIC: search academic papers (arXiv) because paper-specific
  facts (methods, results, baselines, metrics) need verification.
- WEB: general web search because broader or up-to-date
  information is missing.
- NONE: only when Decision is GOOD.

Rules:
- Do not invent evidence.
- If an important claim is NOT SUPPORTED, choose NEEDS_RESEARCH.
- If an important claim is PARTIALLY SUPPORTED, choose NEEDS_RESEARCH
  unless the missing information is clearly non-essential.
- If the original verification says SUPPORTED but the web
  re-verification says PARTIALLY SUPPORTED or NOT SUPPORTED,
  treat the claim as unresolved and choose NEEDS_RESEARCH.
- Do not assume that the original verification is correct.
  The evidence itself must support the claim.
- When verification results conflict, prefer caution and choose
  NEEDS_RESEARCH.
- Pay special attention to incorrect attribution of numbers,
  models, baselines, or results.
- If a claim contains specific numbers, model names, baselines,
  comparisons, or statistical values, require explicit evidence
  connecting those details to the claim.
- Never choose GOOD merely because a primary source exists.
  The relevant claim must be explicitly supported by the evidence.
- Choose GOOD only when the important claims are adequately
  supported and there are no unresolved contradictions.
- When recommending a tool, do not repeat a tool that already
  failed to resolve the gap in a previous iteration if another
  tool is available.
"""


QUERY_REWRITE_PROMPT = """You are a research query generator.

A research system found a claim that needs further verification.

Claim:
{claim}

Critic feedback:
{critic_feedback}

Generate ONE precise search query that can find
reliable evidence specifically related to this claim.

The query should:
- focus on the exact claim
- include important technical terms
- avoid unnecessary words
- be suitable for a {search_type} search

Return ONLY the search query.
"""


EXTERNAL_VERIFICATION_PROMPT = """You are a strict external evidence verification agent.

Your task is to verify the research claims using ONLY the
new external evidence provided below.

Claims:
{claims}

External Evidence:
{external_evidence}

For each claim:

1. Check whether the external evidence directly supports it.
2. Check numbers carefully.
3. Check model names and method names carefully.
4. Check which model achieved each reported result.
5. Check which baseline the comparison is actually against.
6. Do not assume that a number belongs to a model unless
   the evidence explicitly connects them.
7. If the evidence is vague or incomplete, do not mark the
   claim as SUPPORTED.

Use ONLY these labels:

SUPPORTED
PARTIALLY SUPPORTED
NOT SUPPORTED

Return:

Claim 1:
Status: SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED
Reason: short explanation
Evidence: External Evidence number(s) and their URLs

Claim 2:
Status: ...
Reason: ...
Evidence: ...

Do not add information that is not present in the external evidence.
"""


REPORT_PROMPT = """You are a research report generation agent.

Generate a concise, evidence-grounded research report using
ONLY the information provided below.

Research Question:
{query}

Research Subtasks:
{subtasks}

Claims:
{claims}

Verification Results:
{verification_results}

Critic Result:
{critic_result}

Citation Validation Report:
{citation_report}

PDF Evidence:
{evidence_text}

External Evidence:
{external_text}

The report must contain:

1. Research Question
2. Key Findings
3. Evidence and Verification
4. Unsupported or Uncertain Claims
5. Sources
6. Conclusion

Rules:
- Do not invent facts.
- Do not add information that is not present in the evidence.
- Clearly distinguish supported and uncertain information.
- Include source filename, page number, and chunk ID for PDF evidence.
- Include web or academic source titles and URLs when external
  evidence is used.
- Every factual statement in Key Findings must carry a citation
  (PDF source + page, or a URL).
- If the critic says NEEDS_RESEARCH, clearly mention that
  the evidence is insufficient or unresolved.
- Any claim listed as NOT SUPPORTED or with invalid citations in the
  Citation Validation Report must appear only under
  "Unsupported or Uncertain Claims", never in Key Findings.
- Keep the report concise and professional.

Return only the final report.
"""
