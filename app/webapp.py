"""Multi-step wizard UI for the autonomous research agent.

Steps: Upload → Question → Plan → Research → Report

Run (from the project root):
    venv\\Scripts\\python.exe -m streamlit run app\\webapp.py

Then open http://localhost:8501
"""

import glob
import html as html_lib
import os
import time

import streamlit as st

# ---------------------------------------------------------------------------
# Secrets: Streamlit Community Cloud stores keys in
# .streamlit/secrets.toml (exposed via st.secrets); locally they live in
# .env (loaded by python-dotenv inside each module). Bridge secrets into
# the environment BEFORE importing the agent so os.getenv finds them.
# ---------------------------------------------------------------------------
try:
    for _key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    # No secrets.toml present (e.g. local run with .env) — ignore
    pass

from app.graph.workflow import research_graph


st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🔬",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Theme / styling — dark navy with restrained purple accents
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1020;
        --card: rgba(255,255,255,0.04);
        --border: rgba(255,255,255,0.09);
        --accent: #7c3aed;
        --accent-soft: rgba(124,58,237,0.16);
        --text: #e7e9ee;
        --muted: #9aa3b5;
        --green: #22c55e;
        --amber: #f59e0b;
        --red: #ef4444;
    }
    .stApp {
        background:
            radial-gradient(900px 420px at 85% -10%,
                            rgba(124,58,237,0.18), transparent 60%),
            var(--bg);
        color: var(--text);
    }
    .block-container { max-width: 980px; padding-top: 2rem; }

    h1, h2, h3 { letter-spacing: -0.01em; }

    .wizard-header {
        display: flex; align-items: center; gap: .7rem;
        margin-bottom: .4rem;
    }
    .wizard-header .logo {
        font-size: 1.6rem;
        background: var(--accent-soft);
        border: 1px solid rgba(124,58,237,.35);
        border-radius: 12px; padding: .35rem .6rem;
    }
    .wizard-header h1 { font-size: 1.45rem; margin: 0; }
    .wizard-header p  { margin: 0; color: var(--muted); font-size: .9rem; }

    /* ---- Stepper ---- */
    .stepper { display: flex; align-items: flex-start; margin: .8rem 0 1.6rem; }
    .stp { flex: 1; text-align: center; position: relative; }
    .stp .dot {
        width: 34px; height: 34px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        background: var(--card); border: 1.5px solid var(--border);
        color: var(--muted); font-size: .85rem; font-weight: 600;
        position: relative; z-index: 2;
    }
    .stp.active .dot {
        background: var(--accent); border-color: var(--accent);
        color: white; box-shadow: 0 0 0 4px rgba(124,58,237,.25);
    }
    .stp.done .dot {
        background: rgba(34,197,94,.15); border-color: var(--green);
        color: var(--green);
    }
    .stp .lbl {
        display: block; margin-top: .35rem; font-size: .78rem;
        color: var(--muted);
    }
    .stp.active .lbl { color: var(--text); font-weight: 600; }
    .stp.done .lbl { color: #cbd5e1; }
    .stp::before {
        content: ""; position: absolute; top: 17px; left: -50%;
        width: 100%; height: 1.5px; background: var(--border); z-index: 1;
    }
    .stp:first-child::before { display: none; }
    .stp.done::before, .stp.active::before { background: var(--accent); }

    /* ---- Cards ---- */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin: .5rem 0 1rem 0;
    }
    .card h3 { margin: 0 0 .3rem 0; font-size: 1.05rem; }
    .card .sub { color: var(--muted); font-size: .87rem; margin: 0; }

    .row {
        display: flex; align-items: center; justify-content: space-between;
        gap: .8rem; padding: .55rem .8rem; border-radius: 10px;
        border: 1px solid var(--border); background: var(--card);
        margin: .35rem 0;
    }
    .row .name { font-weight: 500; font-size: .92rem; }
    .row .meta { color: var(--muted); font-size: .8rem; }

    .badge {
        display: inline-block; padding: .12rem .55rem;
        border-radius: 999px; font-size: .72rem; font-weight: 600;
        border: 1px solid transparent; white-space: nowrap;
    }
    .b-done    { background: rgba(34,197,94,.12);  color: var(--green); border-color: rgba(34,197,94,.35); }
    .b-prog    { background: rgba(124,58,237,.14); color: #c4b5fd; border-color: rgba(124,58,237,.4); }
    .b-pend    { background: rgba(255,255,255,.05); color: var(--muted); border-color: var(--border); }
    .b-fail    { background: rgba(239,68,68,.12);  color: #fca5a5; border-color: rgba(239,68,68,.4); }
    .b-warn    { background: rgba(245,158,11,.12); color: #fcd34d; border-color: rgba(245,158,11,.4); }

    .ev {
        border-left: 3px solid var(--border);
        padding: .55rem .9rem; border-radius: 8px;
        background: var(--card); margin: .4rem 0;
    }
    .ev.done { border-left-color: var(--green); }
    .ev.fail { border-left-color: var(--red); }
    .ev.tool { border-left-color: var(--amber); }
    .ev.crit { border-left-color: #a855f7; }
    .ev .t { font-weight: 600; font-size: .92rem; }
    .ev .d { color: var(--muted); font-size: .84rem; margin-top: .15rem; }

    .spinner-dot {
        display: inline-block; width: 8px; height: 8px;
        margin-right: .3rem; border-radius: 50%;
        background: var(--accent); animation: pulse 1s infinite ease-in-out;
    }
    .spinner-dot:nth-child(2) { animation-delay: .2s; }
    .spinner-dot:nth-child(3) { animation-delay: .4s; }
    @keyframes pulse {
        0%, 100% { opacity: .25; transform: scale(.85); }
        50%      { opacity: 1;  transform: scale(1); }
    }

    .section-title {
        font-size: .8rem; text-transform: uppercase;
        letter-spacing: .08em; color: var(--muted);
        margin: 1.2rem 0 .4rem 0;
    }

    /* Streamlit button tuning */
    .stButton > button {
        border-radius: 10px; border: 1px solid var(--border);
        background: var(--card); color: var(--text);
        transition: all .15s ease;
    }
    .stButton > button:hover {
        border-color: rgba(124,58,237,.6); color: white;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--accent); border-color: var(--accent);
        font-weight: 600;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #6d28d9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEP_NAMES = ["Upload Papers", "Question", "Plan", "Research", "Report"]
STEP_ICONS = ["📤", "💬", "🧠", "⚡", "📄"]

EXAMPLE_QUESTIONS = [
    "What architecture is proposed in the paper and what results does it achieve?",
    "What datasets and evaluation metrics are used?",
    "What are the limitations and future work mentioned?",
    "Compare the proposed method with its baselines.",
]

# Expected pipeline order used only to show *pending* placeholders.
# Events are only marked Completed when the backend actually emits them.
PIPELINE_ORDER = [
    "plan", "retrieve", "extract_claims", "verify",
    "validate_citations", "critic",
]

NODE_TITLES = {
    "plan": ("🧠", "Query decomposition"),
    "retrieve": ("📚", "PDF retrieval + CrossEncoder reranking"),
    "extract_claims": ("📝", "Claim extraction"),
    "verify": ("✅", "Claim-to-evidence verification"),
    "validate_citations": ("🔎", "Citation validation"),
    "critic": ("🧠", "Evidence-sufficiency critic"),
    "increment_iteration": ("🔄", "Self-correction iteration"),
    "pdf_re_research": ("📄", "Tool switch → PDF re-retrieval"),
    "web_research": ("🌐", "Tool switch → Web search (Tavily)"),
    "academic_research": ("🎓", "Tool switch → Academic search (arXiv)"),
    "web_reverify": ("🔁", "External re-verification"),
    "report": ("📄", "Report generation"),
}

TOOL_NODES = {"pdf_re_research", "web_research", "academic_research"}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
ss = st.session_state

defaults = {
    "wiz_step": 1,
    "max_step": 1,
    "papers": {},            # path -> {name, size, status}
    "question": "",
    "plan": None,
    "plan_for": None,        # question the plan was built for
    "events": [],            # timeline of real backend events
    "result": None,          # accumulated graph state
    "research_done": False,
    "research_error": None,
    "elapsed": 0.0,
}
for key, value in defaults.items():
    if key not in ss:
        ss[key] = value


def fmt_size(num_bytes):
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.1f} MB"
    return f"{max(num_bytes // 1024, 1)} KB"


def esc(text):
    """Escape dynamic text before embedding it in HTML."""
    return html_lib.escape(str(text))


def goto(step):
    ss.wiz_step = step
    ss.max_step = max(ss.max_step, step)
    st.rerun()


# ---------------------------------------------------------------------------
# Header + stepper
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="wizard-header">
      <div class="logo">🔬</div>
      <div>
        <h1>Autonomous Research Agent</h1>
        <p>Upload papers → ask a question → watch the agent research,
           verify, and report — step by step.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


def render_stepper(active):
    cols_html = []
    for i, (icon, name) in enumerate(zip(STEP_ICONS, STEP_NAMES), start=1):
        if i == active:
            cls, dot = "active", str(i)
        elif i < active or (i < ss.max_step and i != active):
            cls, dot = "done", "✓"
        else:
            cls, dot = "", str(i)
        cols_html.append(
            f"<div class='stp {cls}'><div class='dot'>{dot}</div>"
            f"<span class='lbl'>{icon} {name}</span></div>"
        )
    st.markdown(
        f"<div class='stepper'>{''.join(cols_html)}</div>",
        unsafe_allow_html=True
    )

    # Clickable step navigation (completed steps only — no skipping)
    nav_cols = st.columns(len(STEP_NAMES))
    for i, col in enumerate(nav_cols, start=1):
        with col:
            reachable = i <= ss.max_step
            if st.button(
                STEP_NAMES[i - 1],
                key=f"nav_{i}",
                disabled=not reachable or i == active,
                use_container_width=True
            ):
                goto(i)


render_stepper(ss.wiz_step)
st.markdown("---")

# ===========================================================================
# STEP 1 — Upload research papers
# ===========================================================================
if ss.wiz_step == 1:
    st.subheader("📤 Step 1 · Upload research papers")
    st.caption(
        "Upload one or more PDF research papers. They are chunked, "
        "embedded (MiniLM), and indexed into the local vector store."
    )

    # Register any PDFs already present in data/
    for path in glob.glob("data/*.pdf"):
        if path not in ss.papers:
            ss.papers[path] = {
                "name": os.path.basename(path),
                "size": os.path.getsize(path),
                "status": "ready",
            }

    uploaded = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        from app.ingest import ingest_pdfs

        for uf in uploaded:
            save_path = os.path.join("data", uf.name)
            if save_path in ss.papers and ss.papers[save_path]["status"] == "ready":
                continue

            os.makedirs("data", exist_ok=True)
            with open(save_path, "wb") as handle:
                handle.write(uf.getbuffer())

            ss.papers[save_path] = {
                "name": uf.name,
                "size": os.path.getsize(save_path),
                "status": "processing",
            }

            try:
                ingest_pdfs([save_path], rebuild=False)
                ss.papers[save_path]["status"] = "ready"
            except Exception as exc:
                ss.papers[save_path]["status"] = "failed"
                ss.papers[save_path]["error"] = str(exc)

    # Paper list
    if ss.papers:
        st.markdown("<div class='section-title'>Your papers</div>",
                    unsafe_allow_html=True)

        for path in list(ss.papers):
            info = ss.papers[path]

            if info["status"] == "ready":
                badge = "<span class='badge b-done'>✓ Indexed</span>"
            elif info["status"] == "processing":
                badge = "<span class='badge b-prog'>⏳ Processing…</span>"
            else:
                badge = "<span class='badge b-fail'>✕ Failed</span>"

            c1, c2, c3 = st.columns([6, 2, 1])
            with c1:
                st.markdown(
                    f"<div class='row'><div>"
                    f"<span class='name'>📄 {esc(info['name'])}</span><br>"
                    f"<span class='meta'>{fmt_size(info['size'])}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(badge, unsafe_allow_html=True)
            with c3:
                if st.button("🗑️", key=f"rm_{path}",
                             help="Remove paper and re-index"):
                    os.remove(path)
                    del ss.papers[path]

                    # Rebuild the store from the remaining papers so
                    # removed content can no longer be retrieved
                    from app.ingest import ingest_pdfs
                    remaining = [p for p, i in ss.papers.items()
                                 if i["status"] == "ready"]
                    if remaining:
                        ingest_pdfs(remaining, rebuild=True)
                    else:
                        from app.rag.vectorstore import create_vector_store
                        create_vector_store()
                    st.rerun()

            if info["status"] == "failed":
                st.error(f"Ingestion failed for {info['name']}: "
                         f"{info.get('error', 'unknown error')}")

    ready = any(i["status"] == "ready" for i in ss.papers.values())
    processing = any(i["status"] == "processing" for i in ss.papers.values())

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c2:
        if st.button("Continue →", type="primary",
                     disabled=(not ready or processing),
                     use_container_width=True):
            goto(2)

    if processing:
        st.info("⏳ Papers are still being indexed — Continue unlocks when done.")
    elif not ready:
        st.warning(
            "Upload at least one PDF, or tick below to research using "
            "web + academic search only."
        )

    ss.skip_papers = st.checkbox(
        "Continue without local papers (web/arXiv research only)",
        value=ss.get("skip_papers", False)
    )
    if ss.skip_papers and not processing:
        if st.button("Continue without papers →", type="primary"):
            goto(2)

# ===========================================================================
# STEP 2 — Research question
# ===========================================================================
elif ss.wiz_step == 2:
    st.subheader("💬 Step 2 · Research question")
    st.caption("Ask anything about your uploaded papers.")

    st.markdown("<div class='section-title'>Try an example</div>",
                unsafe_allow_html=True)
    ex_cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        with ex_cols[i % 2]:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                ss.question = example
                ss["question_box"] = example
                st.rerun()

    if "question_box" not in ss:
        ss["question_box"] = ss.question

    question = st.text_area(
        "Your research question",
        height=110,
        key="question_box",
        placeholder="e.g. What architecture is proposed and what results does it achieve?"
    )
    ss.question = question

    st.markdown("---")
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("← Back", use_container_width=True):
            goto(1)
    with b2:
        valid = bool(ss.question.strip())
        if st.button("Continue →", type="primary",
                     disabled=not valid, use_container_width=True):
            goto(3)

    if not valid:
        st.caption("Enter a question to continue.")

# ===========================================================================
# STEP 3 — Research plan
# ===========================================================================
elif ss.wiz_step == 3:
    st.subheader("🧠 Step 3 · Research plan")
    st.caption(
        "The agent decomposes your question into focused research "
        "subtasks. These drive per-task retrieval in the pipeline."
    )

    st.markdown("<div class='section-title'>Your question</div>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div class='card'><p class='sub'>{esc(ss.question)}</p></div>",
        unsafe_allow_html=True
    )

    # Generate the plan once per question (real backend call)
    if ss.plan is None or ss.plan_for != ss.question:
        with st.status("🧠 Agent is planning…", expanded=False) as status:
            from app.research.task_decomposer import decompose_query
            try:
                ss.plan = decompose_query(ss.question)
                ss.plan_for = ss.question
                status.update(label="✅ Plan ready", state="complete")
            except Exception as exc:
                status.update(label="✕ Planning failed", state="error")
                st.error(f"Planning failed: {exc}")
                st.stop()

    st.markdown("<div class='section-title'>Research subtasks</div>",
                unsafe_allow_html=True)
    for i, subtask in enumerate(ss.plan, start=1):
        st.markdown(
            f"<div class='row'><div>"
            f"<span class='badge b-prog'>{i}</span> "
            f"<span class='name'>{esc(subtask)}</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button("← Back", use_container_width=True):
            goto(2)
    with b2:
        if st.button("↻ Regenerate plan", use_container_width=True):
            ss.plan = None
            st.rerun()
    with b3:
        if st.button("🚀 Start research", type="primary",
                     use_container_width=True):
            ss.events = []
            ss.result = None
            ss.research_done = False
            ss.research_error = None
            goto(4)

# ===========================================================================
# STEP 4 — Research in progress (live backend events)
# ===========================================================================
elif ss.wiz_step == 4:
    st.subheader("⚡ Step 4 · Research in progress")
    st.caption(
        "Every event below is emitted live by the LangGraph workflow "
        "as the backend actually executes it."
    )

    if not ss.research_done and ss.research_error is None:
        # ---------------- Run the real graph, streaming events ----------------
        max_iterations = ss.get("max_iter", 3)

        initial_state = {
            "query": ss.question,
            "subtasks": ss.plan or [],
            "iteration": 0,
            "max_iterations": max_iterations,
        }
        acc = dict(initial_state)

        start = time.time()

        st.markdown(
            "<div class='card'><span class='spinner-dot'></span>"
            "<span class='spinner-dot'></span>"
            "<span class='spinner-dot'></span> "
            "<b>The agent is researching…</b> "
            "<span class='sub'>each stage appears below as the "
            "backend completes it</span></div>",
            unsafe_allow_html=True
        )

        timeline_box = st.empty()

        def render_timeline(extra_html=""):
            """Re-render the full event timeline."""
            blocks = []
            emitted = {e["node"] for e in ss.events}
            for event in ss.events:
                cls = "fail" if event["status"] == "failed" else (
                    "tool" if event["node"] in TOOL_NODES else (
                        "crit" if event["node"] == "critic" else "done"
                    )
                )
                badge_cls = "b-fail" if event["status"] == "failed" else "b-done"
                badge_txt = "✕ Failed" if event["status"] == "failed" else "✓ Completed"
                blocks.append(
                    f"<div class='ev {cls}'>"
                    f"<span class='t'>{event['icon']} {esc(event['title'])}</span> "
                    f"<span class='badge {badge_cls}'>{badge_txt}</span>"
                    f"<div class='d'>{esc(event['detail'])}</div></div>"
                )
            # Honest pending placeholders for pipeline stages not yet emitted
            for node in PIPELINE_ORDER:
                if node not in emitted and node != "plan":
                    icon, title = NODE_TITLES[node]
                    blocks.append(
                        f"<div class='ev'><span class='t'>{icon} {title}</span> "
                        f"<span class='badge b-pend'>Pending</span></div>"
                    )
            timeline_box.markdown("".join(blocks) + extra_html,
                                  unsafe_allow_html=True)

        def make_event(node, detail):
            icon, title = NODE_TITLES.get(node, ("🔹", node))
            return {
                "node": node,
                "icon": icon,
                "title": title,
                "detail": detail,
                "status": "completed",
            }

        try:
            for chunk in research_graph.stream(initial_state):
                for node_name, update in chunk.items():
                    acc.update(update)

                    # --- Build an honest detail line from the real update ---
                    detail = ""
                    exp_title = None
                    exp_body = None

                    if node_name == "plan":
                        detail = (
                            "Plan generated in Step 3 reused "
                            f"({len(acc.get('subtasks', []))} subtasks)"
                        )
                    elif node_name == "retrieve":
                        ev = acc.get("evidence", [])
                        detail = f"{len(ev)} evidence chunks selected and reranked"
                        exp_title = "Retrieved evidence snippets"
                        exp_body = "\n\n".join(
                            f"**Evidence {i}** — `{e['metadata'].get('source')}`, "
                            f"page {e['metadata'].get('page')}, "
                            f"`{e['metadata'].get('chunk_id')}` "
                            f"(rerank score: {e.get('rerank_score', 0):.4f})\n\n"
                            f"> {e['text'][:350]}…"
                            for i, e in enumerate(ev, start=1)
                        )
                    elif node_name == "extract_claims":
                        detail = "Factual claims extracted from evidence"
                        exp_title = "Extracted claims"
                        exp_body = acc.get("claims", "")
                    elif node_name == "verify":
                        detail = "Each claim checked against the evidence"
                        exp_title = "Verification results"
                        exp_body = acc.get("verification_results", "")
                    elif node_name == "validate_citations":
                        summ = acc.get("citation_report", {}).get("summary", {})
                        detail = (
                            f"{summ.get('supported', 0)} supported · "
                            f"{summ.get('not_supported', 0)} unsupported · "
                            f"{summ.get('claims_with_invalid_citations', 0)} "
                            f"citation errors"
                        )
                        exp_title = "Citation validation report"
                        exp_body = acc.get("citation_report", {}).get("text", "")
                    elif node_name == "critic":
                        critic = acc.get("critic_result", "")
                        if "NEEDS_RESEARCH" in critic:
                            tool = acc.get("selected_tool", "")
                            detail = (
                                "Evidence insufficient — self-correction "
                                f"triggered (next tool: {tool or 'WEB'})"
                            )
                        else:
                            detail = "Evidence is sufficient — no more research needed"
                        exp_title = "Critic reasoning"
                        exp_body = critic
                    elif node_name == "increment_iteration":
                        detail = (
                            f"Iteration {acc.get('iteration', 0)} of "
                            f"{acc.get('max_iterations', 3)}"
                        )
                    elif node_name == "pdf_re_research":
                        detail = (
                            f"Rewritten local query: "
                            f"{acc.get('research_query', '')[:110]}"
                        )
                    elif node_name in ("web_research", "academic_research"):
                        results = acc.get("web_results", [])
                        detail = (
                            f"Query: {acc.get('research_query', '')[:90]} · "
                            f"{len(results)} external sources retrieved"
                        )
                        exp_title = "External sources retrieved"
                        exp_body = "\n\n".join(
                            f"**{r.get('title')}** — {r.get('url')}\n\n"
                            f"> {(r.get('content') or '')[:250]}…"
                            for r in results
                        )
                    elif node_name == "web_reverify":
                        detail = "Claims re-checked against external evidence"
                        exp_title = "External re-verification results"
                        exp_body = acc.get("web_verification_results", "")
                    elif node_name == "report":
                        detail = "Final evidence-grounded report generated ✓"

                    event = make_event(node_name, detail)
                    if exp_body:
                        event["exp_title"] = exp_title
                        event["exp_body"] = exp_body
                    ss.events.append(event)

                    render_timeline()

        except Exception as exc:
            fail_event = {
                "node": "error",
                "icon": "⚠️",
                "title": "Pipeline error",
                "detail": str(exc),
                "status": "failed",
            }
            ss.events.append(fail_event)
            ss.research_error = str(exc)
            render_timeline()

        ss.elapsed = time.time() - start

        if ss.research_error is None:
            ss.result = acc
            ss.research_done = True
            render_timeline()

    # ---------------- Completed state ----------------
    if ss.research_error:
        st.error(f"Research failed: {ss.research_error}")
        if st.button("↻ Retry research", type="primary"):
            ss.events = []
            ss.research_error = None
            ss.research_done = False
            st.rerun()
        if st.button("← Back to plan"):
            goto(3)

    elif ss.research_done:
        st.success(
            f"✅ Research complete in {ss.elapsed:.1f}s — "
            f"{len(ss.events)} pipeline events executed."
        )

        # Timeline with expandable detail
        for event in ss.events:
            badge = "<span class='badge b-done'>✓ Completed</span>"
            st.markdown(
                f"<div class='ev done'>"
                f"<span class='t'>{event['icon']} {esc(event['title'])}</span> {badge}"
                f"<div class='d'>{esc(event['detail'])}</div></div>",
                unsafe_allow_html=True
            )
            if event.get("exp_body"):
                with st.expander(f"View details — {event['title']}"):
                    st.markdown(event["exp_body"])

        st.markdown("---")
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("← Back to plan"):
                goto(3)
        with b2:
            if st.button("View report →", type="primary"):
                goto(5)

# ===========================================================================
# STEP 5 — Research report
# ===========================================================================
elif ss.wiz_step == 5:
    st.subheader("📄 Step 5 · Research report")

    result = ss.result or {}
    report = result.get("final_report", "")
    citation_report = result.get("citation_report", {})
    summary = citation_report.get("summary", {})
    critic = result.get("critic_result", "")

    if not report:
        st.warning("No report available — run the research first.")
        if st.button("← Go to research", type="primary"):
            goto(4)
        st.stop()

    # ---- Actions ----
    a1, a2, a3, a4 = st.columns([2, 2, 2, 2])
    with a1:
        if hasattr(st, "clipboard_button"):
            st.clipboard_button("📋 Copy report", text=report,
                                use_container_width=True)
        else:
            st.download_button("📋 Copy (download)", data=report,
                               file_name="report.md", use_container_width=True)
    with a2:
        st.download_button(
            "⬇️ Download .md", data=report,
            file_name="research_report.md", mime="text/markdown",
            use_container_width=True
        )
    with a3:
        if st.button("↻ New research", use_container_width=True):
            ss.question = ""
            ss.plan = None
            ss.plan_for = None
            ss.events = []
            ss.result = None
            ss.research_done = False
            ss.research_error = None
            goto(2)
    with a4:
        if st.button("🔄 Run again", use_container_width=True):
            ss.events = []
            ss.result = None
            ss.research_done = False
            goto(4)

    st.markdown("---")

    col_main, col_side = st.columns([3, 1])

    with col_main:
        # ---- Report body (LLM output with its native sections) ----
        st.markdown(report)

        # ---- Claim verification results (structured, from backend) ----
        st.markdown("<div class='section-title'>Claim verification results</div>",
                    unsafe_allow_html=True)
        claims = citation_report.get("claims", [])
        if claims:
            rows = [{
                "Claim #": c["claim_number"],
                "Status": c["status"],
                "Cited evidence": ", ".join(
                    f"E{n}" for n in c["valid_citations"]
                ) or "—",
                "Invalid citations": ", ".join(
                    str(n) for n in c["invalid_citations"]
                ) or "—",
            } for c in claims]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No structured verification records available.")

        # ---- Unsupported / uncertain claims ----
        st.markdown(
            "<div class='section-title'>Unsupported or uncertain claims</div>",
            unsafe_allow_html=True
        )
        flagged = [
            c for c in claims
            if c["status"].startswith("NOT SUPPORTED")
            or c["status"].startswith("PARTIALLY")
            or c["invalid_citations"]
        ]
        if flagged:
            for c in flagged:
                st.markdown(
                    f"- **Claim {c['claim_number']}** — {c['status']}"
                    + (f" (invalid cites: {c['invalid_citations']})"
                       if c["invalid_citations"] else "")
                    + (f"\n  - Reason: {c['reason']}" if c.get("reason") else "")
                )
        else:
            st.success("None — every claim is supported with valid citations.")

        # ---- Evidence gaps ----
        st.markdown("<div class='section-title'>Evidence gaps</div>",
                    unsafe_allow_html=True)
        if "NEEDS_RESEARCH" in critic:
            st.warning(
                "The critic flagged remaining gaps after "
                f"{result.get('iteration', 0)} correction iterations. "
                "See critic reasoning in the sidebar diagnostics."
            )
        else:
            st.info("None — the critic confirmed sufficient evidence.")

        # ---- Sources & references ----
        st.markdown("<div class='section-title'>Sources & references</div>",
                    unsafe_allow_html=True)

        pdf_sources = {}
        for c in claims:
            for s in c.get("sources", []):
                key = (s.get("source"), s.get("page"), s.get("chunk_id"))
                pdf_sources[key] = True
        for (source, page, chunk_id) in pdf_sources:
            st.markdown(
                f"- 📄 `{source}` — page **{page}**, chunk `{chunk_id}`"
            )

        for i, r in enumerate(result.get("web_results", []), start=1):
            st.markdown(f"- 🌐 [{r.get('title')}]({r.get('url')})")

        if not pdf_sources and not result.get("web_results"):
            st.info("No sources recorded.")

    with col_side:
        st.markdown("<div class='section-title'>Run summary</div>",
                    unsafe_allow_html=True)

        st.metric("Iterations", result.get("iteration", 0))
        st.metric("Subtasks", len(result.get("subtasks", [])))
        st.metric("Claims validated", summary.get("total_claims", 0))
        st.metric("✅ Supported", summary.get("supported", 0))
        st.metric("❌ Unsupported", summary.get("not_supported", 0))
        st.metric("Citation errors",
                  summary.get("claims_with_invalid_citations", 0))
        st.metric("Time", f"{ss.elapsed:.1f}s")

        with st.expander("🔎 Citation validation report"):
            st.text(citation_report.get("text", "—"))

        with st.expander("🧠 Critic reasoning"):
            st.markdown(critic or "—")

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='section-title'>⚙️ Settings</div>",
                unsafe_allow_html=True)
    ss["max_iter"] = st.slider(
        "Max self-correction iterations",
        min_value=1, max_value=5,
        value=ss.get("max_iter", 3)
    )
    st.caption(
        "Self-correction lets the critic retry retrieval with the "
        "PDF, arXiv, or web search when evidence is insufficient."
    )
    st.markdown("---")
    st.markdown("<div class='section-title'>Pipeline</div>",
                unsafe_allow_html=True)
    st.caption(
        "Plan → PDF retrieval → CrossEncoder reranking → claim "
        "extraction → verification → citation validation → critic → "
        "self-correction (PDF / arXiv / Web) → report"
    )
