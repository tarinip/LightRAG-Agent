"""
Streamlit Interactive Benchmark: Standard LightRAG vs AgenticRAG
=================================================================
Upload a PDF, enter questions, run Standard RAG or AgenticRAG
(or both), and see RAGAS evaluation metrics side by side.

Run:  streamlit run examples/streamlit_benchmark.py

Requirements (beyond LightRAG):
    pip install streamlit pymupdf ragas langchain-google-genai
"""
import os
import sys

# Ensure the project root is on sys.path so lightrag is importable
# even when running via `streamlit run examples/streamlit_benchmark.py`
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import re
import asyncio
import time
import json
import csv
import io
import hashlib
import threading
import numpy as np
import fitz
import pandas as pd
import streamlit as st

from lightrag import LightRAG, QueryParam
from lightrag.agentic_rag import AgenticRAG
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

# RAGAS imports
from ragas import EvaluationDataset, evaluate as ragas_evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics import ContextRelevance
from ragas.metrics import ResponseGroundedness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# =====================================================================
# Constants
# =====================================================================

FAIL_PHRASES = [
    "i do not have enough information",
    "i don't have enough information",
    "not enough information to answer",
    "cannot answer",
    "unable to answer",
    "no relevant information",
    "sorry, i",
    "i cannot find",
    "no information available",
]

# =====================================================================
# Streamlit page config & session state (must come before @st.cache_resource)
# =====================================================================

st.set_page_config(page_title="LightRAG Benchmark", layout="wide")

_defaults = {
    "phase": "idle",
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "working_dir": None,
    "rag": None,
    "pdf_name": None,
    "pdf_text": None,
    "questions": [],
    "results": [],
    "running": False,
    "enable_ragas": True,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# =====================================================================
# Helpers
# =====================================================================


@st.cache_resource
def _get_event_loop():
    """Create a single persistent background event loop for all async operations."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop


def run_async(coro):
    """Run an async coroutine in the persistent background event loop."""
    loop = _get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def read_pdf_from_upload(uploaded_file) -> str:
    """Extract text from a Streamlit UploadedFile (PDF)."""
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text


def compute_pdf_hash(uploaded_file) -> str:
    """Return first 12 chars of MD5 hex digest for working dir uniqueness."""
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(pdf_bytes).hexdigest()[:12]


def is_good_answer(text: str) -> bool:
    if not text or not text.strip():
        return False
    lower = text.lower().strip()
    for phrase in FAIL_PHRASES:
        if phrase in lower:
            return False
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(sentences) >= 5


async def run_single_query(label, coro):
    start = time.time()
    try:
        result = await coro
        if hasattr(result, "__aiter__"):
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
            result = "".join(chunks)
        elapsed = time.time() - start
        text = str(result).strip()
        success = is_good_answer(text)
        return success, text, round(elapsed, 2)
    except Exception as e:
        elapsed = time.time() - start
        return False, f"ERROR: {e}", round(elapsed, 2)


async def get_retrieval_contexts(rag, query):
    """Retrieve text chunks for RAGAS evaluation."""
    try:
        result = await rag.aquery_data(
            query,
            param=QueryParam(
                mode="hybrid", top_k=60, chunk_top_k=10,
                max_entity_tokens=12000, max_relation_tokens=16000,
                max_total_tokens=50000,
            ),
        )
        if result.get("status") != "success":
            return ["No context retrieved."]
        data = result.get("data", {})
        contexts = []
        for c in data.get("chunks", [])[:10]:
            content = c.get("content", "").strip()
            if content:
                contexts.append(content)
        return contexts if contexts else ["No context retrieved."]
    except Exception as e:
        return [f"Context retrieval failed: {e}"]


def safe_float(val):
    if val is None:
        return 0.0
    f = float(val)
    if f != f:
        return 0.0
    return round(f, 4)


def collect_agentic_info(agent):
    """Extract subtask info, grade history, and final grade from AgenticRAG."""
    info = {
        "subtasks": [],
        "num_subtasks": 0,
        "plan_attempts": 0,
        "grade_history": [],
        "final_grade": None,
    }
    if agent.state and agent.state.tasks:
        for tid, task in agent.state.tasks.items():
            info["subtasks"].append({
                "id": task.id,
                "query": task.query,
                "mode": task.mode,
                "status": task.status,
                "result": task.result,
                "result_summary": task.result_summary,
            })
        info["num_subtasks"] = len(info["subtasks"])
    if agent.state:
        info["plan_attempts"] = agent.state.plan_attempts + 1
        info["grade_history"] = agent.state.grade_history
        info["final_grade"] = agent.state.final_grade
    return info


def summarize_subtask_result(result_text, max_len=200):
    """Truncate a sub-task result for display."""
    if not result_text:
        return "No result."
    text = str(result_text).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# =====================================================================
# Factory functions (parameterized by API key)
# =====================================================================


def make_llm_func(api_key):
    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        if history_messages is None:
            history_messages = []
        return await gemini_model_complete(
            prompt, system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key, model_name="gemini-2.0-flash", **kwargs,
        )
    return llm_model_func


def make_embedding_func(api_key):
    @wrap_embedding_func_with_attrs(
        embedding_dim=3072, max_token_size=2048,
        model_name="models/gemini-embedding-001",
    )
    async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
        return await gemini_embed.func(
            texts, api_key=api_key, model="models/gemini-embedding-001",
        )
    return embedding_func


def make_ragas_eval_objects(api_key):
    eval_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key),
        bypass_n=True,
    )
    eval_embed = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=api_key,
        )
    )
    return eval_llm, eval_embed


# =====================================================================
# Per-query runners for each method
# =====================================================================


async def run_standard_rag_query(rag, query, query_num):
    """Run a single query through Standard RAG (all 4 modes). PASS if any mode passes."""
    contexts = await get_retrieval_contexts(rag, query)

    best_ok = False
    best_answer = "No answer produced."
    best_mode = "none"
    total_time = 0.0
    mode_results = {}

    for mode in ("hybrid", "local", "global", "naive"):
        ok, answer, elapsed = await run_single_query(
            f"rag_{mode}",
            rag.aquery(query, param=QueryParam(mode=mode)),
        )
        total_time += elapsed
        mode_results[mode] = {"success": ok, "answer": answer, "time_s": elapsed}
        if ok and not best_ok:
            best_ok = True
            best_answer = answer
            best_mode = mode
        if not best_ok:
            best_answer = answer

    return {
        "query_num": query_num,
        "query": query,
        "contexts": contexts,
        "standard": {
            "success": best_ok,
            "time_s": round(total_time, 2),
            "answer": best_answer,
            "passed_mode": best_mode,
            "mode_results": mode_results,
        },
    }


async def run_agentic_query(rag, agent, query, query_num):
    """Run a single query through AgenticRAG."""
    contexts = await get_retrieval_contexts(rag, query)

    ok, answer, elapsed = await run_single_query(
        "agentic", agent.run(query),
    )
    a_info = collect_agentic_info(agent)

    return {
        "query_num": query_num,
        "query": query,
        "contexts": contexts,
        "agentic": {
            "success": ok, "time_s": elapsed,
            "answer": answer,
            "subtasks": a_info["subtasks"],
            "num_subtasks": a_info["num_subtasks"],
            "plan_attempts": a_info["plan_attempts"],
            "grade_history": a_info["grade_history"],
            "final_grade": a_info["final_grade"],
        },
    }


# =====================================================================
# RAGAS evaluation for a single method
# =====================================================================


def run_ragas_for_method(api_key, results, method_key):
    """Run RAGAS on one method's answers. Mutates results in-place."""
    entries = [r for r in results if method_key in r]
    if not entries:
        return

    eval_llm, eval_embed = make_ragas_eval_objects(api_key)
    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embed),
        ContextRelevance(llm=eval_llm),
        ResponseGroundedness(llm=eval_llm),
    ]

    eval_data = [{
        "user_input": r["query"],
        "response": r[method_key]["answer"] if r[method_key]["success"] else "No answer produced.",
        "retrieved_contexts": r["contexts"],
    } for r in entries]

    try:
        eval_result = ragas_evaluate(
            dataset=EvaluationDataset.from_list(eval_data),
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embed,
        )
        df = eval_result.to_pandas()
    except Exception as e:
        st.error(f"RAGAS evaluation failed for {method_key}: {e}")
        for r in entries:
            r[method_key]["ragas"] = {
                "faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_relevance": 0.0,
                "response_groundedness": 0.0,
            }
        return

    metric_cols = [
        "faithfulness", "answer_relevancy",
        "context_relevance", "response_groundedness",
    ]
    for i, r in enumerate(entries):
        r[method_key]["ragas"] = {}
        for col in metric_cols:
            if col in df.columns:
                r[method_key]["ragas"][col] = safe_float(df.iloc[i].get(col))
            else:
                r[method_key]["ragas"][col] = 0.0


# =====================================================================
# Streamlit App
# =====================================================================

st.title("LightRAG Benchmark")
st.caption("Upload a PDF, enter questions, run Standard LightRAG or AgenticRAG, view RAGAS evaluation metrics.")

# --- Architecture Overview ---
with st.expander("Architecture Overview", expanded=False):
    st.markdown("""
### Standard LightRAG
Direct retrieval using knowledge graph + vector search across 4 modes (hybrid, local, global, naive).
Best answer from any passing mode is selected.

### AgenticRAG Pipeline
Multi-step agentic retrieval with self-correction:

```
Query --> Classifier --> Simple? --> Direct LightRAG query
                    |
                    +--> Complex? --> Planner Agent
                                        |
                                        v
                                  Sub-task DAG (dependency-aware)
                                        |
                                        v
                              Execute each sub-task via LightRAG
                              (local / global / hybrid / naive / reasoning)
                                        |
                                        v
                                  Final Synthesis (merge all sub-task results)
                                        |
                                        v
                                  Grader Agent (evaluate answer quality)
                                        |
                                  PASS --> Return answer
                                  FAIL --> Re-plan with feedback (up to 3x)
```

**Key advantages**: Decomposes complex queries into focused sub-tasks,
uses dependency-aware execution order, and self-corrects via grader feedback loops.
""")

# --- Re-init RAG if server restarted but session survives ---
if (
    st.session_state.phase != "idle"
    and st.session_state.rag is None
    and st.session_state.working_dir
    and st.session_state.api_key
):
    async def _reinit_rag():
        rag = LightRAG(
            working_dir=st.session_state.working_dir,
            llm_model_func=make_llm_func(st.session_state.api_key),
            embedding_func=make_embedding_func(st.session_state.api_key),
            llm_model_name="gemini-2.0-flash",
            llm_model_kwargs={"temperature": 0, "seed": 42},
        )
        await rag.initialize_storages()
        return rag
    st.session_state.rag = run_async(_reinit_rag())

# =====================================================================
# Sidebar
# =====================================================================

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "GEMINI_API_KEY",
        value=st.session_state.api_key,
        type="password",
    )
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key

    st.divider()

    st.subheader("Status")
    st.write(f"Phase: **{st.session_state.phase.upper()}**")
    if st.session_state.pdf_name:
        st.write(f"PDF: {st.session_state.pdf_name}")
    st.write(f"Questions: {len(st.session_state.questions)}")

    # Count results per method
    s_count = sum(1 for r in st.session_state.results if "standard" in r)
    a_count = sum(1 for r in st.session_state.results if "agentic" in r)
    st.write(f"Standard RAG runs: {s_count}")
    st.write(f"AgenticRAG runs: {a_count}")

    st.divider()

    st.subheader("Options")
    st.session_state.enable_ragas = st.checkbox(
        "Enable RAGAS evaluation", value=st.session_state.enable_ragas
    )

    st.divider()

    if st.button("Reset All", type="secondary"):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()

# =====================================================================
# Section 1: Upload Document
# =====================================================================

st.header("1. Upload Document")

if st.session_state.phase == "idle":
    if not st.session_state.api_key:
        st.session_state.api_key = 'AIzaSyAUUznmoKemYlmV2goVU_G46h-R9PcySZg'
        st.warning("Enter your GEMINI_API_KEY in the sidebar first.")

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file is not None and st.session_state.api_key:
        index_btn = st.button("Index Document", type="primary")

        if index_btn:
            pdf_hash = compute_pdf_hash(uploaded_file)
            working_dir = f"./rag_storage_{pdf_hash}"

            st.session_state.pdf_name = uploaded_file.name
            st.session_state.working_dir = working_dir

            progress = st.progress(0, text="Extracting PDF text...")

            pdf_text = read_pdf_from_upload(uploaded_file)
            st.session_state.pdf_text = pdf_text
            progress.progress(0.2, text="Initializing LightRAG...")

            async def _init_and_index():
                os.makedirs(working_dir, exist_ok=True)
                rag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=make_llm_func("AIzaSyAUUznmoKemYlmV2goVU_G46h-R9PcySZg"),
                    embedding_func=make_embedding_func("AIzaSyAUUznmoKemYlmV2goVU_G46h-R9PcySZg"),
                    llm_model_name="gemini-2.0-flash",
                    llm_model_kwargs={"temperature": 0, "seed": 42},
                )
                await rag.initialize_storages()
                if not os.path.exists(os.path.join(working_dir, "vdb_chunks.json")):
                    await rag.ainsert(pdf_text)
                return rag

            progress.progress(0.3, text="Indexing document (this may take several minutes)...")
            rag = run_async(_init_and_index())

            st.session_state.rag = rag
            st.session_state.phase = "ready"
            progress.progress(1.0, text="Indexing complete!")
            st.rerun()

elif st.session_state.phase in ("ready", "done"):
    char_count = len(st.session_state.pdf_text) if st.session_state.pdf_text else 0
    st.success(
        f"Document indexed: **{st.session_state.pdf_name}** ({char_count:,} characters)"
    )

# =====================================================================
# Section 2: Enter Questions
# =====================================================================

if st.session_state.phase in ("ready", "done") and not st.session_state.running:
    st.header("2. Enter Questions")

    bulk_text = st.text_area(
        "Paste multiple questions (one per line)",
        height=150,
        placeholder=(
            "What is the main thesis of the book?\n"
            "How does the author define machine learning?\n"
            "Compare supervised vs unsupervised learning."
        ),
    )

    single_q = st.text_input("Or add a single question", placeholder="Type a question...")

    col1, col2, _col3 = st.columns([1, 1, 2])
    with col1:
        add_btn = st.button("Add Questions", type="primary")
    with col2:
        clear_btn = st.button("Clear All Questions")

    if add_btn:
        new_qs = []
        if bulk_text.strip():
            new_qs.extend([q.strip() for q in bulk_text.strip().split("\n") if q.strip()])
        if single_q.strip():
            new_qs.append(single_q.strip())
        existing = set(st.session_state.questions)
        for q in new_qs:
            if q not in existing:
                st.session_state.questions.append(q)
                existing.add(q)
        st.rerun()

    if clear_btn:
        st.session_state.questions = []
        st.session_state.results = []
        st.session_state.phase = "ready"
        st.rerun()

    if st.session_state.questions:
        st.subheader(f"Question List ({len(st.session_state.questions)} total)")
        for i, q in enumerate(st.session_state.questions):
            has_s = any(r["query"] == q and "standard" in r for r in st.session_state.results)
            has_a = any(r["query"] == q and "agentic" in r for r in st.session_state.results)
            tags = []
            if has_s:
                tags.append("RAG")
            if has_a:
                tags.append("Agentic")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            st.text(f"  Q{i+1}{tag_str}: {q[:100]}{'...' if len(q) > 100 else ''}")

# =====================================================================
# Section 3: Run Benchmark
# =====================================================================

if (
    st.session_state.phase in ("ready", "done")
    and st.session_state.questions
    and not st.session_state.running
):
    st.header("3. Run Benchmark")
    st.write(f"{len(st.session_state.questions)} question(s) loaded.")

    col_rag, col_agentic = st.columns(2)

    with col_rag:
        st.subheader("Standard LightRAG")
        st.caption("Runs all 4 modes (hybrid, local, global, naive) -- PASS if any passes")
        run_rag_btn = st.button("Run Standard RAG", type="primary", key="run_rag")

    with col_agentic:
        st.subheader("AgenticRAG")
        st.caption("Classify -> Plan -> Execute sub-tasks -> Grade -> Re-plan loop")
        run_agentic_btn = st.button("Run AgenticRAG", type="primary", key="run_agentic")

    # ---------- Run Standard RAG ----------
    if run_rag_btn:
        st.session_state.running = True
        rag = st.session_state.rag
        questions = st.session_state.questions

        progress_bar = st.progress(0, text="Running Standard RAG...")
        status_text = st.empty()
        live_table = st.empty()

        for i, query in enumerate(questions):
            query_num = i + 1
            progress_bar.progress(
                (i + 1) / (len(questions) + 1),
                text=f"Standard RAG -- Query {i + 1}/{len(questions)}",
            )
            status_text.text(f"Q{query_num}: {query[:70]}...")

            r = run_async(run_standard_rag_query(rag, query, query_num))

            existing = next((x for x in st.session_state.results if x["query"] == query), None)
            if existing:
                existing["standard"] = r["standard"]
                existing["contexts"] = r["contexts"]
            else:
                st.session_state.results.append(r)

            rows = []
            for res in st.session_state.results:
                if "standard" in res:
                    rows.append({
                        "#": res["query_num"],
                        "Status": "PASS" if res["standard"]["success"] else "FAIL",
                        "Mode": res["standard"].get("passed_mode", "none"),
                        "Time": res["standard"]["time_s"],
                        "Query": res["query"][:55] + "...",
                    })
            live_table.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.session_state.enable_ragas:
            progress_bar.progress(0.95, text="Running RAGAS evaluation for Standard RAG...")
            status_text.text("RAGAS evaluation (may take several minutes)...")
            run_ragas_for_method(st.session_state.api_key, st.session_state.results, "standard")

        st.session_state.running = False
        st.session_state.phase = "done"
        progress_bar.progress(1.0, text="Standard RAG complete!")
        status_text.text("Done!")
        st.rerun()

    # ---------- Run AgenticRAG ----------
    if run_agentic_btn:
        st.session_state.running = True
        rag = st.session_state.rag
        agent = AgenticRAG(rag)
        questions = st.session_state.questions

        progress_bar = st.progress(0, text="Running AgenticRAG...")
        status_text = st.empty()
        live_table = st.empty()

        for i, query in enumerate(questions):
            query_num = i + 1
            progress_bar.progress(
                (i + 1) / (len(questions) + 1),
                text=f"AgenticRAG -- Query {i + 1}/{len(questions)}",
            )
            status_text.text(f"Q{query_num}: {query[:70]}...")

            r = run_async(run_agentic_query(rag, agent, query, query_num))

            existing = next((x for x in st.session_state.results if x["query"] == query), None)
            if existing:
                existing["agentic"] = r["agentic"]
                if "contexts" not in existing:
                    existing["contexts"] = r["contexts"]
            else:
                st.session_state.results.append(r)

            rows = []
            for res in st.session_state.results:
                if "agentic" in res:
                    fg = res["agentic"].get("final_grade") or {}
                    grade_str = "PASS" if fg.get("passed") else "FAIL" if fg else "N/A"
                    rows.append({
                        "#": res["query_num"],
                        "Status": "PASS" if res["agentic"]["success"] else "FAIL",
                        "Time": res["agentic"]["time_s"],
                        "Tasks": res["agentic"]["num_subtasks"],
                        "Attempts": res["agentic"]["plan_attempts"],
                        "Grade": grade_str,
                        "Query": res["query"][:50] + "...",
                    })
            live_table.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.session_state.enable_ragas:
            progress_bar.progress(0.95, text="Running RAGAS evaluation for AgenticRAG...")
            status_text.text("RAGAS evaluation (may take several minutes)...")
            run_ragas_for_method(st.session_state.api_key, st.session_state.results, "agentic")

        st.session_state.running = False
        st.session_state.phase = "done"
        progress_bar.progress(1.0, text="AgenticRAG complete!")
        status_text.text("Done!")
        st.rerun()

# =====================================================================
# Section 4: Results Dashboard
# =====================================================================

if st.session_state.results:
    results = st.session_state.results
    total = len(results)

    has_standard = any("standard" in r for r in results)
    has_agentic = any("agentic" in r for r in results)

    def avg_metric(method, metric):
        vals = [r[method].get("ragas", {}).get(metric, 0) for r in results if method in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def pass_count(method):
        return sum(1 for r in results if method in r and r[method]["success"])

    def method_count(method):
        return sum(1 for r in results if method in r)

    def avg_time(method):
        times = [r[method]["time_s"] for r in results if method in r]
        return round(sum(times) / len(times), 1) if times else 0.0

    def has_nonzero_metric(metric_key):
        """Check if any result has a non-zero value for this RAGAS metric."""
        for r in results:
            for method in ("standard", "agentic"):
                if method in r:
                    val = r[method].get("ragas", {}).get(metric_key, 0)
                    if val and val > 0:
                        return True
        return False

    # ---- Evaluation Matrix ----
    st.header("Evaluation Matrix")

    if has_standard or has_agentic:
        matrix_data = {}
        metrics_list = [
            ("Pass Rate", None),
            ("Avg Time (s)", None),
            ("Faithfulness", "faithfulness"),
            ("Answer Relevancy", "answer_relevancy"),
            ("Context Relevance", "context_relevance"),
            ("Response Groundedness", "response_groundedness"),
        ]
        # Filter out RAGAS metrics that are all zero
        metrics_list = [
            (label, key) for label, key in metrics_list
            if key is None or has_nonzero_metric(key)
        ]

        for label, metric_key in metrics_list:
            row = {}
            if has_standard:
                if label == "Pass Rate":
                    sc = method_count("standard")
                    row["Standard LightRAG"] = f"{pass_count('standard')}/{sc}"
                elif label == "Avg Time (s)":
                    row["Standard LightRAG"] = f"{avg_time('standard')}"
                else:
                    row["Standard LightRAG"] = f"{avg_metric('standard', metric_key):.4f}"
            if has_agentic:
                if label == "Pass Rate":
                    ac = method_count("agentic")
                    row["AgenticRAG"] = f"{pass_count('agentic')}/{ac}"
                elif label == "Avg Time (s)":
                    row["AgenticRAG"] = f"{avg_time('agentic')}"
                else:
                    row["AgenticRAG"] = f"{avg_metric('agentic', metric_key):.4f}"
            matrix_data[label] = row

        st.dataframe(
            pd.DataFrame(matrix_data).T.rename_axis("Metric"),
            use_container_width=True,
        )

    # ---- Summary Metrics Cards ----
    st.header("Summary")
    cols = st.columns(2 if (has_standard and has_agentic) else 1)

    col_idx = 0
    def _render_method_summary(method, label):
        """Render summary metric cards for a method, skipping all-zero RAGAS metrics."""
        mc = method_count(method)
        m1, m2, m3 = st.columns(3)
        m1.metric("Pass Rate", f"{pass_count(method)}/{mc}")
        m2.metric("Avg Time", f"{avg_time(method)}s")
        m3.metric("Faithfulness", f"{avg_metric(method, 'faithfulness'):.3f}")
        ragas_cards = [
            ("Ans Relevancy", "answer_relevancy"),
            ("Ctx Relevance", "context_relevance"),
            ("Groundedness", "response_groundedness"),
        ]
        visible = [(lbl, key) for lbl, key in ragas_cards if has_nonzero_metric(key)]
        if visible:
            card_cols = st.columns(len(visible))
            for i, (lbl, key) in enumerate(visible):
                card_cols[i].metric(lbl, f"{avg_metric(method, key):.3f}")

    if has_standard:
        with cols[col_idx]:
            st.subheader("Standard LightRAG (Best of 4 Modes)")
            _render_method_summary("standard", "Standard LightRAG")
        col_idx += 1

    if has_agentic:
        with cols[col_idx]:
            st.subheader("AgenticRAG")
            _render_method_summary("agentic", "AgenticRAG")

            # AgenticRAG-specific stats
            total_attempts = sum(
                r["agentic"].get("plan_attempts", 1)
                for r in results if "agentic" in r
            )
            avg_attempts = round(total_attempts / ac, 1) if ac else 0
            grade_pass = sum(
                1 for r in results
                if "agentic" in r
                and r["agentic"].get("final_grade", {})
                and r["agentic"]["final_grade"].get("passed")
            )
            m7, m8, m9 = st.columns(3)
            m7.metric("Avg Plan Attempts", f"{avg_attempts}")
            m8.metric("Grader Pass", f"{grade_pass}/{ac}")
            avg_tasks = round(
                sum(r["agentic"]["num_subtasks"] for r in results if "agentic" in r) / ac, 1
            ) if ac else 0
            m9.metric("Avg Sub-tasks", f"{avg_tasks}")

    # ---- Per-Query Results Table ----
    st.header("Per-Query Results")
    table_data = []
    for r in results:
        row = {"#": r["query_num"], "Query": r["query"][:55] + "..."}
        if has_standard and "standard" in r:
            row["RAG"] = "PASS" if r["standard"]["success"] else "FAIL"
            row["RAG Mode"] = r["standard"].get("passed_mode", "N/A")
            row["RAG Time"] = r["standard"]["time_s"]
            gr = r["standard"].get("ragas", {})
            row["RAG Faith"] = safe_float(gr.get("faithfulness"))
            if has_nonzero_metric("answer_relevancy"):
                row["RAG AnsR"] = safe_float(gr.get("answer_relevancy"))
            if has_nonzero_metric("context_relevance"):
                row["RAG CtxR"] = safe_float(gr.get("context_relevance"))
            if has_nonzero_metric("response_groundedness"):
                row["RAG Grnd"] = safe_float(gr.get("response_groundedness"))
        if has_agentic and "agentic" in r:
            row["Agent"] = "PASS" if r["agentic"]["success"] else "FAIL"
            row["A Time"] = r["agentic"]["time_s"]
            row["A Tasks"] = r["agentic"]["num_subtasks"]
            row["A Attempts"] = r["agentic"]["plan_attempts"]
            fg = r["agentic"].get("final_grade") or {}
            row["A Grade"] = "PASS" if fg.get("passed") else "FAIL" if fg else "N/A"
            pr = r["agentic"].get("ragas", {})
            row["A Faith"] = safe_float(pr.get("faithfulness"))
            if has_nonzero_metric("answer_relevancy"):
                row["A AnsR"] = safe_float(pr.get("answer_relevancy"))
            if has_nonzero_metric("context_relevance"):
                row["A CtxR"] = safe_float(pr.get("context_relevance"))
            if has_nonzero_metric("response_groundedness"):
                row["A Grnd"] = safe_float(pr.get("response_groundedness"))
        table_data.append(row)
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ---- Answers Display ----
    st.header("Answers")
    for r in results:
        st.subheader(f"Q{r['query_num']}: {r['query'][:90]}{'...' if len(r['query']) > 90 else ''}")

        ans_cols = st.columns(2 if (has_standard and has_agentic) else 1)
        ac = 0

        if has_standard and "standard" in r:
            with ans_cols[ac]:
                status = "PASS" if r["standard"]["success"] else "FAIL"
                mode = r["standard"].get("passed_mode", "none")
                st.markdown(f"**Standard LightRAG** -- {status} (mode: {mode}, {r['standard']['time_s']}s)")
                st.write(r["standard"].get("answer", "N/A"))
            ac += 1

        if has_agentic and "agentic" in r:
            with ans_cols[ac]:
                status = "PASS" if r["agentic"]["success"] else "FAIL"
                fg = r["agentic"].get("final_grade") or {}
                grade_str = ""
                if fg:
                    grade_status = "PASS" if fg.get("passed") else "FAIL"
                    grade_str = f" | Grade: {grade_status}"
                    if "relevance" in fg:
                        grade_str += f" (R={fg['relevance']:.1f} C={fg['completeness']:.1f} S={fg['sufficiency']:.1f})"
                st.markdown(
                    f"**AgenticRAG** -- {status} "
                    f"({r['agentic']['time_s']}s, "
                    f"{r['agentic']['num_subtasks']} tasks, "
                    f"{r['agentic']['plan_attempts']} attempt(s)"
                    f"{grade_str})"
                )
                st.write(r["agentic"].get("answer", "N/A"))

                # Sub-task summary
                subtasks = r["agentic"].get("subtasks", [])
                if subtasks:
                    st.markdown("---")
                    st.markdown("**Sub-task Breakdown:**")
                    for st_info in subtasks:
                        mode_badge = f"`{st_info['mode']}`"
                        st.markdown(
                            f"**{st_info['id']}** {mode_badge} -- {st_info['query']}"
                        )
                        # Use LLM-generated summary if available, else truncate
                        summary = st_info.get("result_summary")
                        if not summary:
                            summary = summarize_subtask_result(st_info.get("result"), max_len=300)
                        st.caption(summary)

        st.divider()

    # ---- Detailed Results (expandable) ----
    st.subheader("Detailed Results")
    for r in results:
        with st.expander(f"Q{r['query_num']}: {r['query'][:80]}..."):
            st.markdown(f"**Full Query:** {r['query']}")

            detail_cols = st.columns(2 if (has_standard and has_agentic) else 1)
            dc = 0

            if has_standard and "standard" in r:
                with detail_cols[dc]:
                    st.markdown("**Standard LightRAG**")
                    st.write(f"Status: {'PASS' if r['standard']['success'] else 'FAIL'} | Mode: {r['standard'].get('passed_mode', 'N/A')}")
                    st.write(f"Time: {r['standard']['time_s']}s (total across 4 modes)")
                    gr = r["standard"].get("ragas", {})
                    if gr:
                        ragas_parts = [f"Faith: {gr.get('faithfulness', 'N/A')}"]
                        if has_nonzero_metric("answer_relevancy"):
                            ragas_parts.append(f"AnsR: {gr.get('answer_relevancy', 'N/A')}")
                        if has_nonzero_metric("context_relevance"):
                            ragas_parts.append(f"CtxR: {gr.get('context_relevance', 'N/A')}")
                        if has_nonzero_metric("response_groundedness"):
                            ragas_parts.append(f"Grnd: {gr.get('response_groundedness', 'N/A')}")
                        st.write(" | ".join(ragas_parts))
                    mode_results = r["standard"].get("mode_results", {})
                    if mode_results:
                        st.markdown("**Per-mode results:**")
                        for mname, mdata in mode_results.items():
                            ms = "PASS" if mdata["success"] else "FAIL"
                            st.write(f"- {mname}: {ms} ({mdata['time_s']}s)")
                    st.text_area(
                        "Best Answer", r["standard"].get("answer", "N/A"),
                        height=150, disabled=True, key=f"s_{r['query_num']}",
                    )
                dc += 1

            if has_agentic and "agentic" in r:
                with detail_cols[dc]:
                    st.markdown("**AgenticRAG**")
                    st.write(f"Status: {'PASS' if r['agentic']['success'] else 'FAIL'}")
                    st.write(
                        f"Time: {r['agentic']['time_s']}s | "
                        f"Tasks: {r['agentic']['num_subtasks']} | "
                        f"Plan attempts: {r['agentic']['plan_attempts']}"
                    )

                    # RAGAS scores
                    pr = r["agentic"].get("ragas", {})
                    if pr:
                        ragas_parts = [f"Faith: {pr.get('faithfulness', 'N/A')}"]
                        if has_nonzero_metric("answer_relevancy"):
                            ragas_parts.append(f"AnsR: {pr.get('answer_relevancy', 'N/A')}")
                        if has_nonzero_metric("context_relevance"):
                            ragas_parts.append(f"CtxR: {pr.get('context_relevance', 'N/A')}")
                        if has_nonzero_metric("response_groundedness"):
                            ragas_parts.append(f"Grnd: {pr.get('response_groundedness', 'N/A')}")
                        st.write(" | ".join(ragas_parts))

                    # Final grade
                    fg = r["agentic"].get("final_grade") or {}
                    if fg:
                        fg_status = "PASS" if fg.get("passed") else "FAIL"
                        st.markdown(f"**Final Grade: {fg_status}**")
                        if "relevance" in fg:
                            st.write(
                                f"R={fg['relevance']:.2f} "
                                f"C={fg['completeness']:.2f} "
                                f"S={fg['sufficiency']:.2f}"
                            )
                        if fg.get("reasoning"):
                            st.caption(f"Reasoning: {fg['reasoning'][:200]}")

                    # Grade history
                    gh = r["agentic"].get("grade_history", [])
                    if gh and len(gh) > 1:
                        st.markdown("**Grade History:**")
                        for gi, g in enumerate(gh):
                            st.write(
                                f"  Attempt {g.get('attempt', gi+1)} ({g.get('mode', '?')}): "
                                f"{'PASS' if g.get('passed') else 'FAIL'} "
                                f"R={g.get('relevance', 0):.2f} "
                                f"C={g.get('completeness', 0):.2f} "
                                f"S={g.get('sufficiency', 0):.2f}"
                            )

                    st.text_area(
                        "AgenticRAG Answer", r["agentic"].get("answer", "N/A"),
                        height=150, disabled=True, key=f"a_{r['query_num']}",
                    )

            # Sub-tasks with full results
            if "agentic" in r and r["agentic"].get("subtasks"):
                st.markdown("**AgenticRAG Sub-tasks (full results):**")
                for st_info in r["agentic"]["subtasks"]:
                    with st.expander(
                        f"[{st_info['id']}] mode={st_info['mode']} | {st_info['query'][:70]}",
                        expanded=False,
                    ):
                        st.markdown(f"**Query:** {st_info['query']}")
                        st.markdown(f"**Mode:** `{st_info['mode']}` | **Status:** {st_info['status']}")
                        result_text = st_info.get("result", "No result.")
                        if result_text:
                            st.markdown("**Result:**")
                            st.write(str(result_text))

    # ---- Charts ----
    st.header("Charts")

    if has_standard and has_agentic:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Pass / Fail")
            st.bar_chart(pd.DataFrame({
                "Pass": [pass_count("standard"), pass_count("agentic")],
                "Fail": [method_count("standard") - pass_count("standard"),
                         method_count("agentic") - pass_count("agentic")],
            }, index=["Standard LightRAG", "AgenticRAG"]))

        with c2:
            st.subheader("Avg RAGAS Scores")
            chart_metrics = [("Faithfulness", "faithfulness")]
            if has_nonzero_metric("answer_relevancy"):
                chart_metrics.append(("Ans Relevancy", "answer_relevancy"))
            if has_nonzero_metric("context_relevance"):
                chart_metrics.append(("Ctx Relevance", "context_relevance"))
            if has_nonzero_metric("response_groundedness"):
                chart_metrics.append(("Groundedness", "response_groundedness"))
            st.bar_chart(pd.DataFrame({
                "Standard LightRAG": [avg_metric("standard", k) for _, k in chart_metrics],
                "AgenticRAG": [avg_metric("agentic", k) for _, k in chart_metrics],
            }, index=[lbl for lbl, _ in chart_metrics]))

        st.subheader("Per-Query Faithfulness")
        faith_idx = []
        s_faith = []
        a_faith = []
        for r in results:
            faith_idx.append(f"Q{r['query_num']}")
            s_faith.append(r.get("standard", {}).get("ragas", {}).get("faithfulness", 0))
            a_faith.append(r.get("agentic", {}).get("ragas", {}).get("faithfulness", 0))
        st.bar_chart(pd.DataFrame({
            "Standard LightRAG": s_faith, "AgenticRAG": a_faith,
        }, index=faith_idx))

        st.subheader("Per-Query Execution Time (seconds)")
        st.bar_chart(pd.DataFrame({
            "Standard LightRAG": [r.get("standard", {}).get("time_s", 0) for r in results],
            "AgenticRAG": [r.get("agentic", {}).get("time_s", 0) for r in results],
        }, index=[f"Q{r['query_num']}" for r in results]))

    elif has_standard:
        st.subheader("Standard LightRAG -- Per-Query Faithfulness")
        st.bar_chart(pd.DataFrame({
            "Faithfulness": [r["standard"].get("ragas", {}).get("faithfulness", 0) for r in results if "standard" in r],
        }, index=[f"Q{r['query_num']}" for r in results if "standard" in r]))

    elif has_agentic:
        st.subheader("AgenticRAG -- Per-Query Faithfulness")
        st.bar_chart(pd.DataFrame({
            "Faithfulness": [r["agentic"].get("ragas", {}).get("faithfulness", 0) for r in results if "agentic" in r],
        }, index=[f"Q{r['query_num']}" for r in results if "agentic" in r]))

    # ---- Failure Analysis ----
    if has_standard and has_agentic:
        st.header("Failure Analysis")
        both_have = [r for r in results if "standard" in r and "agentic" in r]
        a_only = [r for r in both_have if r["agentic"]["success"] and not r["standard"]["success"]]
        s_only = [r for r in both_have if r["standard"]["success"] and not r["agentic"]["success"]]
        both_fail = [r for r in both_have if not r["standard"]["success"] and not r["agentic"]["success"]]

        if a_only:
            st.subheader(f"AgenticRAG PASS / Standard RAG FAIL ({len(a_only)})")
            for r in a_only:
                st.write(f"- Q{r['query_num']}: {r['query'][:70]}...")
        if s_only:
            st.subheader(f"Standard RAG PASS / AgenticRAG FAIL ({len(s_only)})")
            for r in s_only:
                st.write(f"- Q{r['query_num']}: {r['query'][:70]}...")
        if both_fail:
            st.subheader(f"Both FAIL ({len(both_fail)})")
            for r in both_fail:
                st.write(f"- Q{r['query_num']}: {r['query'][:70]}...")
        if not a_only and not s_only and not both_fail:
            st.success("All queries passed by both methods!")

    # ---- Export ----
    st.header("Export")
    e1, e2 = st.columns(2)

    json_export = [{k: v for k, v in r.items() if k != "contexts"} for r in results]

    with e1:
        st.download_button(
            "Download JSON",
            data=json.dumps(json_export, indent=2, ensure_ascii=False, default=str),
            file_name="benchmark_report.json",
            mime="application/json",
        )

    with e2:
        buf = io.StringIO()
        writer = csv.writer(buf)
        header = ["query_num", "query"]
        if has_standard:
            header += [
                "rag_pass", "rag_time_s",
                "rag_faithfulness", "rag_answer_relevancy",
                "rag_ctx_relevance", "rag_groundedness",
            ]
        if has_agentic:
            header += [
                "agentic_pass", "agentic_time_s",
                "agentic_faithfulness", "agentic_answer_relevancy",
                "agentic_ctx_relevance", "agentic_groundedness",
                "agentic_num_subtasks", "agentic_plan_attempts",
                "agentic_grade_passed", "agentic_R", "agentic_C", "agentic_S",
            ]
        writer.writerow(header)
        for r in results:
            row = [r["query_num"], r["query"]]
            if has_standard:
                gr = r.get("standard", {}).get("ragas", {})
                row += [
                    r.get("standard", {}).get("success", ""),
                    r.get("standard", {}).get("time_s", ""),
                    gr.get("faithfulness", ""),
                    gr.get("answer_relevancy", ""),
                    gr.get("context_relevance", ""),
                    gr.get("response_groundedness", ""),
                ]
            if has_agentic:
                ar = r.get("agentic", {}).get("ragas", {})
                fg = r.get("agentic", {}).get("final_grade") or {}
                row += [
                    r.get("agentic", {}).get("success", ""),
                    r.get("agentic", {}).get("time_s", ""),
                    ar.get("faithfulness", ""),
                    ar.get("answer_relevancy", ""),
                    ar.get("context_relevance", ""),
                    ar.get("response_groundedness", ""),
                    r.get("agentic", {}).get("num_subtasks", ""),
                    r.get("agentic", {}).get("plan_attempts", ""),
                    fg.get("passed", ""),
                    fg.get("relevance", ""),
                    fg.get("completeness", ""),
                    fg.get("sufficiency", ""),
                ]
            writer.writerow(row)
        st.download_button(
            "Download CSV",
            data=buf.getvalue(),
            file_name="benchmark_report.csv",
            mime="text/csv",
        )

elif st.session_state.phase == "idle":
    st.info("Upload a PDF and enter your GEMINI_API_KEY to get started.")
