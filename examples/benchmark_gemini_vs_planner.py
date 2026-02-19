import os
import re
import asyncio
import time
import json
import csv
import numpy as np
import fitz

from lightrag import LightRAG, QueryParam
from lightrag.planner_agent import PlannerAgent
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

# RAGAS imports
from ragas import EvaluationDataset, evaluate as ragas_evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics import _LLMContextPrecisionWithoutReference as LLMContextPrecisionWithoutReference
from ragas.metrics import _ContextUtilization as ContextUtilization
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Constants
WORKING_DIR = "./rag_storage"
PDF_FILE = "./MLBOOK.pdf"
REPORT_FILE = "./benchmark_report.json"
CSV_FILE = "./benchmark_ragas.csv"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# --- LLM & Embedding (for RAG) ---

async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    if history_messages is None:
        history_messages = []
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.0-flash",
        **kwargs,
    )

@wrap_embedding_func_with_attrs(
    embedding_dim=3072,
    max_token_size=2048,
    model_name="models/gemini-embedding-001",
)
async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    return await gemini_embed.func(
        texts,
        api_key=GEMINI_API_KEY,
        model="models/gemini-embedding-001"
    )

# --- RAGAS Eval LLM & Embeddings (Gemini as judge) ---

eval_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY),
    bypass_n=True,
)
eval_embed = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )
)

# --- 30 Queries ---

QUERIES = [
    "Identify a technique in the book that uses 'Search' as its underlying mechanism for learning but does not belong to the Neural Network or Statistical Learning paradigms.",
    "Locate the specific section where the author defines the 'environment' in which the learner operates. Now, identify a learning paradigm discussed in the second half of the book that fundamentally alters that environment's state through its actions, rather than just observing it. Contrast its objective function with the objective function of the most basic linear classifier.",
    "Find the methodology that relies on a 'subset of the power set of all possible instances' to define its hypothesis space. Explain why this specific methodology would be computationally infeasible for the type of signal processing tasks described in the neural network chapters.",
    "The author discusses a 'boundary' that separates successful classification from error. Find a technique in the book where this boundary is not a static line or curve, but is instead represented by a collection of logical rules. How does the 'search' through these rules differ from the 'search' performed by gradient descent?",
    "In the beginning of the text, the author discusses the necessity of a 'bias' to allow for generalization. Find the specific technique introduced much later that uses 'prior knowledge in the form of a partial theory' to guide the learning process. How does this technique mathematically reconcile an existing symbolic rule-set with new, conflicting empirical observations?",
    "The author describes a method for organizing a hypothesis space into a 'nested hierarchy of increasingly complex structures.' If a learner has limited time to find a solution, explain the trade-off the book suggests between the 'depth' of this search and the 'generalization power' of the resulting model. Does this approach favor 'simpler' or 'more specific' explanations in a noisy environment?",
    "Identify the moment in the text where the author stops using predicate calculus to represent knowledge and starts using vectors of real numbers. What is the 'justification' provided for this shift in representation, and does the author acknowledge any loss of interpretability during this transition?",
    "Find the 'convergence theorem' mentioned in the context of simplest linear learners. Now, find the 'PAC learning' bounds discussed in the computational theory section. Does the book provide a specific scenario where the PAC requirements are satisfied but the convergence theorem would still fail to reach a solution in finite time?",
    "Throughout the book, the author discusses 'Symbolic' and 'Sub-symbolic' approaches. Based on the concluding remarks of the chapters, does the author seem to believe that these two will eventually merge into a single theory, or does he argue they are fundamentally suited for different types of intelligence?",
    "Locate the discussion on search strategies where a 'cost-to-go' estimate is combined with the 'cost-accrued' value. How does the book define the mathematical condition that this estimate must satisfy to ensure the shortest path is always discovered, and what happens if this condition is violated?",
    "The text discusses a method for handling non-linearly separable data by mapping the input into a higher-dimensional space. Identify the specific mathematical functions used to perform this mapping and explain why this approach is computationally preferable to explicitly adding more hidden layers to a neural structure.",
    "Locate the discussion regarding the 'expressive power' of models. Identify the specific case where a single-layered arrangement of 'linear thresholds' is logically incapable of representing a parity-check function. How does the book suggest we 'expand the input space' to fix this without adding more layers?",
    "Compare the criteria for removing nodes in a structure built via recursive partitioning versus the criteria for adjusting weights to zero in a structure using a continuous error surface. Is the underlying goal of both operations to reduce the variance of the model, and does the author recommend one over the other for high-noise datasets?",
    "Locate the proof or discussion regarding the 'necessity of non-linearity.' Specifically, identify why a system composed of multiple layers of 'linear sum-and-threshold' units--but without a non-linear activation between them--is mathematically equivalent to a single-layer system. What does this imply about the 'depth' of such a structure?",
    "Find the discussion concerning a learner that is restricted to a 'fixed-length vector' representation. Contrast this with a learner that can represent hypotheses as 'variable-length logical formulas.' According to the text, which of these structures has a higher 'expressive power,' and what is the specific cost of that power in terms of the search space size?",
    "There is a transition in the text from a system that seeks to maximize the 'margin' between classes to a system that seeks to minimize the 'sum of squared differences' between predicted and actual values. Locate the section where the author explains why the latter is more appropriate for 'regression' tasks, even though it can be adapted for 'classification' by applying a nonlinear transform to the output.",
    "Find the part of the book that explains why a system with 1,000 layers of simple linear additions is no more powerful than a system with just one layer.",
    "Find the specific mathematical derivation which proves that a structure composed of multiple successive layers of 'weighted sum and threshold' units is functionally identical to a single-stage system if no non-linear transformation is applied between those stages. What are the implications of this proof for the 'expressive power' of hierarchical systems as discussed in the early chapters?",
    "Identify the methodology used to resolve classification failures in datasets that are not separable by a single hyperplane without increasing the number of learning parameters. How does the text describe the process of projecting the input space into a higher-dimensional feature space, and why is this technically considered a 'fixed' rather than 'adaptive' transformation in the context of phi-functions?",
    "Locate the discussion contrasting a system that optimizes for the 'maximum margin of separation' versus one that optimizes for the 'minimum squared difference' between the output and the label. Under what specific conditions of environmental noise does the author suggest the 'squared difference' approach becomes mathematically problematic compared to the margin-based approach?",
    "Examine the sections detailing gradient-based optimization and identify the specific state where a processing unit's internal sum becomes so large that its output gradient effectively vanishes. How does this 'saturation' phenomenon impact the speed of weight updates in deep architectures, and what initialization strategy does the author suggest to mitigate this at the start of training?",
    "In the discussion of navigating an error surface, describe the scenario where a learner encounters a 'plateau' or a 'narrow ravine.' How does the author explain the failure of standard directional updates in these regions, and what physical-inspired 'memory' term is added to the update rule to help the parameters maintain velocity through these areas?",
    "Summarize the 'Future of Machine Learning' as described in the closing pages. Does the author anticipate the 'Deep Learning' revolution, or does he focus more on the 'Integration of Logic and Neural' methods?",
    "Contrast a system that searches through a space of 'fixed-length numerical vectors' with one that searches through a space of 'variable-length logical literals.' What are the specific computational challenges of the latter regarding the size of the search space, and how does the author suggest using 'bottom-up' grounding to limit the search?",
    "Identify the 'Naive' assumption used in statistical modeling where every input feature is treated as if it has no relationship with any other feature. In what specific cases does the author acknowledge that this assumption is fundamentally false, yet the system still produces remarkably accurate results?",
    "Contrast 'Passive' reinforcement learning, where the agent just watches a fixed policy, with 'Active' learning, where the agent changes the policy. How does the 'Value Iteration' algorithm differ from the 'Policy Iteration' algorithm in terms of computational cycles?",
    "Examine the 'Knowledge-Based Artificial Neural Network' (KBANN). How does the system 'map' a set of 'IF-THEN' rules into a network of weights and biases, and what happens to those 'hand-coded' rules as the system starts to see new empirical data?",
    "Find the discussion on 'Robustness.' How does the book suggest handling a 'noisy' feature value versus a 'mislabeled' instance? Which of these is more damaging to a 'Version Space' versus a 'Neural Network'?",
    "Locate the 'Bagging' or 'Boosting' concepts (even if named differently). Does the author suggest that 'averaging' the votes of 10 weak learners can outperform a single strong learner? What is the requirement for the 'diversity' of those weak learners?",
    "Based on the concluding remarks, what does the author identify as the primary 'computational' limit to scaling these algorithms to human-level complexity? Is it the number of 'neurons,' the amount of 'data,' or the 'speed of the search'?",
]

# --- Helpers ---

def read_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text


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

def is_good_answer(text: str) -> bool:
    """PASS only if answer has 5+ sentences and is not a refusal."""
    if not text or not text.strip():
        return False
    lower = text.lower().strip()
    for phrase in FAIL_PHRASES:
        if phrase in lower:
            return False
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return len(sentences) >= 5


async def run_single_query(label, coro):
    """Run a single query coroutine and return (success, result, elapsed)."""
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
    """Get retrieved contexts (entities, relationships, chunks) for RAGAS evaluation."""
    try:
        result = await rag.aquery_data(
            query,
            param=QueryParam(
                mode="hybrid",
                top_k=60,
                chunk_top_k=30,
                max_entity_tokens=12000,
                max_relation_tokens=16000,
                max_total_tokens=50000,
            ),
        )
        if result.get("status") != "success":
            return ["No context retrieved."]

        data = result.get("data", {})
        contexts = []

        for ent in data.get("entities", []):
            desc = ent.get("description", "").strip()
            if desc:
                name = ent.get("entity_name", "")
                etype = ent.get("entity_type", "")
                contexts.append(f"[Entity: {name} ({etype})] {desc}")

        for rel in data.get("relationships", []):
            desc = rel.get("description", "").strip()
            if desc:
                src = rel.get("src_id", "")
                tgt = rel.get("tgt_id", "")
                contexts.append(f"[Relationship: {src} -> {tgt}] {desc}")

        for c in data.get("chunks", []):
            content = c.get("content", "").strip()
            if content:
                contexts.append(content)

        return contexts if contexts else ["No context retrieved."]
    except Exception as e:
        return [f"Context retrieval failed: {e}"]


def safe_float(val):
    """Convert a value to float, returning 0.0 for NaN/None."""
    if val is None:
        return 0.0
    f = float(val)
    if f != f:  # NaN check
        return 0.0
    return round(f, 4)


# --- Main ---

async def main():
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        llm_model_name="gemini-2.0-flash",
        llm_model_kwargs={"temperature": 0, "seed": 42},
    )
    await rag.initialize_storages()

    # Index if needed
    is_indexed = os.path.exists(os.path.join(WORKING_DIR, "vdb_chunks.json"))
    if not is_indexed:
        if not os.path.exists(PDF_FILE):
            print(f"PDF file {PDF_FILE} not found!")
            return
        print(f"Extracting and indexing {PDF_FILE}...")
        content = read_pdf_text(PDF_FILE)
        if content:
            await rag.ainsert(content)
            print("Indexing complete.")

    agent = PlannerAgent(rag)

    results = []
    gemini_eval_data = []
    planner_eval_data = []

    # ========== PHASE 1: Run all queries ==========
    for i, query in enumerate(QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}/{len(QUERIES)}: {query[:80]}...")
        print(f"{'='*60}")

        # Get retrieval context for RAGAS
        print(f"  [Context]       Retrieving...")
        contexts = await get_retrieval_contexts(rag, query)

        # --- Gemini (direct hybrid query) ---
        print(f"  [Gemini hybrid] Running...")
        g_ok, g_result, g_time = await run_single_query(
            "gemini",
            rag.aquery(query, param=QueryParam(mode="hybrid"))
        )
        status = "PASS" if g_ok else "FAIL"
        print(f"  [Gemini hybrid] {status} ({g_time}s)")

        # --- Planner Agent ---
        print(f"  [PlannerAgent]  Running...")
        p_ok, p_result, p_time = await run_single_query(
            "planner",
            agent.run(query)
        )
        status = "PASS" if p_ok else "FAIL"
        print(f"  [PlannerAgent]  {status} ({p_time}s)")

        # Collect sub-task info (including grader metrics)
        subtasks = []
        if agent.state and agent.state.tasks:
            for tid, task in agent.state.tasks.items():
                subtasks.append({
                    "id": task.id,
                    "query": task.query,
                    "mode": task.mode,
                    "status": task.status,
                    "grader_attempts": task.grader_attempts,
                    "grader_history": task.grader_history,
                })

        # Build RAGAS eval entries (only for queries that produced answers)
        gemini_eval_data.append({
            "user_input": query,
            "response": g_result if g_ok else "No answer produced.",
            "retrieved_contexts": contexts,
        })
        planner_eval_data.append({
            "user_input": query,
            "response": p_result if p_ok else "No answer produced.",
            "retrieved_contexts": contexts,
        })

        results.append({
            "query_num": i,
            "query": query,
            "contexts": contexts,
            "gemini": {
                "success": g_ok,
                "time_s": g_time,
                **({"answer": g_result} if g_ok else {}),
            },
            "planner": {
                "success": p_ok,
                "time_s": p_time,
                **({"answer": p_result} if p_ok else {}),
                "subtasks": subtasks,
            },
        })

    await rag.finalize_storages()

    # ========== PHASE 2: RAGAS Evaluation ==========
    print("\n\n" + "=" * 70)
    print("Running RAGAS evaluation (Gemini as judge)...")
    print("=" * 70)

    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embed),
        LLMContextPrecisionWithoutReference(llm=eval_llm),
        ContextUtilization(llm=eval_llm),
    ]

    # Evaluate Gemini answers
    print("  Evaluating Gemini (hybrid) answers...")
    gemini_dataset = EvaluationDataset.from_list(gemini_eval_data)
    gemini_ragas = ragas_evaluate(dataset=gemini_dataset, metrics=metrics)
    gemini_df = gemini_ragas.to_pandas()

    # Evaluate Planner answers
    print("  Evaluating PlannerAgent answers...")
    planner_dataset = EvaluationDataset.from_list(planner_eval_data)
    planner_ragas = ragas_evaluate(dataset=planner_dataset, metrics=metrics)
    planner_df = planner_ragas.to_pandas()

    # Attach RAGAS scores to results
    metric_cols = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference", "context_utilization"]
    for i, r in enumerate(results):
        r["gemini"]["ragas"] = {}
        r["planner"]["ragas"] = {}
        for col in metric_cols:
            if col in gemini_df.columns:
                r["gemini"]["ragas"][col] = safe_float(gemini_df.iloc[i].get(col))
            if col in planner_df.columns:
                r["planner"]["ragas"][col] = safe_float(planner_df.iloc[i].get(col))

    # ========== PHASE 3: Print Report ==========
    print("\n\n" + "=" * 70)
    print("BENCHMARK REPORT: Gemini (hybrid) vs PlannerAgent")
    print("=" * 70)

    total = len(results)
    g_pass = sum(1 for r in results if r["gemini"]["success"])
    p_pass = sum(1 for r in results if r["planner"]["success"])
    g_total_time = sum(r["gemini"]["time_s"] for r in results)
    p_total_time = sum(r["planner"]["time_s"] for r in results)

    # Compute average RAGAS scores
    def avg_ragas(results_list, method, metric):
        vals = [r[method]["ragas"].get(metric, 0) for r in results_list if r[method].get("ragas")]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    print(f"\n{'Method':<20} {'Pass':>6} {'Fail':>6} {'Time':>8}  {'Faith':>7} {'AnsRel':>7} {'CtxPre':>7} {'CtxUti':>7}")
    print("-" * 85)
    print(
        f"{'Gemini (hybrid)':<20} {g_pass:>6} {total - g_pass:>6} {g_total_time:>7.0f}s"
        f"  {avg_ragas(results, 'gemini', 'faithfulness'):>7.4f}"
        f" {avg_ragas(results, 'gemini', 'answer_relevancy'):>7.4f}"
        f" {avg_ragas(results, 'gemini', 'llm_context_precision_without_reference'):>7.4f}"
        f" {avg_ragas(results, 'gemini', 'context_utilization'):>7.4f}"
    )
    print(
        f"{'PlannerAgent':<20} {p_pass:>6} {total - p_pass:>6} {p_total_time:>7.0f}s"
        f"  {avg_ragas(results, 'planner', 'faithfulness'):>7.4f}"
        f" {avg_ragas(results, 'planner', 'answer_relevancy'):>7.4f}"
        f" {avg_ragas(results, 'planner', 'llm_context_precision_without_reference'):>7.4f}"
        f" {avg_ragas(results, 'planner', 'context_utilization'):>7.4f}"
    )

    # Per-query table
    print(f"\n{'#':<4} {'G':>4} {'P':>4} {'G_Faith':>8} {'P_Faith':>8} {'G_AnsR':>7} {'P_AnsR':>7}  Query")
    print("-" * 90)
    for r in results:
        g_s = "OK" if r["gemini"]["success"] else "--"
        p_s = "OK" if r["planner"]["success"] else "--"
        gf = r["gemini"].get("ragas", {}).get("faithfulness", 0)
        pf = r["planner"].get("ragas", {}).get("faithfulness", 0)
        ga = r["gemini"].get("ragas", {}).get("answer_relevancy", 0)
        pa = r["planner"].get("ragas", {}).get("answer_relevancy", 0)
        print(f"{r['query_num']:<4} {g_s:>4} {p_s:>4} {gf:>8.4f} {pf:>8.4f} {ga:>7.4f} {pa:>7.4f}  {r['query'][:40]}")

    # Queries where they differ
    diffs = [r for r in results if r["gemini"]["success"] != r["planner"]["success"]]
    if diffs:
        print(f"\nQueries where PASS/FAIL DIFFER ({len(diffs)}):")
        for r in diffs:
            g_s = "PASS" if r["gemini"]["success"] else "FAIL"
            p_s = "PASS" if r["planner"]["success"] else "FAIL"
            print(f"  Q{r['query_num']}: Gemini={g_s}, Planner={p_s} -- {r['query'][:60]}")

    # ========== PHASE 4: Save Reports ==========

    # Remove contexts from JSON report to keep it manageable
    json_results = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "contexts"}
        json_results.append(entry)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report saved to {REPORT_FILE}")

    # CSV export
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_num", "query",
            "gemini_pass", "gemini_time_s",
            "gemini_faithfulness", "gemini_answer_relevancy",
            "gemini_llm_context_precision_without_reference", "gemini_context_utilization",
            "planner_pass", "planner_time_s",
            "planner_faithfulness", "planner_answer_relevancy",
            "planner_llm_context_precision_without_reference", "planner_context_utilization",
            "planner_num_subtasks",
        ])
        for r in results:
            gr = r["gemini"].get("ragas", {})
            pr = r["planner"].get("ragas", {})
            writer.writerow([
                r["query_num"], r["query"],
                r["gemini"]["success"], r["gemini"]["time_s"],
                gr.get("faithfulness", ""), gr.get("answer_relevancy", ""),
                gr.get("llm_context_precision_without_reference", ""), gr.get("context_utilization", ""),
                r["planner"]["success"], r["planner"]["time_s"],
                pr.get("faithfulness", ""), pr.get("answer_relevancy", ""),
                pr.get("llm_context_precision_without_reference", ""), pr.get("context_utilization", ""),
                len(r["planner"].get("subtasks", [])),
            ])
    print(f"CSV report saved to {CSV_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
