# LightRAG-Agent

**Agentic RAG pipeline built on top of [LightRAG](https://github.com/HKUDS/LightRAG)** — a graph-based Retrieval-Augmented Generation framework.

This fork extends LightRAG with multi-step agentic retrieval, answer grading with self-correction, and a Streamlit benchmark UI for comparing Standard LightRAG vs AgenticRAG.

---

## Architecture

```
Query
  │
  ▼
Classifier Agent ──► Simple? ──► Direct LightRAG query ──► Answer
  │
  ▼ Complex
Planner Agent
  │
  ▼
Sub-task DAG (dependency-aware)
  │
  ▼
Execute each sub-task via LightRAG
(local / global / hybrid / naive / reasoning)
  │
  ▼
LLM-generated sub-task summaries
  │
  ▼
Final Synthesis (merge all sub-task results)
  │
  ▼
Grader Agent (evaluate answer quality)
  │
  ├── PASS ──► Return answer
  └── FAIL ──► Re-plan with feedback (up to 3 attempts)
```

### Key Components

| Module | Description |
|---|---|
| `lightrag/agentic_rag.py` | `AgenticRAG` — full pipeline orchestrator (classify → plan → execute → grade → re-plan) |
| `lightrag/planner_agent.py` | `PlannerAgent` — decomposes complex queries into dependency-aware sub-task DAGs |
| `lightrag/grader_agent.py` | `GraderAgent` — evaluates answer quality (relevance, completeness, sufficiency) |
| `lightrag/prompt.py` | LLM prompts for classification, planning, grading, and synthesis |
| `examples/streamlit_benchmark.py` | Interactive Streamlit app for benchmarking Standard RAG vs AgenticRAG |
| `examples/benchmark_agentic_rag.py` | CLI benchmark script with RAGAS evaluation |
| `exploratory_data_analysis.py` | EDA done for project |
| `benchmark_agentic_ragas.csv` |  query results |
---

## Quick Start

### Prerequisites

```bash
# Clone the repo
git clone https://github.com/tarinip/LightRAG-Agent.git
cd LightRAG-Agent

# Install dependencies
uv sync
source .venv/bin/activate

# Install evaluation extras
uv sync --extra evaluation
pip install streamlit pymupdf langchain-google-genai
```

### Configuration

```bash
cp env.example .env
# Edit .env with your LLM/embedding API keys (e.g. GEMINI_API_KEY)
```

### Usage

#### Python API

```python
import asyncio
from lightrag import LightRAG
from lightrag.agentic_rag import AgenticRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()

    # Index documents
    await rag.ainsert("Your document text here...")

    # Run AgenticRAG
    agent = AgenticRAG(rag)
    result = await agent.run("What are the key differences between X and Y?")
    print(result)

    # Inspect sub-task results
    if agent.state:
        for task_id, task in agent.state.tasks.items():
            print(f"  [{task.id}] ({task.mode}): {task.result_summary}")

    await rag.finalize_storages()

asyncio.run(main())
```

#### Streamlit Benchmark App

```bash
streamlit run examples/streamlit_benchmark.py
```

Upload a PDF, enter questions, and compare Standard LightRAG (best of 4 retrieval modes) against AgenticRAG side-by-side with RAGAS evaluation metrics.

#### CLI Benchmark

```bash
python examples/benchmark_agentic_rag.py
```

---

## How It Works

### AgenticRAG Pipeline

1. **Classification** — LLM determines if the query is simple (direct retrieval) or complex (needs decomposition)
2. **Planning** — `PlannerAgent` breaks complex queries into a DAG of sub-tasks with explicit dependencies and retrieval modes
3. **Execution** — Sub-tasks execute in topological order; each runs a LightRAG query in its assigned mode (local, global, hybrid, naive, or reasoning)
4. **Summarization** — LLM generates 2-3 sentence summaries per sub-task, used as context for dependent tasks
5. **Synthesis** — All sub-task results are merged into a final answer via LLM
6. **Grading** — `GraderAgent` scores the answer on relevance, completeness, and sufficiency (0.0-1.0 each)
7. **Self-correction** — If the answer fails grading, the pipeline re-plans with grader feedback (up to 3 attempts)

### Grader Agent

The grader evaluates the **synthesized answer** (not raw retrieval), scoring:

- **Relevance** (>= 0.6 to pass) — Does the answer address the query?
- **Completeness** (>= 0.4 to pass) — Does it cover key aspects?
- **Sufficiency** (>= 0.4 to pass) — Is there enough depth?

On failure, the grader provides reasoning and a rewritten query to guide re-planning.

### RAGAS Evaluation

The benchmark tools evaluate with four RAGAS metrics:

- **Faithfulness** — Is the answer grounded in the retrieved context?
- **Answer Relevancy** — Does the answer address the question?


---

## Project Structure

```
lightrag/
├── agentic_rag.py          # AgenticRAG orchestrator
├── planner_agent.py        # Sub-task decomposition & DAG execution
├── grader_agent.py         # Answer quality grading
├── prompt.py               # All LLM prompts
├── lightrag.py             # Core LightRAG class (upstream)
├── operate.py              # Extraction & query operations (upstream)
├── kg/                     # Storage backends (upstream)
├── llm/                    # LLM provider bindings (upstream)
└── api/                    # FastAPI server + WebUI (upstream)

examples/
├── streamlit_benchmark.py  # Interactive benchmark UI
├── benchmark_agentic_rag.py # CLI benchmark with RAGAS
└── lightrag_gemini_demo.py # Gemini integration demo

exploratory_data_analysis.py # EDA for datasets
benchmark_agentic_ragas.csv  # Query results
```

---

## Upstream

This project is a fork of [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG). See the upstream repo for full documentation on the core LightRAG framework, storage backends, LLM providers, and the WebUI.
