import os
import asyncio
import nest_asyncio
import numpy as np
import fitz

from lightrag import LightRAG
from lightrag.agentic_rag import AgenticRAG
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

nest_asyncio.apply()

# Constants
WORKING_DIR = "./rag_storage_planner"
PDF_FILE = "./MLBOOK.pdf"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


# --- HELPERS ---

def read_pdf_text(pdf_path):
    """Extracts text from all pages of a PDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
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
        model="models/gemini-embedding-001",
    )


# --- MAIN ---

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

    # --- PERSISTENCE CHECK ---
    is_indexed = os.path.exists(os.path.join(WORKING_DIR, "vdb_chunks.json"))

    if not is_indexed:
        if not os.path.exists(PDF_FILE):
            print(f"PDF file {PDF_FILE} not found!")
            return

        print(f"Index not found. Extracting text from {PDF_FILE}...")
        content = read_pdf_text(PDF_FILE)

        if content:
            print(f"Indexing {len(content)} characters into LightRAG...")
            await rag.ainsert(content)
            print("Indexing complete.")
    else:
        print("Found existing index. Skipping re-indexing.")

    # Initialize AgenticRAG
    agent = AgenticRAG(rag)

    complex_query =("Find the methodology that relies on a 'subset of the power set of all possible instances' to define its hypothesis space. Explain why this specific methodology would be computationally infeasible for the type of signal processing tasks described in the neural network chapters.")

    print(f"\nQuery: {complex_query}\n")
    answer = await agent.run(complex_query)

    # --- Show grader results per sub-task ---
    if agent.state and agent.state.tasks:
        print("=" * 70)
        print("GRADER RESULTS")
        print("=" * 70)
        for tid, task in agent.state.tasks.items():
            status = "PASS" if task.status == "completed" else "FAIL"
            print(f"\n  [{task.id}] {status} | mode={task.mode} | attempts={task.grader_attempts}")
            print(f"  Query: {task.query}")
            for gh in task.grader_history:
                g = gh.get("grade")
                if g:
                    verdict = "PASS" if g["passed"] else "FAIL"
                    print(f"    Attempt {gh['attempt']}: {verdict} | "
                          f"R={g['relevance']:.2f} C={g['completeness']:.2f} S={g['sufficiency']:.2f}")
                    print(f"      Data: {gh['data_summary']}")

        # --- Show final answer grade ---
        fg = agent.state.final_grade
        if fg:
            print(f"\n  [FINAL GRADE] {'PASS' if fg['passed'] else 'FAIL'}")
            if "relevance" in fg:
                print(f"    R={fg['relevance']:.2f} C={fg['completeness']:.2f} S={fg['sufficiency']:.2f}")
                print(f"    Data: {fg['data_summary']}")
                print(f"    Reasoning: {fg['reasoning'][:200]}")
        print("=" * 70)

    # --- Final answer ---
    print("\n--- FINAL SYNTHESIZED ANSWER ---")
    print(answer)
    print("--------------------------------")


if __name__ == "__main__":
    asyncio.run(main())
