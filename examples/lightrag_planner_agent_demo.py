# import os
# import asyncio
# import numpy as np
# from lightrag import LightRAG
# from lightrag.planner_agent import PlannerAgent
# from lightrag.llm.gemini import gemini_model_complete, gemini_embed
# from lightrag.utils import wrap_embedding_func_with_attrs

# # Constants
# WORKING_DIR = "./rag_storage_planner"
# BOOK_FILE = "./book.txt"
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY environment variable is not set.")

# if not os.path.exists(WORKING_DIR):
#     os.makedirs(WORKING_DIR)

# # --------------------------------------------------
# # LLM and Embedding Functions (Gemini)
# # --------------------------------------------------
# async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
#     return await gemini_model_complete(
#         prompt,
#         system_prompt=system_prompt,
#         history_messages=history_messages,
#         api_key=GEMINI_API_KEY,
#         model_name="gemini-2.0-flash",
#         **kwargs,
#     )

# @wrap_embedding_func_with_attrs(
#     embedding_dim=768,
#     max_token_size=2048,
#     model_name="models/text-embedding-004",
# )
# async def embedding_func(texts: list[str]) -> np.ndarray:
#     return await gemini_embed.func(
#         texts, api_key=GEMINI_API_KEY, model="models/text-embedding-004"
#     )

# # --------------------------------------------------
# # Main Demo
# # --------------------------------------------------
# async def main():
#     # Initialize LightRAG
#     rag = LightRAG(
#         working_dir=WORKING_DIR,
#         llm_model_func=llm_model_func,
#         embedding_func=embedding_func,
#         llm_model_name="gemini-2.0-flash",
#     )
#     await rag.initialize_storages()

#     # Ensure some data is indexed for the demo
#     if not os.path.exists(BOOK_FILE):
#         with open(BOOK_FILE, "w", encoding="utf-8") as f:
#             f.write("LightRAG is a framework for Retrieval-Augmented Generation. It supports naive, local, global, and hybrid search modes. "
#                     "The planner agent is an extension that uses query decomposition to handle complex questions. "
#                     "Gemini is a powerful LLM from Google used here for both text generation and embeddings.")

#     with open(BOOK_FILE, "r", encoding="utf-8") as f:
#         await rag.ainsert(f.read())

#     # Initialize Planner Agent from the dedicated module
#     agent = PlannerAgent(rag)

#     complex_query = "What is LightRAG, what modes does it support, and how does the Planner Agent improve its capabilities?"

#     print(f"\nProcessing Complex Query: {complex_query}")
#     answer = await agent.query(complex_query)

#     print("\n--- FINAL SYNTHESIZED ANSWER ---")
#     print(answer)
#     print("--------------------------------")

# if __name__ == "__main__":
#     asyncio.run(main())
import os
import asyncio
import nest_asyncio
import numpy as np
import shutil
import fitz

from lightrag import LightRAG
from lightrag.planner_agent import PlannerAgent
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

nest_asyncio.apply()

# Constants
WORKING_DIR = "./rag_storage_planner"
PDF_FILE = "./MLBOOK.pdf"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# if os.path.exists(WORKING_DIR):
#         print(f"🧹 Cleaning up existing storage at {WORKING_DIR}...")
#         shutil.rmtree(WORKING_DIR)
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
        print(f"❌ Error reading PDF: {e}")
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
        model="models/gemini-embedding-001"
    )

# --- MAIN ---

async def main():
    # Ensure working directory exists
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    # Initialize LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        llm_model_name="gemini-2.0-flash",
        llm_model_kwargs={"temperature": 0, "seed": 42}
    )
    await rag.initialize_storages()

    # --- PERSISTENCE CHECK ---
    # We check for 'vdb_chunks.json' to see if indexing has already been done
    is_indexed = os.path.exists(os.path.join(WORKING_DIR, "vdb_chunks.json"))

    if not is_indexed:
        if not os.path.exists(PDF_FILE):
            print(f"❌ PDF file {PDF_FILE} not found!")
            return

        print(f"📂 Index not found. Extracting text from {PDF_FILE}...")
        content = read_pdf_text(PDF_FILE)

        if content:
            print(f"🚀 Indexing {len(content)} characters into LightRAG...")
            await rag.ainsert(content)
            print("✅ Indexing complete.")
    else:
        print("💾 Found existing index. Skipping PDF loading and re-indexing.")

    # Initialize Planner Agent
    agent = PlannerAgent(rag)

    #complex_query = "The author discusses a 'boundary' that separates successful classification from error. Find a technique in the book where this boundary is not a static line or curve, but is instead represented by a collection of logical rules. How does the 'search' through these rules differ from the 'search' performed by gradient descent?"
    complex_query= "Examine the 'Knowledge-Based Artificial Neural Network' (KBANN). How does the system 'map' a set of 'IF-THEN' rules into a network of weights and biases, and what happens to those 'hand-coded' rules as the system starts to see new empirical data?"
    answer = await agent.run(complex_query)

    print("\n--- FINAL SYNTHESIZED ANSWER ---")
    print(answer)
    print("--------------------------------")

if __name__ == "__main__":
    asyncio.run(main())