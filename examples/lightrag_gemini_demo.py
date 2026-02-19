# # import os
# import asyncio
# import nest_asyncio
# import numpy as np
# import shutil  # <--- Required to delete old data

# from lightrag import LightRAG, QueryParam
# from lightrag.llm.gemini import gemini_model_complete, gemini_embed
# from lightrag.utils import wrap_embedding_func_with_attrs

# nest_asyncio.apply()

# WORKING_DIR = "./rag_storage"
# BOOK_FILE = "./book.txt"

# # Validate API key
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError(
#         "GEMINI_API_KEY environment variable is not set. "
#         "Please set it with: export GEMINI_API_KEY='your-api-key'"
#     )

# # --------------------------------------------------
# # LLM function
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


# # --------------------------------------------------
# # Embedding function
# # --------------------------------------------------
# @wrap_embedding_func_with_attrs(
#     embedding_dim=3072,  # Correct dimension for gemini-embedding-001
#     max_token_size=2048,
#     model_name="models/gemini-embedding-001",
# )
# # FIX: Added **kwargs to prevent "unexpected keyword argument" errors
# async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
#     return await gemini_embed.func(
#         texts,
#         api_key=GEMINI_API_KEY,
#         model="models/gemini-embedding-001"
#     )


# # --------------------------------------------------
# # Initialize RAG
# # --------------------------------------------------
# async def initialize_rag():
#     rag = LightRAG(
#         working_dir=WORKING_DIR,
#         llm_model_func=llm_model_func,
#         embedding_func=embedding_func,
#         llm_model_name="gemini-2.0-flash",
#     )

#     await rag.initialize_storages()
#     return rag


# # --------------------------------------------------
# # Main
# # --------------------------------------------------
# def main():
#     # --- CLEANUP STEP (CRITICAL) ---
#     # This deletes the old corrupted index so LightRAG re-processes the book.
#     if os.path.exists(WORKING_DIR):
#         print(f"🧹 Cleaning up existing storage at {WORKING_DIR}...")
#         shutil.rmtree(WORKING_DIR)

#     os.mkdir(WORKING_DIR)

#     # Validate book file exists
#     if not os.path.exists(BOOK_FILE):
#         # Create a dummy file if it doesn't exist so the script runs
#         print(f"⚠️ '{BOOK_FILE}' not found. Creating a dummy file for testing.")
#         with open(BOOK_FILE, "w", encoding="utf-8") as f:
#             f.write("Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'.")

#     rag = asyncio.run(initialize_rag())

#     # Insert text
#     print("📖 Reading and indexing book content...")
#     with open(BOOK_FILE, "r", encoding="utf-8") as f:
#         rag.insert(f.read())
#     print("✅ Indexing complete!")

#     query = "What are the top themes?"

#     print("\nNaive Search:")
#     print(rag.query(query, param=QueryParam(mode="naive")))

#     print("\nLocal Search:")
#     print(rag.query(query, param=QueryParam(mode="local")))

#     print("\nGlobal Search:")
#     print(rag.query(query, param=QueryParam(mode="global")))

#     print("\nHybrid Search:")
#     print(rag.query(query, param=QueryParam(mode="hybrid")))


# if __name__ == "__main__":
#     main()
# import os
# import asyncio
# import nest_asyncio
# import numpy as np
# import shutil  # <--- Required to delete old data

# from lightrag import LightRAG, QueryParam
# from lightrag.llm.gemini import gemini_model_complete, gemini_embed
# from lightrag.utils import wrap_embedding_func_with_attrs

# nest_asyncio.apply()

# WORKING_DIR = "./rag_storage"
# BOOK_FILE = "./book.txt"

# # Validate API key
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError(
#         "GEMINI_API_KEY environment variable is not set. "
#         "Please set it with: export GEMINI_API_KEY='your-api-key'"
#     )

# # --------------------------------------------------
# # LLM function
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


# # --------------------------------------------------
# # Embedding function
# # --------------------------------------------------
# @wrap_embedding_func_with_attrs(
#     embedding_dim=3072,  # Correct dimension for gemini-embedding-001
#     max_token_size=2048,
#     model_name="models/gemini-embedding-001",
# )
# # FIX: Added **kwargs to prevent "unexpected keyword argument" errors
# async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
#     return await gemini_embed.func(
#         texts,
#         api_key=GEMINI_API_KEY,
#         model="models/gemini-embedding-001"
#     )


# # --------------------------------------------------
# # Initialize RAG
# # --------------------------------------------------
# async def initialize_rag():
#     rag = LightRAG(
#         working_dir=WORKING_DIR,
#         llm_model_func=llm_model_func,
#         embedding_func=embedding_func,
#         llm_model_name="gemini-2.0-flash",
#     )

#     await rag.initialize_storages()
#     return rag


# # --------------------------------------------------
# # Main
# # --------------------------------------------------
# def main():
#     # --- CLEANUP STEP (CRITICAL) ---
#     # This deletes the old corrupted index so LightRAG re-processes the book.
#     if os.path.exists(WORKING_DIR):
#         print(f"🧹 Cleaning up existing storage at {WORKING_DIR}...")
#         shutil.rmtree(WORKING_DIR)

#     os.mkdir(WORKING_DIR)

#     # Validate book file exists
#     if not os.path.exists(BOOK_FILE):
#         # Create a dummy file if it doesn't exist so the script runs
#         print(f"⚠️ '{BOOK_FILE}' not found. Creating a dummy file for testing.")
#         with open(BOOK_FILE, "w", encoding="utf-8") as f:
#             f.write("Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'.")

#     rag = asyncio.run(initialize_rag())

#     # Insert text
#     print("📖 Reading and indexing book content...")
#     with open(BOOK_FILE, "r", encoding="utf-8") as f:
#         rag.insert(f.read())
#     print("✅ Indexing complete!")

#     query = "What are the top themes?"

#     print("\nNaive Search:")
#     print(rag.query(query, param=QueryParam(mode="naive")))

#     print("\nLocal Search:")
#     print(rag.query(query, param=QueryParam(mode="local")))

#     print("\nGlobal Search:")
#     print(rag.query(query, param=QueryParam(mode="global")))

#     print("\nHybrid Search:")
#     print(rag.query(query, param=QueryParam(mode="hybrid")))


# if __name__ == "__main__":
#     main()
import os
import asyncio
import nest_asyncio
import numpy as np
import shutil
import fitz  # <--- PyMuPDF for reading PDFs

from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

nest_asyncio.apply()

WORKING_DIR = "./rag_storage"
PDF_FILE = "./MLBOOK.pdf"  # <--- Your PDF file path

# Validate API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Please set it with: export GEMINI_API_KEY='your-api-key'"
    )

# --------------------------------------------------
# LLM function
# --------------------------------------------------
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.0-flash",
        **kwargs,
    )


# --------------------------------------------------
# Embedding function
# --------------------------------------------------
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


# --------------------------------------------------
# Helper: Read PDF
# --------------------------------------------------
def read_pdf_text(pdf_path):
    """Extracts all text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        # Iterate through pages and append text
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return None


# --------------------------------------------------
# Initialize RAG
# --------------------------------------------------
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        llm_model_name="gemini-2.0-flash",
    )
    await rag.initialize_storages()
    return rag


# --------------------------------------------------
# Main
# --------------------------------------------------
async def main():
    # --- CLEANUP STEP ---
    # if os.path.exists(WORKING_DIR):
    #     print(f"🧹 Cleaning up existing storage at {WORKING_DIR}...")
    #     shutil.rmtree(WORKING_DIR)

    #os.mkdir(WORKING_DIR)

    # Validate PDF file exists
    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(f"'{PDF_FILE}' not found. Please place your PDF in the folder.")

    rag = await initialize_rag()

    # --- READ PDF CONTENT ---
    print(f"📖 Extracting text from {PDF_FILE}...")
    pdf_content = read_pdf_text(PDF_FILE)

    if not pdf_content:
        print("❌ Failed to extract text from PDF. Exiting.")
        return

    print(f"✅ Extracted {len(pdf_content)} characters.")
    print("⏳ Indexing content into LightRAG (this may take a moment)...")

    # Insert text into RAG
    rag.insert(pdf_content)
    print("✅ Indexing complete!")

    # --- QUERY ---
    #query = "Find the methodology that relies on a 'subset of the power set of all possible instances' to define its hypothesis space. Explain why this specific methodology would be computationally infeasible for the type of signal processing tasks described in the neural network chapters."
   # query=""Find the 'convergence theorem' mentioned in the context of simplest linear learners. Now, find the 'PAC learning' bounds discussed in the computational theory section. Does the book provide a specific scenario where the PAC requirements are satisfied but the convergence theorem would still fail to reach a solution in finite time?"
    query="Identify the methodology used to resolve classification failures in datasets that are not separable by a single hyperplane without increasing the number of learning parameters. How does the text describe the process of projecting the input space into a higher-dimensional feature space, and why is this technically considered a 'fixed' rather than 'adaptive' transformation in the context of Φ-functions?"
    print(rag.query(query, param=QueryParam(mode="naive")))

    print("\nLocal Search:")
    print(rag.query(query, param=QueryParam(mode="local")))

    print("\nGlobal Search:")
    print(rag.query(query, param=QueryParam(mode="global")))

    print("\nHybrid Search:")
    print(rag.query(query, param=QueryParam(mode="hybrid")))


if __name__ == "__main__":
    asyncio.run(main())
