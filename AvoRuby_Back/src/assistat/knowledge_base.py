from __future__ import annotations
import os
import sys
import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ── Ensure config is importable when running standalone ───────────────────────
_this_dir = Path(__file__).resolve().parent        # assistat/
_src_dir = _this_dir.parent                        # src/
_back_dir = _src_dir.parent                        # AvoRuby_Back/
_project_root = _back_dir.parent                   # AvoRuby_Cash_Project/

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AvoRuby_Back.src.config import (
    DATA_AGRICOLE_PATH,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
)

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("AvoRubyKB")


#  CORE FUNCTIONS
def _load_documents(data_dir: str | None = None) -> list:
    """Load all .txt files from the data_agricole directory."""
    data_dir = data_dir or DATA_AGRICOLE_PATH

    if not os.path.isdir(data_dir):
        logger.error(f" Directory not found: {data_dir}")
        return []

    logger.info(f" Loading .txt documents from: {data_dir}")
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    logger.info(f"{len(docs)} document(s) loaded.")
    return docs


def _get_embeddings() -> OllamaEmbeddings:
    """Create Ollama embeddings client using nomic-embed-text."""
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def _split_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """Split documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunk(s).")
    return chunks


def build_vectorstore(force_rebuild: bool = True) -> Chroma:
    """
    Build (or rebuild) the ChromaDB vector store from data_agricole.
    
    Args:
        force_rebuild: If True, deletes existing DB and rebuilds from scratch.
    
    Returns:
        A Chroma vectorstore instance ready for retrieval.
    """
    embeddings = _get_embeddings()
    db_path = CHROMA_DB_PATH

    # Delete existing DB if force rebuilding
    if force_rebuild and os.path.isdir(db_path):
        import shutil
        logger.info(f"Deleting existing ChromaDB at: {db_path}")
        shutil.rmtree(db_path)

    # Load and chunk documents
    docs = _load_documents()
    if not docs:
        logger.warning("No documents found. Creating empty vector store.")
        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
        )

    chunks = _split_documents(docs)

    # Build ChromaDB
    logger.info(f"Building ChromaDB at: {db_path}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    logger.info(f"ChromaDB built successfully with {len(chunks)} chunks!")
    return vectorstore


def get_or_build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    Get the existing ChromaDB vector store, or build it if it doesn't exist.
    
    This is the main entry point used by the RAG engine.
    
    Args:
        force_rebuild: If True, always rebuild from scratch.
    
    Returns:
        A Chroma vectorstore instance ready for retrieval.
    """
    embeddings = _get_embeddings()
    db_path = CHROMA_DB_PATH

    # Check if DB already exists
    db_exists = (
        os.path.isdir(db_path)
        and any(Path(db_path).iterdir())
    )

    if db_exists and not force_rebuild:
        logger.info(f"Loading existing ChromaDB from: {db_path}")
        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
        )

    # Build from scratch
    return build_vectorstore(force_rebuild=True)


#  STANDALONE EXECUTION — Run this file to (re)build the knowledge base
if __name__ == "__main__":
    print("=" * 60)
    print("  AvoRuby Cash — Knowledge Base Builder")
    print("=" * 60)
    print(f"  Data source : {DATA_AGRICOLE_PATH}")
    print(f"  ChromaDB    : {CHROMA_DB_PATH}")
    print(f"  Embeddings  : {OLLAMA_EMBEDDING_MODEL}")
    print(f"  Ollama URL  : {OLLAMA_BASE_URL}")
    print("=" * 60)

    vs = build_vectorstore(force_rebuild=True)

    # Quick sanity check
    collection = vs._collection
    count = collection.count()
    print(f"\n🎉 Done! {count} vectors stored in ChromaDB.")
    print("You can now run the Streamlit app with: streamlit run app.py")
