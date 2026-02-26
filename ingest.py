import logging
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pathlib import Path
import re

from config import (
    DOCS_PATH, PERSIST_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Header-aware chunking:
    1. Split by markdown headers (#, ##, ###, etc.)
    2. If section exceeds chunk_size, split with overlap
    """

    if not text.strip():
        return []

    # Split at markdown headers but keep the header in the chunk
    sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # If section is small enough, keep as-is
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            # Fallback to size-based chunking inside the section
            start = 0
            while start < len(section):
                end = start + chunk_size
                chunk = section[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += chunk_size - overlap

    return chunks

def load_documents() -> List[Dict[str, str]]:
    """
    Load & chunk all documents from a selected subfolder inside DOCS_PATH.
    Prompts the user to select which folder to process.
    """
    documents = []
    docs_path = Path(DOCS_PATH)

    if not docs_path.exists():
        logger.error(f"Docs path does not exist: {DOCS_PATH}")
        return documents

    # List subfolders
    subfolders = [f for f in docs_path.iterdir() if f.is_dir()]
    if not subfolders:
        logger.error(f"No subfolders found in {DOCS_PATH}")
        return documents

    # Prompt user to pick a folder
    print("Available Document Folders:")
    for idx, folder in enumerate(subfolders, 1):
        print(f"{idx}: {folder.name}")

    while True:
        choice = input("Enter the number of the folder to process: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        choice = int(choice)
        if 1 <= choice <= len(subfolders):
            selected_folder = subfolders[choice - 1]
            break
        else:
            print("Number out of range, Try again.")

    logger.info(f"Processing folder: {selected_folder.name}")

    # Only process .md files in that folder
    files = [
        f for f in selected_folder.iterdir()
        if f.suffix == ".md" and "_sources" not in f.name
    ]

    logger.info(f"Found {len(files)} Markdown documents to process")

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                logger.warning(f"Skipping empty file: {filepath.name}")
                continue

            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            logger.info(f"Chunked {filepath.name} into {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                documents.append({
                    "id": f"{filepath.name}_{i}",
                    "text": chunk,
                    "source": filepath.name
                })

        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}")
            continue

    return documents

def ingest(clear_existing: bool = False):
    """
    Ingest documents into ChromaDB.
    Args: clear_existing: If True, clear existing collection before ingesting
    """
    global collection

    if clear_existing:
        client.delete_collection(name=COLLECTION_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents = load_documents()

    if not documents:
        print("No Documents Found!")
        return

    texts = [doc["text"] for doc in documents]
    embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()
    ids = [doc["id"] for doc in documents]
    metadatas = [{"source": doc["source"]} for doc in documents]

    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    logger.info(f"✓ Successfully Ingested {len(documents)} Chunks from {len(set(d['source'] for d in documents))} Files")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--clear", action="store_true", help="Clear existing collection before ingesting")
    args = parser.parse_args()

    ingest(clear_existing=args.clear)