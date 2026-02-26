import os
import logging

# Suppress transformers and sentence-transformers verbose output
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "docs")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# CHROMA CONFIG
COLLECTION_NAME = "library_docs"

# EMBEDDING CONFIG
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# CHUNKING CONFIG
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# QUERY CONFIG
DEFAULT_TOP_K = 3