import chromadb
from chromadb import QueryResult
from sentence_transformers import SentenceTransformer
import logging
from config import PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, DEFAULT_TOP_K


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

try:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

except Exception as e:
    logger.error(f"Failed to connect to ChromaDB: {e}")
    raise


def search(query: str, top_k: int = DEFAULT_TOP_K) -> QueryResult:
    """
    Search the documentation collection for relevant chunks.
    
    Args:
        query: The search query string
        top_k: Number of results to return (default from config)
        
    Returns: Dictionary containing search results from ChromaDB
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    return results


def format_results(results) -> str:
    """
    Args: results: Raw results from ChromaDB query
    Returns: Formatted string representation of results
    """

    if not results["documents"][0]:
        return "No results found."

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = ["\n" + "=" * 30, "SEARCH RESULTS", "=" * 30 + "\n"]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        # Header with source & score
        output.append(f"[{i + 1}] {meta['source']} (score: {1 - dist:.3f})")
        output.append("=" * 60)

        # Content - simple truncation
        display_text = doc[:500] + "..." if len(doc) > 500 else doc
        output.append(display_text)
        output.append("=" * 60 + "\n")

    return "\n".join(output)


def interactive_query():
    """Run an interactive query loop."""
    print("\n📚 Documentation Search")
    print("Type 'exit' or 'quit' to stop\n")
    
    while True:
        try:
            question = input("🔍 Ask a question: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            results = search(question)
            print(format_results(results))
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    interactive_query()