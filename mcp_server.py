import anyio
import chromadb
import logging
from sentence_transformers import SentenceTransformer
from mcp.server import Server
from mcp.types import Tool
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions, ServerCapabilities
from config import PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, DEFAULT_TOP_K


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Documentation-Rag")

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
except Exception as e:
    logger.error(f"Failed to load Embedding model: {e}")
    raise

try:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
except Exception as e:
    logger.error(f"Failed to Connect to ChromaDB: {e}")
    raise

# MCP SERVER
server = Server("Documentation-Rag")

@server.list_tools()
async def list_tools():
    """List available tools for the MCP server."""
    return [
        Tool(
            name="search_docs",
            description=(
                "Search the documentation and return relevant sections. "
                "This tool uses semantic search to find the most relevant "
                "documentation chunks based on your query."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3, max: 10)",
                        "default": DEFAULT_TOP_K,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name, arguments):
    """Handle tool calls from the MCP client."""
    if name != "search_docs":
        error_msg = f"Unknown tool: {name}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    query = arguments.get("query", "").strip()
    top_k = arguments.get("top_k", DEFAULT_TOP_K)

    if not query:
        error_msg = "Query cannot be Empty!"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        query_embedding = embedding_model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            print("No results found for query")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No relevant documentation found for your query."
                    }
                ]
            }

        # Format results
        output = f"Found {len(documents)} Relevant Sections:\n"
        output += "=" * 60 + "\n"

        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            output += f"\n📄 Result {i}\n"
            output += f"Source: {meta['source']}\n"

            if distances and i - 1 < len(distances):
                relevance = 1 - distances[i - 1]
                output += f"Relevance: {relevance:.2%}\n"

            output += "-" * 60 + "\n"
            output += doc + "\n\n"

        return {
            "content": [
                {
                    "type": "text",
                    "text": output.strip()
                }
            ]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"❌ Error: {str(e)}"}],
            "isError": True
        }


async def main():
    """Main entry point for the MCP server."""
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                initialization_options=InitializationOptions(
                    server_name="documentation-rag",
                    server_version="1.0.0",
                    capabilities=ServerCapabilities(
                    tools=None
                    )
                )
            )
    except Exception as e:
        logger.error(f"Server Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        anyio.run(main)
    except Exception as e:
        logger.error(f"Fatal Error: {e}", exc_info=True)
        raise