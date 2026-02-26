# Documentation RAG with MCP Server

A Model Context Protocol (MCP) server that enables semantic search over documentation files using ChromaDB and sentence transformers. Integrates seamlessly with Claude Desktop for AI-powered documentation queries.

## ✨ Features

- 🔍 Semantic search across documentation files (Markdown, TXT, RST, HTML)
- 🤖 MCP server integration with Claude Desktop
- 📚 ChromaDB vector database for efficient retrieval
- 🎯 Configurable chunking and embedding strategies
- 🛠️ Command-line tools for document ingestion and querying
- 📊 Relevance scoring for search results

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Conda or virtualenv
- Claude Desktop (for MCP integration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mountofsorrow/documentation-rag-mcp.git
cd documentation-rag-mcp
```

2. Create and activate conda environment:
```bash
conda create -n docrag python=3.10
conda activate docrag
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Add Documentation Files

Place your documentation files in the `docs/` folder:
```
docs/
├── python.txt
├── api-reference.md
└── user-guide.md
```

#### 2. Ingest Documents
```bash
python ingest.py
```

To clear existing data and re-ingest:
```bash
python ingest.py --clear
```

#### 3. Test Queries (Optional)
```bash
python query.py
```

#### 4. Configure Claude Desktop

Edit your Claude Desktop config file:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**macOS/Linux:** `~/Library/Application Support/Claude/claude_desktop_config.json`
```json
{
  "mcpServers": {
    "documentation-rag": {
      "command": "/path/to/python",
      "args": ["/path/to/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop and start asking questions about your documentation!

## 📁 Project Structure
```
documentation-rag-mcp/
├── docs/                  # Your documentation files
├── chroma_db/            # Vector database storage (auto-generated)
├── config.py             # Shared configuration
├── ingest.py             # Document ingestion script
├── query.py              # Standalone query tool
├── mcp_server.py         # MCP server for Claude Desktop
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

- `CHUNK_SIZE`: Size of text chunks (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150)
- `EMBEDDING_MODEL`: Sentence transformer model (default: BAAI/bge-small-en-v1.5)
- `DEFAULT_TOP_K`: Number of results to return (default: 3)

## 🔧 Supported File Types

- Markdown (`.md`)
