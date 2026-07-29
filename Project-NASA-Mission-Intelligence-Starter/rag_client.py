import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories (directories whose name contains "chroma")
    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and "chroma" in d.name.lower()
    ]

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            # Loop through each collection found
            for collection in collections:
                # Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}::{collection.name}"

                # Get document count with fallback for unsupported operations
                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = "unknown"

                # Build information dictionary
                backends[key] = {
                    "chroma_dir": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{collection.name} ({chroma_dir.name}) - {doc_count} docs",
                    "document_count": doc_count,
                }

        except Exception as e:
            # Handle connection or access errors gracefully
            key = f"{chroma_dir.name}::error"
            error_msg = str(e)
            if len(error_msg) > 50:
                error_msg = error_msg[:50] + "..."
            backends[key] = {
                "chroma_dir": str(chroma_dir),
                "collection_name": "unknown",
                "display_name": f"{chroma_dir.name} (error: {error_msg})",
                "document_count": "unknown",
            }

    # Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # Create a chromadb PersistentClient
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    # Return the collection with the collection_name
    return client.get_collection(name=collection_name)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    where_filter = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() != "all":
        # Create filter dictionary with appropriate field-value pairs
        where_filter = {"mission": mission_filter}

    # Execute database query
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    # Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # Initialize list with header text for context section
    context_parts = ["Relevant NASA mission documents:"]

    # Loop through paired documents and their metadata using enumeration
    for i, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        metadata = metadata or {}

        # Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "unknown")
        # Clean up mission name formatting (replace underscores, capitalize)
        mission_display = mission.replace("_", " ").title()

        # Extract source information from metadata with fallback value
        source = metadata.get("source", "unknown")

        # Extract category information from metadata with fallback value
        category = metadata.get("document_category", "unknown")
        # Clean up category name formatting (replace underscores, capitalize)
        category_display = category.replace("_", " ").title()

        # Create formatted source header with index number and extracted information
        source_header = (
            f"\n[{i}] Mission: {mission_display} | "
            f"Source: {source} | Category: {category_display}"
        )
        # Add source header to context parts list
        context_parts.append(source_header)

        # Check document length and truncate if necessary
        max_length = 1000
        if len(document) > max_length:
            content = document[:max_length] + "..."
        else:
            content = document
        # Add truncated or full document content to context parts list
        context_parts.append(content)

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)