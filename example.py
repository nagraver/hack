import logging
import os
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tg-store")


async def parse_document(file_path: str | Path) -> list[Document]:
    """Parse a document from the given file path."""
    logger.info("Parsing document from file: %s", file_path)
    docs = await DoclingLoader(file_path=str(file_path)).aload()
    logger.info("Document parsing completed.")
    return docs


def configure_qdrant() -> tuple[QdrantClient, QdrantVectorStore]:
    """Configure and return a Qdrant client and vector store."""
    logger.info("Configuring Qdrant client and vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={'truncation': True, 'max_length': 512}
    )
    embeddings.embed_query("test")  # Warm up the model

    client = QdrantClient(path='/tmp/langchain_qdrant')

    # Create collection if it doesn't exist
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    except ValueError:
        logger.info("Collection already exists. Skipping creation.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    logger.info("Qdrant configuration completed.")
    return client, vector_store


async def update_store(vector_store: QdrantVectorStore, docs: list[Document]) -> dict[str, Document]:
    """Update the vector store with the given documents."""
    logger.info("Splitting documents into chunks...")
    splitter = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=512,
        chunk_overlap=50,
        model_name="BAAI/bge-m3"
    )
    splitted_docs = splitter.split_documents(docs)
    logger.info("Document splitting completed.")

    ids = [str(uuid4()) for _ in splitted_docs]
    logger.info("Adding documents to vector store...")
    await vector_store.aadd_documents(documents=splitted_docs, ids=ids)
    logger.info("Documents successfully added to vector store.")

    return {ids[i]: splitted_docs[i] for i in range(len(ids))}


def configure_retriever(
    vector_store: QdrantVectorStore,
    search_type: str = "mmr",
    search_kwargs: dict[str, Any] = {"k": 5}
):
    """Configure and return a retriever for the vector store."""
    logger.info("Configuring retriever with search type: %s and search kwargs: %s", search_type, search_kwargs)
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
