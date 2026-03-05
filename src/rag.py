"""
rag.py
------
The RAG (Retrieval Augmented Generation) pipeline.

When the user asks a question, this module:
  1. RETRIEVES the most relevant document chunks from the vector store
  2. AUGMENTS the prompt with that context
  3. GENERATES a streamed answer using Claude claude-opus-4-6

This is the core of the project — the other modules feed into this one.
"""

import os
from typing import Generator, List, Dict

import anthropic
from dotenv import load_dotenv

from src.vectorstore import SalesforceVectorStore
from src.ingestion import load_all_documents, DEFAULT_SALESFORCE_URLS

# Load ANTHROPIC_API_KEY from .env file if present
load_dotenv()

# System prompt — tells Claude its role and how to use the retrieved context
SYSTEM_PROMPT = """You are a knowledgeable Salesforce assistant. Your job is to answer
questions about Salesforce using the documentation excerpts provided to you.

Guidelines:
- Base your answers primarily on the provided context.
- If the context does not contain enough information, say so honestly and share
  what general Salesforce knowledge you do have.
- Be concise but thorough. Use bullet points or numbered lists when helpful.
- When referencing specific features or terms, briefly explain them so beginners
  can follow along.
- If asked about something unrelated to Salesforce, politely redirect.
"""


class SalesforceRAG:
    """
    End-to-end RAG pipeline for answering Salesforce questions.

    Usage:
        rag = SalesforceRAG()
        rag.load_documents()                  # ingest & embed docs (once)
        for token in rag.query("What is SOQL?"):
            print(token, end="", flush=True)
    """

    def __init__(self, persist_dir: str = "./vectorstore"):
        """
        Set up the vector store and Anthropic client.

        Args:
            persist_dir: Where ChromaDB stores its data between runs.
        """
        self.vector_store = SalesforceVectorStore(persist_dir=persist_dir)
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    def load_documents(
        self,
        urls: List[str] = None,
        data_dir: str = "data",
        force_reload: bool = False,
    ) -> int:
        """
        Ingest Salesforce documentation into the vector store.

        Skips ingestion if documents are already loaded (unless force_reload=True).

        Args:
            urls:         URLs to scrape (defaults to DEFAULT_SALESFORCE_URLS).
            data_dir:     Directory for local .txt files.
            force_reload: If True, clears existing data and re-ingests everything.

        Returns:
            Number of chunks stored.
        """
        if not self.vector_store.is_empty() and not force_reload:
            count = self.vector_store.collection.count()
            print(f"Vector store already has {count} chunks. Skipping ingestion.")
            return count

        if force_reload:
            self.vector_store.reset()

        chunks = load_all_documents(urls=urls, data_dir=data_dir)
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def _build_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into a context string for the prompt.

        Args:
            retrieved_chunks: Output from vector_store.search().

        Returns:
            A formatted string listing each chunk and its source.
        """
        if not retrieved_chunks:
            return "No relevant documentation found."

        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            parts.append(
                f"[Excerpt {i}] (Source: {chunk['source']})\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def query(self, question: str, n_context_chunks: int = 5) -> Generator[str, None, None]:
        """
        Answer a Salesforce question using RAG with streaming output.

        Yields text tokens as Claude generates them — perfect for streaming
        to a UI without waiting for the full response.

        Args:
            question:        The user's question.
            n_context_chunks: How many document chunks to retrieve as context.

        Yields:
            String tokens from Claude's response.
        """
        # Step 1: Retrieve relevant chunks from the vector store
        relevant_chunks = self.vector_store.search(question, n_results=n_context_chunks)

        # Step 2: Build an augmented prompt that includes the retrieved context
        context = self._build_context(relevant_chunks)

        user_message = f"""Here is relevant Salesforce documentation to help answer the question:

{context}

---

Question: {question}

Please answer based on the documentation above."""

        # Step 3: Stream Claude's response token by token
        # Using claude-opus-4-6 with adaptive thinking for high-quality answers
        with self.client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1024,
            thinking={"type": "adaptive"},  # let Claude think when needed
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    def get_sources(self, question: str, n_results: int = 5) -> List[Dict]:
        """
        Return the source chunks used to answer a question (without generating an answer).
        Useful for debugging or displaying citations in the UI.

        Args:
            question:  The user's question.
            n_results: Number of chunks to retrieve.

        Returns:
            List of {"text", "source", "score"} dicts.
        """
        return self.vector_store.search(question, n_results=n_results)
