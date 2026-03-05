"""
vectorstore.py
--------------
A lightweight vector store built on numpy + sentence-transformers.

Replaces ChromaDB for Python 3.14 compatibility, and is easier to
understand — making it great for a portfolio/learning project.

How it works:
  - Documents are embedded into float vectors using sentence-transformers
  - Vectors and metadata are saved to disk as .npy and .json files
  - At query time, cosine similarity is computed between the query vector
    and all stored document vectors to find the most relevant chunks
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict


# "all-MiniLM-L6-v2" is small (~80 MB), fast on CPU, and produces
# 384-dimensional embeddings that work well for semantic search.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # output size of the model above


class SalesforceVectorStore:
    """
    A simple persistent vector store for Salesforce documentation chunks.

    Data is stored in two files inside `persist_dir`:
      - embeddings.npy  — numpy array of shape (N, 384)
      - docs.json       — list of {"text", "source", "chunk_id"} dicts

    Cosine similarity is used to rank chunks by relevance to a query.
    """

    def __init__(self, persist_dir: str = "./vectorstore"):
        """
        Initialize the vector store, loading any previously saved data.

        Args:
            persist_dir: Directory where embeddings and docs are saved.
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self._embeddings_path = os.path.join(persist_dir, "embeddings.npy")
        self._docs_path = os.path.join(persist_dir, "docs.json")

        print("Initializing embedding model (first run may download ~80 MB)...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Load existing data from disk (if any)
        self._load()

    def _load(self) -> None:
        """Load persisted embeddings and document metadata from disk."""
        if os.path.exists(self._embeddings_path) and os.path.exists(self._docs_path):
            self._embeddings = np.load(self._embeddings_path)
            with open(self._docs_path, "r", encoding="utf-8") as f:
                self._docs = json.load(f)
            print(f"Loaded {len(self._docs)} chunks from disk.")
        else:
            # Start empty — shape (0, 384) so vstack works on first add
            self._embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            self._docs = []

    def _save(self) -> None:
        """Persist embeddings and document metadata to disk."""
        np.save(self._embeddings_path, self._embeddings)
        with open(self._docs_path, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False)

    def is_empty(self) -> bool:
        """Return True if no documents have been stored yet."""
        return len(self._docs) == 0

    def add_documents(self, chunks: List[Dict]) -> None:
        """
        Embed and store a list of document chunks.

        Args:
            chunks: List of dicts with "text", "source", "chunk_id" keys
                    (produced by ingestion.chunk_documents).
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]

        print(f"Embedding {len(texts)} chunks (this may take a minute)...")
        new_embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Stack new embeddings onto any existing ones
        if self._embeddings.shape[0] == 0:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self._docs.extend(chunks)
        self._save()
        print(f"Stored {len(texts)} chunks. Total: {len(self._docs)}")

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Find the most semantically similar chunks for a query.

        Uses cosine similarity:  sim(a, b) = dot(a, b) / (|a| * |b|)
        A score of 1.0 means identical, 0.0 means unrelated.

        Args:
            query:     The user's question.
            n_results: Number of top chunks to return.

        Returns:
            List of {"text", "source", "score"} dicts, best match first.
        """
        if self.is_empty():
            return []

        # Embed the query with the same model used for documents
        query_vec = self.embedder.encode([query], convert_to_numpy=True)[0].astype(np.float32)

        # Compute cosine similarity between query and all stored embeddings
        # dot product of normalized vectors = cosine similarity
        doc_norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)

        # Avoid division by zero for any zero vectors
        denom = doc_norms * query_norm
        denom = np.where(denom == 0, 1e-10, denom)

        similarities = np.dot(self._embeddings, query_vec) / denom

        # Get indices of top-N most similar chunks
        n = min(n_results, len(self._docs))
        top_indices = np.argsort(similarities)[::-1][:n]

        return [
            {
                "text": self._docs[i]["text"],
                "source": self._docs[i].get("source", "unknown"),
                "score": float(similarities[i]),
            }
            for i in top_indices
        ]

    def reset(self) -> None:
        """Delete all stored documents and embeddings."""
        self._embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self._docs = []
        self._save()
        print("Vector store cleared.")
