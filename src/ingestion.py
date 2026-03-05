"""
ingestion.py
------------
Responsible for loading Salesforce documentation content, cleaning it,
and splitting it into smaller chunks that can be embedded and stored.

Two sources are supported:
  1. Local .txt files in the data/ directory
  2. Remote URLs (fetched with requests + cleaned with BeautifulSoup)
"""

import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict


# ---------------------------------------------------------------------------
# Default Salesforce documentation URLs.
# These are publicly accessible plain-HTML pages from Salesforce Help.
# You can add more URLs here or pass your own list to ingest_urls().
# ---------------------------------------------------------------------------
DEFAULT_SALESFORCE_URLS = [
    "https://help.salesforce.com/s/articleView?id=sf.basics_intro.htm",
    "https://help.salesforce.com/s/articleView?id=sf.crm_oa_objects_overview.htm",
    "https://developer.salesforce.com/docs/atlas.en-us.api_rest.meta/api_rest/intro_rest.htm",
]


def fetch_url(url: str, timeout: int = 10) -> str:
    """
    Fetch a web page and return its visible text content.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Cleaned plain text extracted from the page, or empty string on failure.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SalesforceRAG/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARNING] Could not fetch {url}: {e}")
        return ""

    # Parse HTML and extract readable text
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove navigation, script, and style elements that add noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Get text, collapse whitespace
    text = soup.get_text(separator=" ", strip=True)
    # Collapse multiple spaces/newlines into single spaces
    text = " ".join(text.split())
    return text


def load_local_docs(data_dir: str = "data") -> List[Dict]:
    """
    Load all .txt files from the data/ directory as documents.

    Each document is a dict with:
        - "text":   the content
        - "source": file path (used as metadata in the vector store)

    Args:
        data_dir: Path to the directory containing .txt files.

    Returns:
        List of document dicts.
    """
    documents = []

    if not os.path.isdir(data_dir):
        return documents

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if text:
            documents.append({"text": text, "source": filepath})
            print(f"  Loaded local file: {filename}")

    return documents


def ingest_urls(urls: List[str]) -> List[Dict]:
    """
    Fetch and process a list of URLs into documents.

    Args:
        urls: List of URLs to scrape.

    Returns:
        List of document dicts with "text" and "source" keys.
    """
    documents = []

    for url in urls:
        print(f"  Fetching: {url}")
        text = fetch_url(url)

        if text and len(text) > 100:  # skip empty / near-empty pages
            documents.append({"text": text, "source": url})
        else:
            print(f"  [SKIP] Not enough content from {url}")

    return documents


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 600,
    overlap: int = 80,
) -> List[Dict]:
    """
    Split documents into overlapping text chunks for embedding.

    Smaller chunks make retrieval more precise; overlap ensures we don't
    cut important context at chunk boundaries.

    Args:
        documents: List of {"text": ..., "source": ...} dicts.
        chunk_size: Target number of characters per chunk.
        overlap:    Number of characters to repeat between adjacent chunks.

    Returns:
        List of chunk dicts with "text", "source", and "chunk_id" keys.
    """
    chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source,
                        "chunk_id": f"{source}::chunk{chunk_index}",
                    }
                )
                chunk_index += 1

            # Move forward by (chunk_size - overlap) so chunks overlap
            start += chunk_size - overlap

    print(f"  Created {len(chunks)} chunks from {len(documents)} document(s)")
    return chunks


def load_all_documents(
    urls: List[str] = None,
    data_dir: str = "data",
) -> List[Dict]:
    """
    Convenience function: load docs from both local files and URLs,
    then chunk everything.

    Args:
        urls:     URLs to scrape (defaults to DEFAULT_SALESFORCE_URLS).
        data_dir: Directory for local .txt files.

    Returns:
        List of chunked document dicts ready for embedding.
    """
    if urls is None:
        urls = DEFAULT_SALESFORCE_URLS

    print("Loading local documents...")
    local_docs = load_local_docs(data_dir)

    print("Fetching documents from URLs...")
    url_docs = ingest_urls(urls)

    all_docs = local_docs + url_docs
    if not all_docs:
        raise ValueError(
            "No documents were loaded. Add .txt files to data/ or check your URLs."
        )

    print("Chunking documents...")
    return chunk_documents(all_docs)
