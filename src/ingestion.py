import os
import hashlib
import time
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import PyPDF2
from bs4 import BeautifulSoup
import markdown
import yaml
from dotenv import load_dotenv
from retrieval import build_bm25_index

load_dotenv()

# Load config
with open("config/ingestion_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

EMBED_MODEL = config.get('embed_model', 'all-MiniLM-L6-v2')
VECTOR_DB = config.get('vector_db', 'qdrant')
COLLECTION_NAME = config.get('collection_name', 'research_docs')
CHUNK_SIZE = config.get('chunk_size', 1000)
OVERLAP = config.get('overlap', 200)

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
client = QdrantClient(url=qdrant_url)

# Initialize embedding model
embedder = SentenceTransformer(EMBED_MODEL)

def parse_document(file_path: str) -> Optional[Dict[str, Any]]:
    """Parse a document and extract text with metadata."""
    ext = Path(file_path).suffix.lower()
    metadata = {"source_path": str(file_path), "timestamp": datetime.now().isoformat()}

    if ext == '.pdf':
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                pages = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    pages.append({"page": i+1, "text": page_text})
                metadata["pages"] = pages
                # Extract title from first page or metadata
                if reader.metadata and reader.metadata.title:
                    metadata["title"] = reader.metadata.title
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
    elif ext == '.html':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                text = soup.get_text()
                metadata["title"] = soup.title.string if soup.title else "Untitled"
                # Extract sections
                sections = []
                for header in soup.find_all(['h1', 'h2', 'h3']):
                    sections.append({"level": int(header.name[1]), "text": header.get_text()})
                metadata["sections"] = sections
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
    elif ext == '.md':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
                # Extract title and sections
                headers = soup.find_all(['h1', 'h2', 'h3'])
                if headers:
                    metadata["title"] = headers[0].get_text()
                sections = [{"level": int(h.name[1]), "text": h.get_text()} for h in headers]
                metadata["sections"] = sections
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
    else:
        print(f"Unsupported file type: {ext}")
        return None

    return {"text": text, "metadata": metadata}

def semantic_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[Dict[str, Any]]:
    """Chunk text using semantic boundaries (sentences/paragraphs)."""
    # Simple semantic chunking: split by paragraphs, then sentences if needed
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para + "\n"
            else:
                # Force split if para is too long
                sentences = para.split('. ')
                temp = ""
                for sent in sentences:
                    if len(temp) + len(sent) > chunk_size:
                        if temp:
                            chunks.append(temp.strip())
                        temp = sent + ". "
                    else:
                        temp += sent + ". "
                if temp:
                    current_chunk = temp
        else:
            current_chunk += para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap by merging with previous
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev_end = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            chunk = prev_end + " " + chunk
        overlapped_chunks.append(chunk)

    return [{"text": chunk, "chunk_index": i} for i, chunk in enumerate(overlapped_chunks)]

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Embed text chunks."""
    return embedder.encode(chunks).tolist()

def store_in_qdrant(chunks: List[Dict[str, Any]], embeddings: List[List[float]], base_metadata: Dict[str, Any]):
    """Store chunks in Qdrant with enhanced metadata."""
    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
            )

        points = []
        for i, (chunk_info, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.md5(f"{base_metadata['source_path']}_{i}_{base_metadata['timestamp']}".encode()).hexdigest()
            metadata = {
                **base_metadata,
                "chunk_id": chunk_id,
                "chunk_index": chunk_info["chunk_index"],
                "hash": hashlib.sha256(chunk_info["text"].encode()).hexdigest()
            }
            points.append(PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={"text": chunk_info["text"], **metadata}
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as e:
        print(f"Error storing in Qdrant: {e}")

def process_documents(data_dir: str):
    """Process all documents in the data directory."""
    data_path = Path(data_dir)
    for file_path in data_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.html', '.md']:
            print(f"Processing {file_path}")
            try:
                parsed = parse_document(str(file_path))
                if parsed is None:
                    continue
                chunks = semantic_chunk_text(parsed["text"])
                embeddings = embed_chunks([c["text"] for c in chunks])
                store_in_qdrant(chunks, embeddings, parsed["metadata"])
                print(f"Stored {len(chunks)} chunks for {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Rebuild BM25 index after ingestion
    build_bm25_index(force_rebuild=True)

if __name__ == "__main__":
    data_dir = "data/corpus"
    process_documents(data_dir)
