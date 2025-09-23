import os
from typing import List, Dict, Any
import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Load config
with open("config/retrieval_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

EMBED_MODEL = config.get('embed_model', 'all-MiniLM-L6-v2')
RERANK_MODEL = config.get('rerank_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
COLLECTION_NAME = config.get('collection_name', 'research_docs')
TOP_N = config.get('top_n', 50)  # For reranking
FUSION_WEIGHT_DENSE = config.get('fusion_weight_dense', 0.7)
FUSION_WEIGHT_SPARSE = config.get('fusion_weight_sparse', 0.3)

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
client = QdrantClient(url=qdrant_url)

# Initialize models
embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

# Global BM25 index (to be built from corpus)
bm25_index = None
corpus_texts = []

def build_bm25_index():
    """Build BM25 index from all documents in collection."""
    global bm25_index, corpus_texts
    if bm25_index is None:
        # Fetch all points (in production, cache or precompute)
        points = client.scroll(collection_name=COLLECTION_NAME, limit=10000)[0]
        corpus_texts = [point.payload['text'] for point in points]
        tokenized_corpus = [text.split() for text in corpus_texts]
        bm25_index = BM25Okapi(tokenized_corpus)

def embed_query(query: str) -> List[float]:
    """Embed the query."""
    return embedder.encode(query).tolist()

def retrieve_dense(query: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Perform dense similarity search."""
    query_vector = embed_query(query)
    search_filter = None
    if filters:
        # Convert filters to Qdrant filter format
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        search_filter = Filter(must=conditions)

    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=search_filter
    )
    results = []
    for hit in search_result:
        results.append({
            "chunk_id": hit.id,
            "text": hit.payload["text"],
            "score": hit.score,
            "source_path": hit.payload.get("source_path"),
            "page": hit.payload.get("page"),
            "section": hit.payload.get("section"),
            "title": hit.payload.get("title"),
            "metadata": hit.payload
        })
    return results

def retrieve_sparse(query: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Perform sparse BM25 search."""
    build_bm25_index()
    tokenized_query = query.split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        # Fetch point by id or simulate
        # For simplicity, assume we have corpus_texts
        point = client.retrieve(collection_name=COLLECTION_NAME, ids=[f"chunk_{idx}"])[0]  # Placeholder
        results.append({
            "chunk_id": f"chunk_{idx}",
            "text": corpus_texts[idx],
            "score": bm25_scores[idx],
            "source_path": point.payload.get("source_path"),
            "page": point.payload.get("page"),
            "section": point.payload.get("section"),
            "title": point.payload.get("title"),
            "metadata": point.payload
        })
    return results

def fuse_scores(dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
    """Fuse dense and sparse scores."""
    # Normalize scores
    dense_scores = [r['score'] for r in dense_results]
    sparse_scores = [r['score'] for r in sparse_results]
    dense_norm = (np.array(dense_scores) - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores) + 1e-8)
    sparse_norm = (np.array(sparse_scores) - np.min(sparse_scores)) / (np.max(sparse_scores) - np.min(sparse_scores) + 1e-8)

    # Combine all unique chunks
    all_chunks = {r['chunk_id']: r for r in dense_results + sparse_results}
    for chunk_id, chunk in all_chunks.items():
        dense_score = next((r['score'] for r in dense_results if r['chunk_id'] == chunk_id), 0)
        sparse_score = next((r['score'] for r in sparse_results if r['chunk_id'] == chunk_id), 0)
        fused_score = FUSION_WEIGHT_DENSE * dense_score + FUSION_WEIGHT_SPARSE * sparse_score
        chunk['fused_score'] = fused_score

    # Sort by fused score
    sorted_chunks = sorted(all_chunks.values(), key=lambda x: x['fused_score'], reverse=True)
    return sorted_chunks

def rerank_results(query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """Re-rank candidates using cross-encoder."""
    pairs = [(query, cand['text']) for cand in candidates[:TOP_N]]
    rerank_scores = reranker.predict(pairs)
    for i, score in enumerate(rerank_scores):
        candidates[i]['rerank_score'] = score
    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    return candidates[:top_k]

def search(query: str, top_k: int = 10, mode: str = "hybrid", filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Main search function as per tool spec."""
    if mode == "dense":
        results = retrieve_dense(query, top_k=top_k, filters=filters)
    elif mode == "sparse":
        results = retrieve_sparse(query, top_k=top_k, filters=filters)
    elif mode == "hybrid":
        dense_res = retrieve_dense(query, top_k=top_k, filters=filters)
        sparse_res = retrieve_sparse(query, top_k=top_k, filters=filters)
        fused = fuse_scores(dense_res, sparse_res)
        results = rerank_results(query, fused, top_k=top_k)
    else:
        raise ValueError("Invalid mode")

    # Ensure output format
    return [{
        "chunk_id": r["chunk_id"],
        "text": r["text"],
        "score": r.get("rerank_score", r.get("fused_score", r["score"])),
        "source_path": r["source_path"],
        "page": r.get("page"),
        "section": r.get("section"),
        "title": r.get("title")
    } for r in results]

# Alias for backward compatibility
def retrieve(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    return search(query, top_k=top_k, mode="hybrid")

if __name__ == "__main__":
    # Example usage
    query = "What are the key principles of AI ethics?"
    results = retrieve(query, top_k=5)
    for res in results:
        print(f"Score: {res['score']:.4f}, Text: {res['text'][:100]}...")
