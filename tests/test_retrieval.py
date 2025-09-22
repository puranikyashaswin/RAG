import pytest
from unittest.mock import patch, MagicMock
from src.retrieval import retrieve_dense, retrieve_sparse, fuse_scores, rerank_results, search

@pytest.fixture
def mock_qdrant_client():
    with patch('src.retrieval.client') as mock_client:
        yield mock_client

def test_retrieve_dense_basic(mock_qdrant_client):
    mock_qdrant_client.search.return_value = [
        MagicMock(payload={'text': 'test text', 'source_path': 'test.pdf', 'chunk_id': '1'}, score=0.9)
    ]
    results = retrieve_dense("test query", top_k=5)
    assert len(results) == 1
    assert results[0]['text'] == 'test text'
    assert results[0]['score'] == 0.9

def test_retrieve_dense_with_filters(mock_qdrant_client):
    mock_qdrant_client.search.return_value = []
    results = retrieve_dense("test query", top_k=5, filters={'source_path': 'test.pdf'})
    mock_qdrant_client.search.assert_called_once()
    # Check filter was applied

def test_retrieve_sparse_basic():
    with patch('src.retrieval.bm25_index', [MagicMock(get_scores=lambda q: [0.8, 0.6])]), \
         patch('src.retrieval.corpus_texts', ['text1', 'text2']), \
         patch('src.retrieval.client.retrieve', return_value=MagicMock(payload={'source_path': 'test.pdf'})):
        results = retrieve_sparse("test query", top_k=2)
        assert len(results) == 2
        assert results[0]['score'] == 0.8

def test_fuse_scores():
    dense_results = [{'chunk_id': '1', 'score': 0.9}, {'chunk_id': '2', 'score': 0.7}]
    sparse_results = [{'chunk_id': '1', 'score': 0.8}, {'chunk_id': '3', 'score': 0.6}]
    fused = fuse_scores(dense_results, sparse_results)
    assert len(fused) == 3
    assert fused[0]['fused_score'] > fused[1]['fused_score']  # Assuming weights

def test_rerank_results():
    candidates = [
        {'text': 'relevant text', 'score': 0.8},
        {'text': 'irrelevant text', 'score': 0.6}
    ]
    with patch('src.retrieval.reranker.predict', return_value=[0.9, 0.4]):
        reranked = rerank_results("query", candidates, top_k=2)
        assert reranked[0]['rerank_score'] == 0.9
        assert reranked[1]['rerank_score'] == 0.4

def test_search_hybrid_mode(mock_qdrant_client):
    with patch('src.retrieval.retrieve_dense', return_value=[{'chunk_id': '1', 'text': 'text', 'score': 0.8}]), \
         patch('src.retrieval.retrieve_sparse', return_value=[{'chunk_id': '1', 'text': 'text', 'score': 0.7}]), \
         patch('src.retrieval.rerank_results', return_value=[{'chunk_id': '1', 'text': 'text', 'rerank_score': 0.85}]):
        results = search("query", mode="hybrid", top_k=1)
        assert len(results) == 1
        assert results[0]['score'] == 0.85

def test_search_dense_mode(mock_qdrant_client):
    with patch('src.retrieval.retrieve_dense', return_value=[{'chunk_id': '1', 'text': 'text', 'score': 0.8}]):
        results = search("query", mode="dense", top_k=1)
        assert results[0]['score'] == 0.8

def test_search_sparse_mode():
    with patch('src.retrieval.retrieve_sparse', return_value=[{'chunk_id': '1', 'text': 'text', 'score': 0.7}]):
        results = search("query", mode="sparse", top_k=1)
        assert results[0]['score'] == 0.7
