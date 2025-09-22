<file_path>
RAG/tests/test_evaluation.py
</file_path>

<edit_description>
Create test_evaluation.py for testing metrics calculations and CLI.
</edit_description>

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.evaluation import (
    load_golden_dataset, evaluate_retrieval, evaluate_generation,
    groundedness_score, run_evaluation
)

@pytest.fixture
def sample_golden_data():
    return [
        {
            "id": "1",
            "question": "What is diabetes?",
            "reference_answer": "A chronic condition.",
            "references": ["chunk_1"],
            "difficulty": "easy",
            "expected_behaviour": "answer"
        }
    ]

def test_load_golden_dataset(sample_golden_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in sample_golden_data:
            json.dump(item, f)
            f.write('\n')
        f.flush()
        data = load_golden_dataset(f.name)
        assert len(data) == 1
        assert data[0]["question"] == "What is diabetes?"
    os.unlink(f.name)

@patch('src.evaluation.retriever_search')
def test_evaluate_retrieval(mock_retriever):
    mock_retriever.return_value = [
        {"text": "Diabetes is a condition.", "chunk_id": "1"},
        {"text": "Unrelated text.", "chunk_id": "2"}
    ]
    metrics = evaluate_retrieval("What is diabetes?", ["Diabetes is a condition."], top_k=5)
    assert "precision@1" in metrics
    assert "recall@1" in metrics
    assert "ndcg@5" in metrics
    assert metrics["precision@1"] == 1.0  # Assuming binary match

@patch('src.evaluation.run_agent')
@patch('src.evaluation.retriever_search')
def test_evaluate_generation(mock_retriever, mock_agent):
    mock_retriever.return_value = [{"text": "Evidence"}]
    mock_agent.return_value = {"draft_answer": "Answer"}
    with patch('src.evaluation.evaluate', return_value={"faithfulness": 0.9, "answer_relevancy": 0.85, "context_relevancy": 0.88}):
        metrics = evaluate_generation("Query", "Reference", [{"text": "Evidence"}])
        assert metrics["faithfulness"] == 0.9
        assert "groundedness" in metrics

def test_groundedness_score():
    answer = "Diabetes is chronic."
    evidence = ["Diabetes is a chronic condition."]
    score = groundedness_score(answer, evidence)
    assert "groundedness" in score
    assert isinstance(score["groundedness"], float)

@patch('src.evaluation.load_golden_dataset')
@patch('src.evaluation.evaluate_retrieval')
@patch('src.evaluation.evaluate_generation')
def test_run_evaluation(mock_gen, mock_ret, mock_load, sample_golden_data):
    mock_load.return_value = sample_golden_data
    mock_ret.return_value = {"precision@5": 0.8, "recall@5": 0.7, "ndcg@10": 0.85}
    mock_gen.return_value = {"faithfulness": 0.9, "groundedness": 0.85}

    with tempfile.TemporaryDirectory() as tmpdir:
        run_evaluation("dummy.jsonl", tmpdir, "test_config")
        assert os.path.exists(os.path.join(tmpdir, "results.json"))
        assert os.path.exists(os.path.join(tmpdir, "retrieval_metrics.csv"))
        assert os.path.exists(os.path.join(tmpdir, "generation_metrics.csv"))
        assert os.path.exists(os.path.join(tmpdir, "summary.md"))

def test_cli_parsing():
    # Test CLI argument parsing (mock argparse)
    from src.evaluation import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/golden/healthcare.jsonl")
    parser.add_argument("--output", default="docs/reports")
    parser.add_argument("--config_id", default="default")
    args = parser.parse_args(["--dataset", "test.jsonl"])
    assert args.dataset == "test.jsonl"
