import pytest
from unittest.mock import patch, MagicMock
from src.agent import plan_node, retrieve_node, synthesize_node, verify_node, decide_node, run_agent, AgentState

@pytest.fixture
def sample_state():
    return AgentState(
        query="What is diabetes?",
        subgoals=[],
        retrieved_evidence=[],
        draft_answer="",
        citations=[],
        scores={},
        trace_id="test_trace",
        hitl_status="none",
        checkpoints=[],
        attempts=0,
        summary_bundle=None
    )

@patch('src.agent.llm')
def test_plan_node(mock_llm, sample_state):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Subgoal 1: Retrieve info\nSubgoal 2: Summarize"
    mock_llm.__or__ = MagicMock(return_value=mock_chain)
    mock_chain.__or__ = MagicMock(return_value=mock_chain)

    result = plan_node(sample_state)
    assert "subgoals" in result
    assert len(result["subgoals"]) == 2
    assert result["subgoals"][0] == "Subgoal 1: Retrieve info"

@patch('src.agent.retriever_search')
def test_retrieve_node(mock_retriever, sample_state):
    sample_state["subgoals"] = ["Retrieve diabetes info"]
    mock_retriever.return_value = [{"chunk_id": "1", "text": "Diabetes info", "score": 0.9}]

    result = retrieve_node(sample_state)
    assert len(result["retrieved_evidence"]) == 1
    assert result["retrieved_evidence"][0]["text"] == "Diabetes info"

@patch('src.agent.llm')
def test_synthesize_node(mock_llm, sample_state):
    sample_state["retrieved_evidence"] = [{"chunk_id": "1", "text": "Diabetes is a condition."}]
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Diabetes is a chronic condition. [chunk_id: 1]"
    mock_llm.__or__ = MagicMock(return_value=mock_chain)
    mock_chain.__or__ = MagicMock(return_value=mock_chain)

    result = synthesize_node(sample_state)
    assert result["draft_answer"] == "Diabetes is a chronic condition. [chunk_id: 1]"
    assert len(result["citations"]) == 1

@patch('src.agent.groundedness_score')
def test_verify_node(mock_groundedness, sample_state):
    sample_state["draft_answer"] = "Answer"
    sample_state["retrieved_evidence"] = [{"text": "Evidence"}]
    mock_groundedness.return_value = {"groundedness": 0.85}

    result = verify_node(sample_state)
    assert result["scores"]["groundedness"] == 0.85
    assert result["scores"]["faithfulness"] == 0.9  # Default

def test_decide_node_finalize(sample_state):
    sample_state["scores"] = {"groundedness": 0.85, "faithfulness": 0.85}
    sample_state["attempts"] = 0

    decision = decide_node(sample_state)
    assert decision == "finalize"

def test_decide_node_refine(sample_state):
    sample_state["scores"] = {"groundedness": 0.75, "faithfulness": 0.85}
    sample_state["attempts"] = 0

    decision = decide_node(sample_state)
    assert decision == "refine"

def test_decide_node_escalate(sample_state):
    sample_state["scores"] = {"groundedness": 0.75, "faithfulness": 0.85}
    sample_state["attempts"] = 4  # Exceeds max

    decision = decide_node(sample_state)
    assert decision == "escalate"

@patch('src.agent.agent_graph')
def test_run_agent(mock_graph, sample_state):
    mock_graph.invoke.return_value = sample_state

    result = run_agent("Test query")
    assert result["query"] == "Test query"
    assert "trace_id" in result
