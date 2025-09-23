import os
import uuid
import time
from typing import Dict, List, Any, TypedDict, Optional
from datetime import datetime
import yaml
from langgraph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client as LangSmithClient
from dotenv import load_dotenv
from retrieval import search as retriever_search
from evaluation import groundedness_score  # Assuming implemented

load_dotenv()

# Load config
with open("config/agent_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

LLM_MODEL = config.get('llm_model', 'gpt-4o-mini')
MAX_REFINE_LOOPS = config.get('max_refine_loops', 3)
GROUNDEDNESS_TARGET = config.get('groundedness_target', 0.8)
FAITHFULNESS_TARGET = config.get('faithfulness_target', 0.85)

# Initialize LLM and tracing
llm = ChatOpenAI(model=LLM_MODEL, api_key=os.getenv("OPENAI_API_KEY"))
langsmith_client = LangSmithClient(api_key=os.getenv("LANGSMITH_API_KEY"))

# State Schema
class AgentState(TypedDict):
    query: str
    subgoals: List[str]
    retrieved_evidence: List[Dict[str, Any]]
    draft_answer: str
    citations: List[Dict[str, Any]]
    scores: Dict[str, float]
    trace_id: str
    hitl_status: str  # 'none', 'pending', 'approved', 'rejected'
    checkpoints: List[Dict[str, Any]]
    attempts: int
    summary_bundle: Optional[Dict[str, Any]]

# Nodes
def plan_node(state: AgentState) -> AgentState:
    """Decompose query into subgoals and plan retrieval."""
    trace_id = state['trace_id']
    langsmith_client.create_run(
        name="plan", run_type="llm", inputs={"query": state['query']}, trace_id=trace_id
    )
    prompt = ChatPromptTemplate.from_template(
        "Decompose the query into subgoals: {query}\n"
        "Output a list of subgoals and a retrieval plan with query rewrites and filters."
    )
    chain = prompt | llm | StrOutputParser()
    plan_text = chain.invoke({"query": state["query"]})
    # Parse subgoals (simple split)
    subgoals = [g.strip() for g in plan_text.split('\n') if g.strip()]
    new_state = {**state, "subgoals": subgoals, "attempts": 0}
    langsmith_client.update_run(trace_id, outputs={"subgoals": subgoals})
    return new_state

def retrieve_node(state: AgentState) -> AgentState:
    """Execute retrieval for subgoals."""
    trace_id = state['trace_id']
    langsmith_client.create_run(
        name="retrieve", run_type="tool", inputs={"subgoals": state['subgoals']}, trace_id=trace_id
    )
    evidence = []
    for goal in state['subgoals']:
        results = retriever_search(goal, top_k=10, mode="hybrid")
        evidence.extend(results)
    # Deduplicate by chunk_id
    unique_evidence = {e['chunk_id']: e for e in evidence}.values()
    new_state = {**state, "retrieved_evidence": list(unique_evidence)}
    langsmith_client.update_run(trace_id, outputs={"evidence_count": len(unique_evidence)})
    return new_state

def synthesize_node(state: AgentState) -> AgentState:
    """Draft answer grounded in evidence with citations."""
    trace_id = state['trace_id']
    langsmith_client.create_run(
        name="synthesize", run_type="llm", inputs={"evidence": [e['text'] for e in state['retrieved_evidence']]}, trace_id=trace_id
    )
    evidence_text = "\n".join([f"{e['text']} [chunk_id: {e['chunk_id']}]" for e in state['retrieved_evidence']])
    prompt = ChatPromptTemplate.from_template(
        "Synthesize an answer to: {query}\n"
        "Using evidence: {evidence}\n"
        "Include citations in format [source_path:page|section|chunk_id]."
    )
    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({"query": state["query"], "evidence": evidence_text})
    # Extract citations (placeholder)
    citations = [{"chunk_id": e['chunk_id'], "source": e['source_path']} for e in state['retrieved_evidence']]
    new_state = {**state, "draft_answer": draft, "citations": citations}
    langsmith_client.update_run(trace_id, outputs={"draft": draft})
    return new_state

def verify_node(state: AgentState) -> AgentState:
    """Compute scores and validate."""
    trace_id = state['trace_id']
    langsmith_client.create_run(
        name="verify", run_type="tool", inputs={"draft": state['draft_answer']}, trace_id=trace_id
    )
    evidence_texts = [e['text'] for e in state['retrieved_evidence']]
    groundedness = groundedness_score(state['draft_answer'], evidence_texts)
    # Placeholder for other scores
    scores = {"groundedness": groundedness['groundedness'], "faithfulness": 0.9, "relevance": 0.85}
    new_state = {**state, "scores": scores}
    langsmith_client.update_run(trace_id, outputs=scores)
    return new_state

def decide_node(state: AgentState) -> str:
    """Decide next step."""
    scores = state['scores']
    if scores['groundedness'] >= GROUNDEDNESS_TARGET and scores['faithfulness'] >= FAITHFULNESS_TARGET:
        return "finalize"
    elif state['attempts'] < MAX_REFINE_LOOPS:
        return "refine"
    else:
        # Escalate to HITL
        summary_bundle = {
            "query": state['query'],
            "plan": state['subgoals'],
            "evidence": state['retrieved_evidence'],
            "draft": state['draft_answer'],
            "risks": "Low groundedness",
            "proposed_action": "Approve or edit"
        }
        interrupt(summary_bundle)  # LangGraph interrupt
        return "escalate"

def finalize_node(state: AgentState) -> AgentState:
    """Finalize answer."""
    return {**state, "hitl_status": "approved"}

def refine_node(state: AgentState) -> AgentState:
    """Refine plan."""
    new_attempts = state['attempts'] + 1
    # Modify subgoals (placeholder)
    refined_subgoals = state['subgoals'] + ["Refined goal"]
    return {**state, "subgoals": refined_subgoals, "attempts": new_attempts}

def escalate_node(state: AgentState) -> AgentState:
    """Handle HITL escalation."""
    # After interrupt, assume approval
    return {**state, "hitl_status": "approved"}

# Graph
def create_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("verify", verify_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("refine", refine_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges(
        "verify",
        decide_node,
        {"finalize": "finalize", "refine": "refine", "escalate": "escalate"}
    )
    graph.add_edge("finalize", END)
    graph.add_edge("refine", "retrieve")  # Loop back
    graph.add_edge("escalate", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

agent_graph = create_graph()

def run_agent(query: str) -> AgentState:
    trace_id = str(uuid.uuid4())
    initial_state = {
        "query": query,
        "subgoals": [],
        "retrieved_evidence": [],
        "draft_answer": "",
        "citations": [],
        "scores": {},
        "trace_id": trace_id,
        "hitl_status": "none",
        "checkpoints": [],
        "attempts": 0,
        "summary_bundle": None
    }
    result = agent_graph.invoke(initial_state, {"configurable": {"thread_id": trace_id}})
    return result

if __name__ == "__main__":
    query = "Summarize AI ethics principles."
    result = run_agent(query)
    print("Answer:", result["draft_answer"])
    print("Scores:", result["scores"])
