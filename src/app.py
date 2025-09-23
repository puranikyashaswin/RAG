import os
import uuid
import logging
import traceback
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml
from agent import run_agent, agent_graph
from evaluation import run_evaluation
from dotenv import load_dotenv

load_dotenv()

# Load config
try:
    with open("config/app_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    raise RuntimeError("Configuration file config/app_config.yaml not found.")
except yaml.YAMLError as e:
    raise RuntimeError(f"Error parsing config file: {e}")

app = FastAPI(title="Agentic RAG Research Assistant")

class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    top_k: int = 10
    hitl: bool = False

class QueryResponse(BaseModel):
    answer: str
    citations: list
    scores: Dict[str, float]
    trace_id: str

class HITLApproveRequest(BaseModel):
    trace_id: str
    decision: str  # 'approved', 'rejected', 'edit'
    edits: Optional[str] = None

class EvalRunRequest(BaseModel):
    dataset_path: str
    config_id: str

class EvalRunResponse(BaseModel):
    report_path: str
    aggregates: Dict[str, float]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query_assistant(request: QueryRequest):
    try:
        result = run_agent(request.query)
        return QueryResponse(
            answer=result["draft_answer"],
            citations=result["citations"],
            scores=result["scores"],
            trace_id=result["trace_id"]
        )
    except Exception as e:
        logging.error(f"Error in /query for query '{request.query}': {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hitl/approve")
async def hitl_approve(request: HITLApproveRequest):
    # Resume the graph with decision
    try:
        # Assuming we can resume with thread_id = trace_id
        config = {"configurable": {"thread_id": request.trace_id}}
        if request.decision == "approved":
            # Resume with approval
            result = agent_graph.invoke(None, config)  # Placeholder for resume
        elif request.decision == "edit":
            # Apply edits
            pass
        return {"status": "resumed"}
    except Exception as e:
        logging.error(f"Error in /hitl/approve for trace_id '{request.trace_id}': {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eval/run", response_model=EvalRunResponse)
async def run_eval(request: EvalRunRequest):
    try:
        output_dir = f"docs/reports/{uuid.uuid4()}"
        run_evaluation(request.dataset_path, output_dir, request.config_id)
        # Load aggregates from summary
        with open(f"{output_dir}/summary.md", 'r') as f:
            # Parse aggregates (placeholder)
            aggregates = {"avg_groundedness": 0.8}  # Placeholder
        return EvalRunResponse(report_path=output_dir, aggregates=aggregates)
    except Exception as e:
        logging.error(f"Error in /eval/run for dataset '{request.dataset_path}': {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.get("host", "0.0.0.0"), port=config.get("port", 8000))
