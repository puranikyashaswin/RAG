import json
import os
import argparse
from typing import List, Dict, Any
import pandas as pd
from sklearn.metrics import ndcg_score
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, Groundedness
from ragas.llms import LangchainLLM
from langchain_openai import ChatOpenAI

from datasets import Dataset
from retrieval import search as retriever_search
from agent import run_agent
import yaml
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM for ragas
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Config will be loaded in run_evaluation

def load_golden_dataset(path: str) -> List[Dict[str, Any]]:
    """Load the golden dataset from JSONL."""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

# Groundedness is handled by ragas Groundedness metric

def evaluate_retrieval(query: str, relevant_docs: List[str], top_k: int = 10, mode: str = "hybrid") -> Dict[str, float]:
    """Evaluate retrieval metrics."""
    retrieved = retriever_search(query, top_k=top_k, mode=mode)
    retrieved_texts = [doc["text"] for doc in retrieved]
    retrieved_ids = [doc["chunk_id"] for doc in retrieved]

    # Binary relevance (assume relevant if text contains any relevant phrase)
    y_true = [1 if any(rel in text for rel in relevant_docs) else 0 for text in retrieved_texts]
    y_score = [1] * len(retrieved_texts)  # Assume retrieved are scored high

    metrics = {}
    for k in K_LIST:
        if k <= len(y_true):
            precision_k = sum(y_true[:k]) / k
            recall_k = sum(y_true[:k]) / len(relevant_docs) if relevant_docs else 0
            ndcg = ndcg_score([y_true[:k]], [y_score[:k]], k=k)
            metrics[f"precision@{k}"] = precision_k
            metrics[f"recall@{k}"] = recall_k
            metrics[f"ndcg@{k}"] = ndcg
    return metrics

def evaluate_generation(query: str, reference_answer: str, retrieved_docs: List[Dict[str, Any]], mode: str = "hybrid") -> Dict[str, float]:
    """Evaluate generation metrics."""
    try:
        result = run_agent(query)
        answer = result["draft_answer"]
    except Exception as e:
        print(f"Agent failed for query '{query}': {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_relevancy": 0.0,
            "groundedness": 0.0,
            "completeness": 0.0
        }
    contexts = [doc["text"] for doc in retrieved_docs]

    if not contexts:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_relevancy": 0.0,
            "groundedness": 0.0,
            "completeness": 0.0
        }

    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [reference_answer]
    }
    dataset = Dataset.from_dict(data)
    scores = evaluate(dataset, metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), Groundedness()], llm=LangchainLLM(llm))
    return {
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "context_relevancy": scores["context_precision"],
        "groundedness": scores["groundedness"],
        "completeness": 0.8  # Placeholder
    }

def run_evaluation(dataset_path: str, output_dir: str, config_id: str, config_path: str, mode: str):
    """Run full evaluation and generate reports."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K_LIST = config.get('k_list', [1, 3, 5, 10])
    REGRESSION_TOLERANCE = config.get('regression_tolerance', 0.05)

    golden_data = load_golden_dataset(dataset_path)
    results = []
    aggregates = {f"avg_{k}": [] for k in ["precision@5", "recall@5", "ndcg@10", "groundedness", "faithfulness", "answer_relevancy", "context_relevancy", "completeness"]}

    for item in golden_data:
        query = item["question"]
        reference = item["reference_answer"]
        relevant_docs = item.get("references", [])

        # Retrieval metrics
        ret_metrics = evaluate_retrieval(query, relevant_docs, top_k=max(K_LIST), mode=mode)

        # Generation metrics
        retrieved = retriever_search(query, top_k=10, mode=mode)
        gen_metrics = evaluate_generation(query, reference, retrieved, mode=mode)

        result = {
            "id": item["id"],
            "query": query,
            "difficulty": item.get("difficulty", "medium"),
            "retrieval": ret_metrics,
            "generation": gen_metrics
        }
        results.append(result)

        # Collect for aggregates
        for key in aggregates:
            if key in ret_metrics:
                aggregates[key].append(ret_metrics[key])
            elif key in gen_metrics:
                aggregates[key].append(gen_metrics[key])

    # Compute aggregates
    agg_summary = {k: sum(v)/len(v) if v else 0 for k, v in aggregates.items()}

    # Check regression
    # Placeholder: compare to baseline
    baseline = config.get("baseline_metrics", {})
    regressions = {}
    for key, current in agg_summary.items():
        if key in baseline:
            diff = current - baseline[key]
            if diff < -REGRESSION_TOLERANCE:
                regressions[key] = diff

    if regressions:
        print(f"Regressions detected: {regressions}")
        # Fail CI
        exit(1)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # CSV for retrieval
    ret_df = pd.DataFrame([{"id": r["id"], **r["retrieval"]} for r in results])
    ret_df.to_csv(f"{output_dir}/retrieval_metrics.csv", index=False)

    # CSV for generation
    gen_df = pd.DataFrame([{"id": r["id"], **r["generation"]} for r in results])
    gen_df.to_csv(f"{output_dir}/generation_metrics.csv", index=False)

    # Markdown summary
    with open(f"{output_dir}/summary.md", 'w') as f:
        f.write("# Evaluation Summary\n\n")
        f.write(f"Config ID: {config_id}\n\n")
        f.write("## Aggregates\n")
        for k, v in agg_summary.items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write("\n## Regressions\n")
        if regressions:
            for k, v in regressions.items():
                f.write(f"- {k}: {v:.4f}\n")
        else:
            f.write("None\n")

    print(f"Reports saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--dataset", default="data/golden/healthcare.jsonl", help="Path to golden dataset")
    parser.add_argument("--output", default="docs/reports", help="Output directory")
    parser.add_argument("--config", default="config/eval_config.yaml", help="Path to config file")
    parser.add_argument("--config_id", default="default", help="Config ID")
    parser.add_argument("--mode", default="hybrid", help="Retrieval mode: dense, sparse, hybrid")
    args = parser.parse_args()
    run_evaluation(args.dataset, args.output, args.config_id, args.config, args.mode)
