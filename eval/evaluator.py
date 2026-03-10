"""
BizIntel Agent Evaluator

Run multiple test queries against the agent and record the metrics:
- Mode Accuracy
- Factual Recall (presence of required facts)
- NLI Confidence (from EvidenceVerifier)
"""

import json
import logging
import sys
import getpass
from pathlib import Path
from datetime import datetime

from agent.schemas import AnalysisMode
from agent.orchestrator import BizIntelAgent
from agent.config import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def evaluate(benchmark_file: Path, output_file: Path):
    with open(benchmark_file) as f:
        benchmarks = json.load(f)

    logger.info(f"Loaded {len(benchmarks)} benchmark cases.")
    agent = BizIntelAgent()

    results = []
    total_score = 0
    
    for i, case in enumerate(benchmarks):
        query = case["query"]
        expected_mode = case["expected_mode"]
        required_facts = case.get("required_facts", [])
        
        logger.info(f"\n[{i+1}/{len(benchmarks)}] Evaluating: '{query}'")
        
        try:
            # Let the agent auto-detect the mode
            res = agent.research(query=query)
            memo = res["memo_object"]
            markdown = res["memo_markdown"]
            
            # 1. Mode Accuracy
            mode_correct = (memo.mode.name == expected_mode)
            
            # 2. Factual Recall
            markdown_lower = markdown.lower()
            facts_found = [fact for fact in required_facts if fact.lower() in markdown_lower]
            recall_score = len(facts_found) / len(required_facts) if required_facts else 1.0
            
            # 3. Overall Confidence
            confidence = memo.overall_confidence
            
            # Score
            score = (mode_correct * 0.4) + (recall_score * 0.4) + (confidence * 0.2)
            total_score += score
            
            logger.info(f"  Mode Correct: {mode_correct} (Expected {expected_mode}, Got {memo.mode.name})")
            logger.info(f"  Recall Score: {recall_score:.2f} ({len(facts_found)}/{len(required_facts)} facts)")
            logger.info(f"  NLI Confidence: {confidence:.0%}")
            
            results.append({
                "query": query,
                "score": score,
                "metrics": {
                    "mode_correct": mode_correct,
                    "expected_mode": expected_mode,
                    "actual_mode": memo.mode.name,
                    "recall_score": recall_score,
                    "confidence": confidence,
                    "facts_missing": [f for f in required_facts if f not in facts_found]
                }
            })
            
        except Exception as e:
            logger.error(f"  Error evaluating '{query}': {e}")
            results.append({"query": query, "error": str(e), "score": 0})
            
    # Summary
    avg_score = total_score / len(benchmarks) if benchmarks else 0
    logger.info(f"\n--- Evaluation Complete ---")
    logger.info(f"Average Score: {avg_score:.2f}")
    
    # Save results
    out = {
        "timestamp": datetime.now().isoformat(),
        "average_score": avg_score,
        "cases": results
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)
        
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    if not settings.openai_api_key:
        print("未检测到 API Key，评测程序需要真实的 LLM API：")
        api_key = getpass.getpass("API Key (隐式输入): ")
        if not api_key.strip():
            logger.error("❌必须提供 API Key 才能完成评测")
            sys.exit(1)
        settings.openai_api_key = api_key.strip()
        
    benchmark_path = settings.eval_cases_dir / "benchmark.json"
    output_path = settings.data_dir.parent / "eval" / "results" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    if not benchmark_path.exists():
        logger.error(f"Benchmark file not found at {benchmark_path}")
        sys.exit(1)
        
    evaluate(benchmark_path, output_path)
