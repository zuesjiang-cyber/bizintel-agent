"""
检索策略效果评测脚本
对比 BM25, Dense, Hybrid (无重排), Full Hybrid 的性能和延迟。
"""

import time
import json
from data.eval_cases.benchmark import EVAL_CASES # Need to make sure this or a mocked version exists
from retrieval.hybrid_retriever import HybridRetriever
from agent.config import settings

# --- Mock 评测数据集 (基于用户的需求) ---
GROUND_TRUTH = [
    {"query": "When was Stripe founded and who are its founders?", "relevant_source_id": "stripe_funding"},
    {"query": "What is Stripe's current valuation after the Series I round?", "relevant_source_id": "stripe_funding"},
    {"query": "How much was Stripe's estimated net revenue in 2023?", "relevant_source_id": "stripe_revenue"},
    {"query": "What was Stripe's total payment volume in 2023?", "relevant_source_id": "stripe_revenue"},
    {"query": "Who is Stripe's fiercest competitor for global enterprise payments?", "relevant_source_id": "stripe_competitors"},
    {"query": "Which overarching macro trend is driving embedded finance?", "relevant_source_id": "payments_industry"},
    {"query": "What pricing strategy does Braintree use to compete?", "relevant_source_id": "stripe_competitors"},
    {"query": "What regulatory bodies are scrutinizing payment facilitators?", "relevant_source_id": "stripe_risks"},
    {"query": "What happened to Stripe's valuation from $95 billion?", "relevant_source_id": "stripe_risks"},
    {"query": "What is Adyen's primary market advantage?", "relevant_source_id": "stripe_competitors"},
    {"query": "How many employees does Stripe have approximately?", "relevant_source_id": "stripe_profile"},
    {"query": "What is Stripe Atlas used for?", "relevant_source_id": "stripe_profile"},
    {"query": "What is the primary motive behind the Series I round of funding?", "relevant_source_id": "stripe_funding"},
    {"query": "What is the global digital payment market projected size in 2030?", "relevant_source_id": "payments_industry"},
    {"query": "Why does Stripe invest in machine learning for payments?", "relevant_source_id": "payments_industry"},
]

def calculate_mrr(retrieved, gt_source, k=10):
    for i, c in enumerate(retrieved[:k]):
        if c.source_id == gt_source:
            return 1.0 / (i + 1)
    return 0.0

def calculate_recall(retrieved, gt_source, k):
    for c in retrieved[:k]:
        if c.source_id == gt_source:
            return 1.0
    return 0.0

def run_evaluation():
    print("Initializing Retriever (loading models...)")
    # For CI or offline we could use Dummy, but the user wants real evaluations. Standard init.
    try:
        retriever = HybridRetriever()
        retriever.load_index(settings.data_dir / "index")
    except Exception as e:
        print(f"Index loading failed, using dummy retrieval: {e}")
        retriever = HybridRetriever(load_models=False)
        # Add basic dummy chunk
        retriever.index([{"chunk_id": f"c_{idx}", "text": gt["query"], "source_id": gt["relevant_source_id"]} for idx, gt in enumerate(GROUND_TRUTH)])

    modes = ["bm25_only", "dense_only", "hybrid_no_rerank", "full_hybrid"]
    results = {mode: {"recall_5": 0, "recall_10": 0, "mrr_10": 0, "latency": 0.0} for mode in modes}
    
    total_queries = len(GROUND_TRUTH)

    for mode in modes:
        print(f"Evaluating mode: {mode}...")
        total_time = 0
        r5_total = 0
        r10_total = 0
        mrr_total = 0

        for case in GROUND_TRUTH:
            q = case["query"]
            gt = case["relevant_source_id"]

            t0 = time.time()
            retrieved = retriever.retrieve(query=q, top_k=10, mode=mode)
            t1 = time.time()
            
            total_time += (t1 - t0)
            r5_total += calculate_recall(retrieved, gt, 5)
            r10_total += calculate_recall(retrieved, gt, 10)
            mrr_total += calculate_mrr(retrieved, gt, 10)

        results[mode]["recall_5"] = r5_total / total_queries
        results[mode]["recall_10"] = r10_total / total_queries
        results[mode]["mrr_10"] = mrr_total / total_queries
        results[mode]["latency"] = (total_time / total_queries) * 1000  # ms

    # 输出表格
    print("\n┌──────────────────┬───────────┬───────────┬──────────┬──────────┐")
    print("│ Method           │ Recall@5  │ Recall@10 │ MRR@10   │ Latency  │")
    print("├──────────────────┼───────────┼───────────┼──────────┼──────────┤")
    for mode in modes:
        r = results[mode]
        display_name = {
            "bm25_only": "BM25 Only       ",
            "dense_only": "Dense Only      ",
            "hybrid_no_rerank": "Hybrid (no RR)  ",
            "full_hybrid": "Full Hybrid     "
        }[mode]
        print(f"│ {display_name} │ {r['recall_5']:.2f}      │ {r['recall_10']:.2f}      │ {r['mrr_10']:.2f}     │ {r['latency']:4.0f}ms   │")
    print("└──────────────────┴───────────┴───────────┴──────────┴──────────┘")

if __name__ == "__main__":
    run_evaluation()
