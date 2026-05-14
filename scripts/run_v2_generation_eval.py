"""
v2 Generation Evaluation using LlamaIndex FaithfulnessEvaluator + RelevancyEvaluator.

Writes results to:
  data/eval/v2_generation_summary.json
  data/eval/v2_generation_results.json

Usage
-----
    $env:PYTHONPATH = "."
    python scripts/run_v2_generation_eval.py
    python scripts/run_v2_generation_eval.py --limit 10
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v2.indexing    import load_v2_index, v2_index_ready
from v2.retrieval   import get_v2_retriever
from v2.generation  import build_v2_query_engine
from v2.evaluation.eval_generation import run_v2_generation_eval

EVAL_DIR = ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_JSONL = ROOT / "data" / "chunks" / "semantic_chunks.jsonl"


def _load_nodes():
    from llama_index.core.schema import TextNode
    nodes = []
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            c = json.loads(s)
            nodes.append(TextNode(
                text=c["text"],
                id_=c["chunk_id"],
                metadata=c.get("metadata", {}),
            ))
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 generation evaluation.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of gold questions to evaluate (default 20).")
    parser.add_argument("--strategy", default="hybrid",
                        choices=["semantic", "bm25", "hybrid"],
                        help="Retrieval strategy to use (default: hybrid).")
    args = parser.parse_args()

    if not v2_index_ready():
        print("ERROR: v2 index not ready. Run `python scripts/run_v2_index.py` first.")
        sys.exit(1)

    print("[v2] Loading index …")
    index = load_v2_index()
    nodes = _load_nodes()

    print(f"[v2] Building query engine (strategy={args.strategy}) …")
    retriever = get_v2_retriever(index=index, nodes=nodes, strategy=args.strategy, top_k=5)
    engine    = build_v2_query_engine(retriever)

    print(f"[v2] Running generation eval on {args.limit} questions …")
    result = run_v2_generation_eval(engine, limit=args.limit)

    s = result["summary"]
    print(f"\n[v2] Faithfulness : {s.get('faithfulness')}")
    print(f"[v2] Relevancy     : {s.get('relevancy')}")
    print(f"[v2] Mean latency  : {s['mean_latency_ms']} ms")

    (EVAL_DIR / "v2_generation_summary.json").write_text(
        json.dumps(s, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / "v2_generation_results.json").write_text(
        json.dumps(result["per_question"], indent=2), encoding="utf-8"
    )
    print(f"\n[v2] Results written to data/eval/v2_generation_*.json")


if __name__ == "__main__":
    main()
