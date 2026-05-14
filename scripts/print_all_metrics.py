"""Print all real evaluation metrics for v1 and v2 in a clean table."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# v1 retrieval
section("v1 Retrieval (50 gold questions)")
for row in json.loads((EVAL / "retrieval_summary.json").read_text()):
    print(
        f"  {row['strategy']:<18s} "
        f"R@1={row['recall@1']:.3f}  "
        f"R@5={row['recall@5']:.3f}  "
        f"R@10={row['recall@10']:.3f}  "
        f"MRR={row['mrr']:.4f}  "
        f"lat={row['mean_latency_ms']:.1f}ms"
    )

# v1 generation (RAGAS)
section("v1 Generation - RAGAS (20 questions)")
ragas = json.loads((EVAL / "ragas_summary.json").read_text())
for k, v in ragas.items():
    print(f"  {k:<22s} {v}")

# v2 retrieval
section("v2 Retrieval (50 gold questions)")
for row in json.loads((EVAL / "v2_retrieval_summary.json").read_text()):
    print(
        f"  {row['strategy']:<18s} "
        f"R@1={row['recall@1']:.3f}  "
        f"R@5={row['recall@5']:.3f}  "
        f"R@10={row['recall@10']:.3f}  "
        f"MRR={row['mrr']:.4f}  "
        f"lat={row['mean_latency_ms']:.1f}ms"
    )

# v2 generation (LlamaIndex evaluators)
section("v2 Generation - LlamaIndex (20 questions)")
v2gen = json.loads((EVAL / "v2_generation_summary.json").read_text())
for k, v in v2gen.items():
    print(f"  {k:<22s} {v}")
