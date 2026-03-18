"""Quick test: Feed existing legal_signal_graph.json into Agent 6 v3 Hybrid Validator."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from mcp.tools.evaluate_statute_ingredients import evaluate_statute_ingredients_v1
from mcp.mcp_server import _call_gemini

# Use the last successful signal graph (13 signals, 20 candidates)
SIGNAL_GRAPH = "cases/case_4267450c0e26/legal_signal_graph.json"

if not os.path.exists(SIGNAL_GRAPH):
    print(f"ERROR: {SIGNAL_GRAPH} not found. Use a different case.")
    sys.exit(1)

print("Testing Agent 6 v3 Hybrid Validator...")
result = evaluate_statute_ingredients_v1(
    case_id="case_4267450c0e26",
    legal_signal_graph_path=SIGNAL_GRAPH,
    cases_dir="cases",
    _gemini_caller=_call_gemini,
)

print(f"Evaluated: {result['evaluated_count']}")
print(f"Rejected:  {result['rejected_count']}")
print(f"LLM calls: {result['llm_calls_made']}")

# Load and show top results
with open(result["result_ref"], "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\nCase Domains: {data.get('case_domains', [])}")
print(f"Case Facts:   {len(data.get('case_facts', []))}")

print("\n=== TOP EVALUATED STATUTES ===")
for ev in data["statute_evaluations"][:10]:
    status_icon = "✅" if ev["status"] == "evaluated" else "❌"
    print(f"  {status_icon} {ev['statute_id']:>5} | {ev['name'][:60]:<60} | score={ev['score']:.4f} | {ev['confidence']} | {ev['status']}")
    if ev["status"] == "rejected":
        for r in ev.get("reasoning", []):
            print(f"        → {r}")

print("\n=== REJECTED STATUTES ===")
for ev in data["statute_evaluations"]:
    if ev["status"] == "rejected":
        print(f"  ❌ {ev['statute_id']:>5} | {ev['name'][:60]}")
        for r in ev.get("reasoning", []):
            print(f"        → {r}")
