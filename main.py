"""
Lexplain — Main entry point.
MIT License | See README for MCP provenance contract.

Usage:
    python main.py

Prompts the user for a case description, runs the 11-agent pipeline sequentially,
and stores all JSON artifacts in data/cases/{case_id}/.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present (for GEMINI_API_KEY)
load_dotenv()

# Repository root
ROOT = Path(__file__).parent
CASES_DIR = str(ROOT / "data" / "cases")
os.makedirs(CASES_DIR, exist_ok=True)

# Agent imports
import agents.agent_0_ingest as agent_0
import agents.agent_1_normalize as agent_1
import agents.agent_2_segmentation as agent_2
import agents.agent_3_role_tagging as agent_3
import agents.agent_4_fact_graph as agent_4
import agents.agent_5_signal_extractor as agent_5
import agents.agent_6_statute_evaluator as agent_6
import agents.agent_7_retrieval as agent_7
import agents.agent_8_comparator as agent_8
import agents.agent_9_loophole_miner as agent_9
import agents.agent_10_argument_generator as agent_10


def main() -> None:
    print("=" * 60)
    print("  LEXPLAIN — Indian Law Multi-Agent Case Analysis")
    print("  Powered by MCP + Gemini (fallback: deterministic)")
    print("=" * 60)

    # Step 1: Prompt for case text
    print("\nEnter the case description (press Enter twice when done):\n")
    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
    except EOFError:
        pass
    case_text = "\n".join(lines).strip()

    if not case_text:
        print("No case text provided. Using sample case for demonstration.")
        case_text = (
            "On 15 March 2024, the complainant alleged that the accused cheated and defrauded him "
            "of Rs. 5,00,000 by making false representations about a property investment scheme. "
            "The accused dishonestly induced the complainant to transfer the amount by signing a "
            "fraudulent agreement. The complainant also stated that the accused threatened him when "
            "he demanded the money back, causing him alarm and fear. The accused intimidated the "
            "complainant by saying 'do not dare to go to the police or you will face consequences'. "
            "The complainant filed a complaint with the police on 20 March 2024."
        )
        print(f"\n[Using sample case]\n{case_text}\n")

    # Ask for role
    print("\nGenerate arguments for: (1) Both prosecution & defense  (2) Prosecution only  (3) Defense only")
    try:
        choice = input("Enter 1, 2, or 3 [default: 1]: ").strip()
    except EOFError:
        choice = ""
    role_map = {"1": "both", "2": "prosecution", "3": "defense", "": "both"}
    role = role_map.get(choice, "both")

    print(f"\nRunning pipeline for role='{role}'...\n")

    # ── Pipeline ──────────────────────────────────────────────────────────────

    # Agent 0: Ingest
    r0 = agent_0.run(case_text, CASES_DIR)
    case_id = r0["case_id"]
    case_dir = r0["case_dir"]

    # Agent 1: Normalize
    agent_1.run(case_id, case_dir, CASES_DIR)

    # Agent 2: Segmentation
    agent_2.run(case_id, case_dir, CASES_DIR)

    # Agent 3: Role Tagging
    agent_3.run(case_id, case_dir, CASES_DIR)

    # Agent 4: Fact Event Graph
    agent_4.run(case_id, case_dir, CASES_DIR)

    # Agent 5: Statute Candidates
    agent_5.run(case_id, case_dir, CASES_DIR)

    # Agent 6: Ingredient Engine
    agent_6.run(case_id, case_dir, CASES_DIR)

    # Agent 7: Retrieval (placeholder)
    agent_7.run(case_id, case_dir, CASES_DIR)

    # Agent 8: Comparator
    agent_8.run(case_id, case_dir, CASES_DIR)

    # Agent 9: Loophole Miner
    agent_9.run(case_id, case_dir, CASES_DIR)

    # Agent 10: Argument Generator (Gemini or fallback)
    r10 = agent_10.run(case_id, case_dir, CASES_DIR, role=role)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Pipeline complete! Case ID: {case_id}")
    print(f"  Artifacts saved in: {case_dir}")
    print(f"  LLM used: {'Gemini' if r10.get('llm_used') else 'Deterministic fallback'}")
    print("=" * 60)
    print("\nGenerated files:")
    for fname in sorted(os.listdir(case_dir)):
        fpath = os.path.join(case_dir, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:40s} ({size:,} bytes)")
    print()


if __name__ == "__main__":
    main()
