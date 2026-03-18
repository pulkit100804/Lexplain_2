import os
import sys

# ensure we import agents correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env for GEMINI_API_KEY
from dotenv import load_dotenv
load_dotenv()

from mcp.mcp_server import MCPServer
server = MCPServer()

raw_text = """
On the evening of 5th January, Arjun was seen leaving a restaurant with his colleague Vivek after a heated argument regarding money. CCTV footage from outside the restaurant shows both of them walking towards Vivek’s car at around 9:15 PM.

The next morning, Vivek’s car was found abandoned near a highway on the outskirts of the city. Later that day, Vivek’s dead body was discovered in a nearby forest area with signs of blunt force injury.

During investigation, it was found that Arjun had returned home late that night. Witnesses in his neighborhood reported seeing him enter his house around midnight. When questioned, Arjun failed to provide a clear explanation of his whereabouts between 9:30 PM and midnight.

A search of Arjun’s house led to the recovery of Vivek’s wristwatch and wallet from a drawer in his bedroom. No eyewitness has seen the actual incident of violence.
"""

cases_dir = "cases"
os.makedirs(cases_dir, exist_ok=True)

import agents.agent_0_ingest as a0
import agents.agent_1_normalize as a1
import agents.agent_2_segmentation as a2
import agents.agent_3_role_tagging as a3
import agents.agent_4_fact_graph as a4
import agents.agent_5_signal_extractor as a5
import agents.agent_6_statute_evaluator as a6

print("Agent 0")
r0 = a0.run(raw_text, cases_dir)
case_id = r0["case_id"]
case_dir = r0["case_dir"]

print("Agent 1-4")
a1.run(case_id, case_dir, cases_dir)
a2.run(case_id, case_dir, cases_dir)
a3.run(case_id, case_dir, cases_dir)
a4.run(case_id, case_dir, cases_dir)

print("Agent 5 — Legal Signal Extraction")
r5 = a5.run(case_id, case_dir, cases_dir)

print("Agent 6 — Statute Ingredient Evaluation")
r6 = a6.run(case_id, case_dir, cases_dir)

print(f"\nDone! Check output files in: cases/{case_id}/")
print(f"  - legal_signal_graph.json")
print(f"  - statute_evaluation.json")
