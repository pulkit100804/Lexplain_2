import os
import sys

# ensure we import agents correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.mcp_server import MCPServer
server = MCPServer()

raw_text = "The complainant alleged that the accused Ram cheated him of Rs 50,000 by making false promises about a job. Ram intentionally tricked the complainant and ran away with the money."

cases_dir = "cases"
os.makedirs(cases_dir, exist_ok=True)

import agents.agent_0_ingest as a0
import agents.agent_1_normalize as a1
import agents.agent_2_segmentation as a2
import agents.agent_3_role_tagging as a3
import agents.agent_4_fact_graph as a4
import agents.agent_5_statute_candidates as a5
import agents.agent_6_ingredient_engine as a6
import agents.agent_7_retrieval as a7
import agents.agent_8_comparator as a8
import agents.agent_9_loophole_miner as a9
import agents.agent_10_argument_generator as a10

print("Agent 0")
r0 = a0.run(raw_text, cases_dir)
case_id = r0["case_id"]
case_dir = r0["case_dir"]

print("Agent 1-4")
a1.run(case_id, case_dir, cases_dir)
a2.run(case_id, case_dir, cases_dir)
a3.run(case_id, case_dir, cases_dir)
a4.run(case_id, case_dir, cases_dir)

print("Agent 5")
a5.run(case_id, case_dir, cases_dir)

print("Agent 6")
a6.run(case_id, case_dir, cases_dir)

print("Agent 7")
a7.run(case_id, case_dir, cases_dir)

print("Agent 8")
a8.run(case_id, case_dir, cases_dir)

print("Agent 9")
a9.run(case_id, case_dir, cases_dir)

print("Agent 10")
a10.run(case_id, case_dir, cases_dir)

print("Finished all 10 agents.")
