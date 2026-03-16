# Lexplain — Indian Law Multi-Agent Case Analysis

Lexplain is an open-source multi-agent pipeline for Indian criminal law analysis. It ingests a raw case description, runs it through 11 sequential agents (powered by a local MCP server), and produces structured JSON artifacts covering role-tagged graphs, fact-event graphs, statute candidates, ingredient evaluations, loophole maps, and legal arguments.

Gemini 1.5 Flash is used for argument generation when `GEMINI_API_KEY` is set. All other stages are fully deterministic.

---

## Quick Start

```bash
# 1. Clone and set up a virtual environment
git clone https://github.com/your-org/Lexplain_2.git
cd Lexplain_2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set Gemini API key for LLM-powered argument generation
export GEMINI_API_KEY="your-key-here"
# Or create a .env file:  echo 'GEMINI_API_KEY=your-key-here' > .env

# 4. Run the pipeline
python main.py
```

Enter a case description when prompted, or press Enter twice to use the built-in sample case.

---

## Run Tests

```bash
pytest tests/test_pipeline_smoke.py -v
```

---

## Project Structure

```
├── main.py                        Entry point — runs the full 11-agent pipeline
├── requirements.txt
├── mcp/                           Local MCP server + tool registry
│   ├── mcp_server.py              MCP server (tool runner, Gemini adapter)
│   ├── mcp_client.py              Thin client used by all agents
│   ├── tool_registry.py           Tool name → callable mapping
│   └── tools/                     One file per registered MCP tool
│       ├── extract_sentences.py
│       ├── tag_roles.py
│       ├── build_fact_graph.py
│       ├── identify_statutes.py
│       ├── evaluate_ingredients.py
│       ├── search_precedents.py   ← RAG PLACEHOLDER (always empty)
│       ├── compare_precedents.py
│       ├── mine_loopholes.py
│       └── generate_argument.py   ← Calls Gemini or deterministic fallback
├── agents/                        11 agents (agent_0 … agent_10)
├── knowledge_base/
│   └── ingredients_ipc.json       IPC section ingredients for evaluation
├── data/cases/                    Runtime artifacts (git-ignored per case)
├── utils/
│   ├── text_utils.py
│   ├── graph_utils.py
│   └── provenance.py
└── tests/
    └── test_pipeline_smoke.py
```

---

## Pipeline Stages

| # | Agent | MCP Tool | Output |
|---|-------|----------|--------|
| 0 | Ingest | (direct) | `raw.txt`, `metadata.json` |
| 1 | Normalize | (direct) | `normalized_text.txt` |
| 2 | Segmentation | `extract_sentences_v1` | `document_graph.json` |
| 3 | Role Tagging | `tag_legal_roles_v1` | `role_tagged_graph.json` |
| 4 | Fact Graph | `build_fact_graph_v1` | `fact_event_graph.json` |
| 5 | Statute Candidates | `identify_statutes_v1` | `statute_candidates.json` |
| 6 | Ingredient Engine | `evaluate_ingredients_v1` | `ingredient_report.json` |
| 7 | Retrieval | `search_precedents_v1` | `precedent_matches.json` |
| 8 | Comparator | `compare_precedents_v1` | `comparator_report.json` |
| 9 | Loophole Miner | `mine_loopholes_v1` | `loophole_map.json` |
| 10 | Argument Generator | `generate_argument_v1` | `argument_package.json` |

---

## RAG / Precedent Retrieval

> **RAG is intentionally left as a placeholder.**  
> `search_precedents_v1` always returns an empty list in this prototype.  
> To add real retrieval, replace the stub in `mcp/tools/search_precedents.py`  
> with an Elasticsearch, Qdrant, or other vector-search client.

---

## MCP Provenance Contract

Every JSON artifact produced by the pipeline must contain a `provenance` block:

```json
{
  "provenance": {
    "trace_id": "<uuid4>",
    "tool_name": "extract_sentences_v1",
    "tool_version": "1.0.0",
    "model_version": null,
    "timestamp": "2024-03-15T10:00:00+00:00",
    "input_refs": ["path/to/input"]
  }
}
```

**Rules:**
- Agents **MUST NOT** call Gemini directly. All LLM calls go through `MCPClient → MCPServer → _call_gemini`.
- Every MCP tool must call `make_provenance()` and include the result in its output JSON.
- `trace_id` is injected by `MCPServer.call_tool()` and propagated through the chain.
- `model_version` is `null` for deterministic tools; `"gemini-1.5-flash"` when Gemini is used.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Optional | Enables Gemini 1.5 Flash for argument generation. If absent, deterministic fallback is used. |

---

## License

MIT License. See source file headers.