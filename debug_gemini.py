"""Debug: test the full extraction prompt with Gemini."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
from mcp.mcp_server import _call_gemini
from utils.offence_category_index import get_all_categories
from mcp.tools.extract_legal_signals import _EXTRACTION_PROMPT, ALLOWED_EVENT_TYPES, _parse_llm_response

cats = get_all_categories()
print(f"Categories count: {len(cats)}")
prompt = _EXTRACTION_PROMPT.format(event_types=", ".join(ALLOWED_EVENT_TYPES), categories=", ".join(cats))
print(f"Prompt length: {len(prompt)} chars")

case_text = "The complainant alleged that the accused Ram cheated him of Rs 50,000 by making false promises about a job. Ram intentionally tricked the complainant and ran away with the money."
raw = _call_gemini(prompt, f"Case Facts:\n{case_text}", temperature=0.1, max_tokens=4096)
print(f"Raw response length: {len(raw)}")
print(f"Raw response:\n{raw[:2000]}")

parsed = _parse_llm_response(raw)
if parsed:
    print(f"\nParsed successfully!")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
else:
    print("\nParsed: None (FAILED)")
