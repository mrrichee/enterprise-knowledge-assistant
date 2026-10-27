"""
check_models.py - find out which Groq / Gemini models work with YOUR keys.

Run from this folder:   python check_models.py
It lists the models each key can see, sends a tiny test message to each, and
prints which ones answered. Put a working name in .env if you want to pin it
(GROQ_MODEL=... or GEMINI_MODEL=...), or leave them empty for auto-detect.
"""
import io
import sys

# Make Windows consoles print unicode safely
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.core.config import get_config
from app.core.exceptions import APIKeyError

cfg = get_config()
print("Keys found in .env:",
      "GROQ" if cfg.GROQ_API_KEY else "-no groq key-",
      "| GEMINI" if cfg.GEMINI_API_KEY else "| -no gemini key-")

from app.services.llm_service import LLMService
try:
    svc = LLMService()
except APIKeyError:
    sys.exit("No GROQ_API_KEY or GEMINI_API_KEY found in .env")

working = {}
for provider, model, ok, detail in svc.diagnose():
    print(f"  [{'OK ' if ok else 'FAIL'}] {provider:7} {model:45} {detail[:110]}")
    if ok and model != "(list models)":
        working.setdefault(provider, []).append(model)

print()
if not working:
    print("No model answered. Read the FAIL lines above:")
    print("  401/403 or 'API key not valid' -> key is wrong/revoked (make a new one)")
    print("  429                            -> rate/daily limit reached, wait or use the other provider")
    print("  'Network error'                -> internet/VPN/firewall blocks the API")
else:
    for provider, models in working.items():
        env = "GROQ_MODEL" if provider == "groq" else "GEMINI_MODEL"
        print(f"{provider}: {len(models)} working model(s). Best: {models[0]}")
        print(f"   optional .env line:  {env}={models[0]}")
    print("\nThe app picks a working one automatically, so pinning is optional.")
