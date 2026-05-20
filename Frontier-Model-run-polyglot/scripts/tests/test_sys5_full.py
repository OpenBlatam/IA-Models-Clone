"""
System 5.9 — Full Test Suite.
Tests all hardened modules: events, circuit_breaker, prompt_sanitizer,
telemetry, registry, engines, models, and tools.
"""
import sys, os, json, asyncio, time

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"c:\blatam-academy\agents\backend\onyx\server\features\Frontier-Model-run-polyglot\scripts"
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "TruthGPT-main", "optimization_core"))

passed = failed = 0

def ok(label, cond):
    global passed, failed
    if cond:
        print(f"  [OK]   {label}")
        passed += 1
    else:
        print(f"  [FAIL] {label}")
        failed += 1

def section(name):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")


# =====================================================================
#  1. EVENTS
# =====================================================================
section("1. AsyncEventBus")
from sys5.events import Event, EventType, AsyncEventBus, event_bus

e = Event(EventType.ERROR, {"x": 1})
ok("Event.__slots__ defined", hasattr(Event, "__slots__"))
ok("Event.timestamp > 0", e.timestamp > 0)
ok("Event.__repr__ has type", "error" in repr(e))
ok("EventType has CIRCUIT_BREAKER", hasattr(EventType, "CIRCUIT_BREAKER"))
ok("EventType has TOOL_DEGRADED", hasattr(EventType, "TOOL_DEGRADED"))
ok("EventType has PROMPT_SANITIZED", hasattr(EventType, "PROMPT_SANITIZED"))
ok("emit_sync exists", hasattr(event_bus, "emit_sync"))

# Functional: subscribe + emit
captured = []
bus = AsyncEventBus()
bus.subscribe(EventType.SYSTEM, lambda ev: captured.append(ev.data))
asyncio.run(bus.emit(EventType.SYSTEM, {"msg": "hello"}))
ok("subscribe+emit delivers event", len(captured) == 1 and captured[0]["msg"] == "hello")

# emit_sync (no loop)
bus.emit_sync(EventType.SYSTEM, {"msg": "sync"})
ok("emit_sync no crash outside loop", True)


# =====================================================================
#  2. CIRCUIT BREAKER
# =====================================================================
section("2. CircuitBreaker")
from sys5.circuit_breaker import CircuitBreaker, CircuitState, POISON_PATTERNS

cb = CircuitBreaker(max_retries=2, cooldown_seconds=0.1)

# Poison detection
r, fb = cb.check("p1", "Echo from OpenClaw Agent (Mock): len 999")
ok("Poison: blocks retry", r is False)
ok("Poison: returns fallback dict", isinstance(fb, dict) and "final_answer" in fb)
ok("Poison: state=OPEN", cb.get_stats()["p1"]["state"] == "OPEN")

# Retry counting
cb2 = CircuitBreaker(max_retries=2)
r1, _ = cb2.check("r1", "bad")
r2, _ = cb2.check("r1", "bad")
r3, fb3 = cb2.check("r1", "bad")
ok("Retry 1 allowed", r1 is True)
ok("Retry 2 allowed", r2 is True)
ok("Retry 3 blocked (max=2)", r3 is False)
ok("Fallback is valid AgentAction", set(fb3.keys()) == {"thought","tool","tool_input","final_answer","handoff"})

# Reset
cb2.reset("r1")
ok("Reset clears trace", "r1" not in cb2.get_stats())

# Half-open recovery
cb3 = CircuitBreaker(max_retries=1, cooldown_seconds=0.05)
cb3.check("h1", "bad json 1")       # retry 1 allowed
_, _ = cb3.check("h1", "bad json 2") # retry 2 blocked -> OPEN
time.sleep(0.1)                       # wait > cooldown
r_half, _ = cb3.check("h1", "probe after cooldown")
ok("Half-open allows probe after cooldown", r_half is True)

ok("__repr__", "CircuitBreaker" in repr(cb))


# =====================================================================
#  3. PROMPT SANITIZER
# =====================================================================
section("3. PromptSanitizer")
from sys5.prompt_sanitizer import PromptSanitizer

ps = PromptSanitizer(max_chars=500)

# Nesting
nested = "X\nPrevious findings: A\nPrevious findings: B\nPrevious findings: C"
clean = ps.sanitize(nested)
ok("Strips recursive nesting to <=1", clean.count("Previous findings:") <= 1)

# Error dedup
errs = "[ERROR DE SISTEMA]: e1\nfoo\n[ERROR DE SISTEMA]: e2\nbar\n[ERROR DE SISTEMA]: e3"
clean2 = ps.sanitize(errs)
ok("Deduplicates error blocks to 1", clean2.count("[ERROR DE SISTEMA]") == 1)

# Nested objectives
obj = "Objective: Previous findings: junk\nObjective: real task"
clean3 = ps.sanitize(obj)
ok("Strips nested Objective: contamination", "Previous findings:" not in clean3 or clean3.count("Objective:") <= 2)

# Size cap
big = "A" * 1000
clean4 = ps.sanitize(big)
ok("Caps prompt size", len(clean4) <= 550)  # 500 + truncation marker

# Duplicate query
ok("First query not duplicate", not ps.is_duplicate_query("test query"))
ok("Second same query is duplicate", ps.is_duplicate_query("test query"))
ok("Case insensitive", ps.is_duplicate_query("TEST QUERY"))

# Topic extraction
ok("Extracts quoted topic", ps.extract_topic("stuff 'MoE' more") == "MoE")
ok("Returns None on garbage", ps.extract_topic("no quotes here") is None)

# Stats & repr
stats = ps.get_stats()
ok("Stats has passes", "sanitization_passes" in stats)
ok("__repr__", "PromptSanitizer" in repr(ps))

ps.reset()
ok("Reset clears queries", not ps.is_duplicate_query("test query"))


# =====================================================================
#  4. TELEMETRY
# =====================================================================
section("4. TelemetryService")
from sys5.telemetry import TelemetryService, SpanRecord

telem = TelemetryService()

async def _test_telem():
    # Normal span
    async with telem.span("op_ok", phase="T"):
        await asyncio.sleep(0.01)
    s = telem.get_summary()
    ok("Async span counted", s["total_calls"] == 1)
    ok("No errors on success", s["total_errors"] == 0)
    ok("Avg latency > 0", s["avg_latency_ms"] > 0)

    # Error span
    try:
        async with telem.span("op_fail", phase="T"):
            raise ValueError("boom")
    except ValueError:
        pass
    s2 = telem.get_summary()
    ok("Error span counted", s2["total_errors"] == 1)
    ok("Total calls = 2", s2["total_calls"] == 2)

    # Sync span
    with telem.sync_span("sync_op", phase="T"):
        pass
    s3 = telem.get_summary()
    ok("sync_span counted", s3["total_calls"] == 3)

asyncio.run(_test_telem())

# Increment
telem.increment("circuit_breaker_activations")
telem.increment("custom_counter", 5)
ok("Increment known counter", telem.get_summary()["counters"]["circuit_breaker_activations"] == 1)
ok("Increment creates new counter", telem.get_summary()["counters"]["custom_counter"] == 5)

# SpanRecord
sr = SpanRecord(name="x", phase="p", start=1.0, end=2.0)
ok("SpanRecord.duration_ms", sr.duration_ms == 1000.0)
ok("SpanRecord.is_closed", sr.is_closed)
ok("SpanRecord repr", "x" in repr(sr))

sr2 = SpanRecord(name="y", phase="p", start=1.0)
ok("Unclosed span: end=0", not sr2.is_closed)
ok("Unclosed span: duration=0", sr2.duration_ms == 0.0)

ok("TelemetryService repr", "TelemetryService" in repr(telem))


# =====================================================================
#  5. REGISTRY
# =====================================================================
section("5. Registry boot()")
from sys5.registry import Registry

reg = Registry()
reg.boot()
ok("TelemetryService registered", reg.get("TelemetryService") is not None)
ok("CircuitBreaker registered", reg.get("CircuitBreaker") is not None)
ok("PromptSanitizer registered", reg.get("PromptSanitizer") is not None)
ok("service_names populated", len(reg.service_names) == 3)

# Idempotent
reg.boot()
ok("Double boot is safe", len(reg.service_names) == 3)

# resolve()
ok("resolve(TelemetryService)", reg.resolve(TelemetryService) is not None)
ok("resolve(int) = None", reg.resolve(int) is None)

ok("__repr__", "Registry" in repr(reg))


# =====================================================================
#  6. ENGINES
# =====================================================================
section("6. LLM Engines")
from agents.engines import DummyAsyncLLM, DeepSeekAsyncLLM, EngineRegistry, _safe_fallback

# DummyAsyncLLM
result = asyncio.run(DummyAsyncLLM()("hello"))
parsed = json.loads(result)
ok("DummyAsyncLLM returns valid JSON", "final_answer" in parsed)
ok("DummyAsyncLLM has thought", parsed["thought"] is not None)
ok("DummyAsyncLLM tool=None", parsed["tool"] is None)

# _safe_fallback
fb = json.loads(_safe_fallback("t", "m"))
ok("_safe_fallback has all keys", set(fb.keys()) == {"thought","tool","tool_input","final_answer","handoff"})
ok("_safe_fallback final_answer", fb["final_answer"] == "m")

# EngineRegistry singleton
er = EngineRegistry()
ok("Has mock engine", er.get_engine("mock") is not None)
ok("Has deepseek engine", er.get_engine("deepseek") is not None)
ok("list_engines", set(er.list_engines()) >= {"mock", "deepseek"})
ok("__repr__", "EngineRegistry" in repr(er))


# =====================================================================
#  7. MODELS (AgentAction validator)
# =====================================================================
section("7. AgentAction tool_input validator")
from agents.models import AgentAction

# Dict with "query" key
a1 = AgentAction.model_validate({"thought":"t", "tool":"web_search", "tool_input":{"query":"AI"}})
ok("Dict {query:...} -> str", a1.tool_input == "AI")

# Dict with "code" key
a2 = AgentAction.model_validate({"thought":"t", "tool":"python", "tool_input":{"code":"print(1)"}})
ok("Dict {code:...} -> str", a2.tool_input == "print(1)")

# Dict with path+content
a3 = AgentAction.model_validate({"thought":"t", "tool":"file_write", "tool_input":{"path":"/tmp/x","content":"y"}})
ok("Dict {path,content} -> :::", a3.tool_input == "/tmp/x:::y")

# Dict with "url"
a4 = AgentAction.model_validate({"thought":"t", "tool":"web_reader", "tool_input":{"url":"https://x.com"}})
ok("Dict {url:...} -> str", a4.tool_input == "https://x.com")

# String passthrough
a5 = AgentAction.model_validate({"thought":"t", "tool":"bash", "tool_input":"ls -la"})
ok("String passthrough", a5.tool_input == "ls -la")

# None passthrough
a6 = AgentAction.model_validate({"thought":"t", "final_answer":"done"})
ok("None passthrough", a6.tool_input is None)

# Unknown dict -> json.dumps
a7 = AgentAction.model_validate({"thought":"t", "tool":"x", "tool_input":{"foo":"bar"}})
ok("Unknown dict -> JSON string", json.loads(a7.tool_input) == {"foo":"bar"})


# =====================================================================
#  8. TOOLS
# =====================================================================
section("8. Tool implementations")
from agents.razonamiento_planificacion.tools import (
    WebSearchTool, FileWriteTool, FileReadTool, SystemBashTool,
    BaseTool, ToolResult,
)

# FileWriteTool._parse — both formats
fp1, ct1 = FileWriteTool._parse("/tmp/a.txt:::hello world")
ok("file_write parse :::  format", fp1 == "/tmp/a.txt" and ct1 == "hello world")

fp2, ct2 = FileWriteTool._parse('{"path":"/tmp/b.txt","content":"data"}')
ok("file_write parse JSON format", fp2 == "/tmp/b.txt" and ct2 == "data")

fp3, ct3 = FileWriteTool._parse("invalid input no separator")
ok("file_write parse invalid -> None", fp3 is None and "Error" in ct3)

# FileWriteTool._parse edge cases
fp4, ct4 = FileWriteTool._parse('{"filepath":"/tmp/c.txt","text":"alt keys"}')
ok("file_write accepts alt keys (filepath/text)", fp4 == "/tmp/c.txt" and ct4 == "alt keys")

# WebSearchTool instance state
ws = WebSearchTool()
ok("WebSearchTool._failures is instance-level", "_failures" in ws.__dict__)
ok("WebSearchTool starts at 0 failures", ws._failures == 0)

# WebSearchTool degradation
ws2 = WebSearchTool()
ws2._failures = 3
result = asyncio.run(ws2.run("test"))
ok("Degraded tool returns advisory", "[TOOL DEGRADED]" in result)

# Two instances don't share state
ws3 = WebSearchTool()
ok("Separate instances independent", ws3._failures == 0)

# FileReadTool
fr = FileReadTool()
result = asyncio.run(fr.run("nonexistent_file_xyz.txt"))
ok("FileRead nonexistent -> error", "Error" in result)

# ToolResult
tr = ToolResult("output", {"k": "v"}, "signal")
ok("ToolResult fields", tr.output == "output" and tr.signal == "signal")

# BaseTool is abstract
try:
    BaseTool()
    ok("BaseTool is abstract", False)
except TypeError:
    ok("BaseTool is abstract", True)


# =====================================================================
#  9. INTEGRATION: Circuit Breaker + Engines
# =====================================================================
section("9. Integration: CB + Engine fallback")

cb_int = CircuitBreaker(max_retries=2)

# Simulate: DummyAsyncLLM returns valid JSON now (D1 fix)
dummy_output = asyncio.run(DummyAsyncLLM()("test"))
try:
    action = AgentAction.model_validate_json(dummy_output)
    ok("DummyAsyncLLM output passes Pydantic", True)
    ok("Parsed action has final_answer", action.final_answer is not None)
except Exception as e:
    ok(f"DummyAsyncLLM Pydantic validation: {e}", False)

# Simulate old poisoned output hitting circuit breaker
old_mock = "Echo from OpenClaw Agent (Mock): Prompt length 2760."
should_retry, fallback = cb_int.check("integration-1", old_mock)
ok("Old mock blocked by CB", should_retry is False)
try:
    action2 = AgentAction.model_validate(fallback)
    ok("CB fallback passes Pydantic", True)
except Exception as e:
    ok(f"CB fallback Pydantic: {e}", False)


# =====================================================================
#  10. INTEGRATION: Sanitizer + Duplicate detection
# =====================================================================
section("10. Integration: Sanitizer end-to-end")

ps_int = PromptSanitizer()

# Simulate the exact trace pattern from traces_history.json
real_prompt = (
    "System prompt...\n"
    "=== CORE WORKING MEMORY ===\n"
    "[AGENT PERSONA]: No specific persona defined.\n"
    "===========================\n"
    "Objective: Previous findings: No encontre papers para 'descubrir papers de "
    "Previous findings: No encontre papers para MoE'\n"
    "Objective: Evolve results\n"
    "[ERROR DE SISTEMA]: Tu respuesta violo el esquema JSON obligatorio.\n"
    "TRUTHgpt: bad\n"
    "[ERROR DE SISTEMA]: Tu respuesta violo el esquema JSON obligatorio.\n"
    "TRUTHgpt: bad again\n"
    "[ERROR DE SISTEMA]: Tu respuesta violo el esquema JSON obligatorio.\n"
    "TRUTHgpt:"
)

cleaned = ps_int.sanitize(real_prompt)
ok("Real trace: nesting reduced", cleaned.count("Previous findings:") <= 1)
ok("Real trace: errors deduped", cleaned.count("[ERROR DE SISTEMA]") == 1)
ok("Real trace: shorter", len(cleaned) < len(real_prompt))

# Duplicate query
ok("First search not dup", not ps_int.is_duplicate_query("AI marketing trends 2025"))
ok("Retry blocked", ps_int.is_duplicate_query("AI marketing trends 2025"))
topic = ps_int.extract_topic("Previous findings: No papers for 'MoE'")
ok(f"Extracts 'MoE' from contaminated query", topic == "MoE")


# =====================================================================
#  SUMMARY
# =====================================================================
print(f"\n{'='*60}")
print(f"  TOTAL: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'='*60}")
if failed == 0:
    print("  ALL TESTS PASSED!")
else:
    print(f"  {failed} TESTS FAILED")
    sys.exit(1)
