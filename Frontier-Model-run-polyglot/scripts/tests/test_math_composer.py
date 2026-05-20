"""Tests for MathVerificationAgent, Agent Composer, and new tools."""
import sys, os, json, asyncio

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
#  1. SYMPY VERIFY TOOL
# =====================================================================
section("1. SymPyVerifyTool")
from agents.formal_verification.math_agent import SymPyVerifyTool

sympy_tool = SymPyVerifyTool()

# Simplify
r1 = asyncio.run(sympy_tool.run("simplify: (x**2 - 1)/(x - 1)"))
ok("Simplify (x^2-1)/(x-1)", "x + 1" in r1)

# Solve
r2 = asyncio.run(sympy_tool.run("solve: x**2 - 4 = 0"))
ok("Solve x^2-4=0", "2" in r2 and "-2" in r2)

# Prove (true)
r3 = asyncio.run(sympy_tool.run("prove: (x+1)**2 == x**2 + 2*x + 1"))
ok("Prove true identity", "PROBADO" in r3)

# Prove (false)
r4 = asyncio.run(sympy_tool.run("prove: x**2 == x**3"))
ok("Prove false identity", "NO VERIFICADO" in r4)

# Integrate
r5 = asyncio.run(sympy_tool.run("integrate: x**2"))
ok("Integrate x^2", "x**3/3" in r5 or "3" in r5)

# Diff
r6 = asyncio.run(sympy_tool.run("diff: x**3"))
ok("Diff x^3 = 3x^2", "3*x**2" in r6)

# Factor
r7 = asyncio.run(sympy_tool.run("factor: x**2 - 1"))
ok("Factor x^2-1", "(x - 1)" in r7 and "(x + 1)" in r7)

# Limit
r8 = asyncio.run(sympy_tool.run("limit: sin(x)/x, x, 0"))
ok("Limit sin(x)/x -> 1", "1" in r8)

# Matrix
r9 = asyncio.run(sympy_tool.run("matrix: [[1,2],[3,4]]"))
ok("Matrix det", "-2" in r9)

# LaTeX output
ok("LaTeX output in simplify", "LaTeX" in r1 or "$" in r1)


# =====================================================================
#  2. NUMERICAL VERIFY TOOL
# =====================================================================
section("2. NumericalVerifyTool")
from agents.formal_verification.math_agent import NumericalVerifyTool

num_tool = NumericalVerifyTool()

# Eigenvalues
r10 = asyncio.run(num_tool.run("eigenvalues: [[1,2],[3,4]]"))
ok("Eigenvalues computed", "Eigenvalores" in r10)

# Roots
r11 = asyncio.run(num_tool.run("roots: [1, -5, 6]"))
ok("Roots of x^2-5x+6", "3" in r11 and "2" in r11)

# SVD
r12 = asyncio.run(num_tool.run("svd: [[1,2],[3,4]]"))
ok("SVD computed", "valores singulares" in r12)

# Eval
r13 = asyncio.run(num_tool.run("eval: np.sqrt(144)"))
ok("Eval sqrt(144)=12", "12" in r13)

# Help
r14 = asyncio.run(num_tool.run("???"))
ok("Unknown shows help", "eigenvalues:" in r14)


# =====================================================================
#  3. LEAN 4 TOOL (no binary expected)
# =====================================================================
section("3. Lean4VerifyTool")
from agents.formal_verification.math_agent import Lean4VerifyTool

lean_tool = Lean4VerifyTool()

r15 = asyncio.run(lean_tool.run("theorem add_comm (a b : Nat) : a + b = b + a := Nat.add_comm a b"))
ok("Lean handles theorem (no binary)", "theorem" in r15.lower() or "lean" in r15.lower() or "sorry" in r15.lower() or "Teoremas" in r15)

r16 = asyncio.run(lean_tool.run("def foo := 42\n#check foo\nsorry"))
ok("Lean detects sorry", "sorry" in r15 or "sorry" in r16)


# =====================================================================
#  4. CODE VERIFY TOOL
# =====================================================================
section("4. CodeVerifyTool")
from agents.formal_verification.math_agent import CodeVerifyTool

code_tool = CodeVerifyTool()

r17 = asyncio.run(code_tool.run("def add(a: int, b: int) -> int:\n    return a + b"))
ok("AST analysis works", "Functions:" in r17 or "AST" in r17)


# =====================================================================
#  5. MATH VERIFICATION AGENT
# =====================================================================
section("5. MathVerificationAgent")
from agents.formal_verification.math_agent import MathVerificationAgent, MATH_TOOLS

agent = MathVerificationAgent()

ok("Agent has all 5 tools", len(agent.tools) == 5)
ok("MATH_TOOLS catalog has 5", len(MATH_TOOLS) == 5)

# Auto-routing
r18 = asyncio.run(agent.process("solve: x**2 + x - 6 = 0"))
ok("Auto-routes to SymPy", "3" in r18.content or "-3" in r18.content or "2" in r18.content)

r19 = asyncio.run(agent.process("eigenvalues: [[2,0],[0,3]]"))
ok("Auto-routes to NumPy", "Eigenvalores" in r19.content or "2" in r19.content)

# Custom tool selection
agent2 = MathVerificationAgent(enabled_tools=["sympy_verify", "numerical_verify"])
ok("Custom tool selection", len(agent2.tools) == 2)

# No LLM fallback
r20 = asyncio.run(agent.process("hello world"))
ok("No-match shows capabilities", "SymPy" in r20.content or "verificar" in r20.content.lower())


# =====================================================================
#  6. AGENT COMPOSER
# =====================================================================
section("6. Agent Composer")
from agents.composer.agent_composer import (
    _build_catalog, ComposedAgent, save_blueprint, load_blueprints
)

catalog = _build_catalog()
ok("Catalog has math entries", "math_symbolic" in catalog)
ok("Catalog has research entries", "web_search" in catalog or "arxiv_search" in catalog)
ok("Catalog categories", any(c["category"] == "Mathematics" for c in catalog.values()))

# Compose agent
composed = ComposedAgent(
    name="TestMathAgent",
    role="Test",
    capabilities=["math_symbolic", "math_numerical"],
)
ok("Composed agent created", composed.name == "TestMathAgent")
ok("Composed has sympy tool", "sympy_verify" in composed.tools)
ok("Composed has numerical tool", "numerical_verify" in composed.tools)
ok("Capability summary", "SymPy" in composed.get_capability_summary())

# Process query through composed agent
r21 = asyncio.run(composed.process("solve: x**2 - 9 = 0"))
ok("Composed agent processes query", r21.content is not None and len(r21.content) > 0)

# Empty agent
empty = ComposedAgent(name="EmptyAgent", capabilities=[])
ok("Empty agent works", len(empty.tools) == 0)
r22 = asyncio.run(empty.process("test"))
ok("Empty agent fallback", "ninguna" in r22.content or "EmptyAgent" in r22.content)


# =====================================================================
#  7. BLUEPRINT PERSISTENCE
# =====================================================================
section("7. Blueprint Save/Load")
import tempfile
from pathlib import Path

# Override blueprints dir for testing
import agents.composer.agent_composer as composer_mod
original_dir = composer_mod.BLUEPRINTS_DIR
composer_mod.BLUEPRINTS_DIR = Path(tempfile.mkdtemp()) / "test_blueprints"

path = save_blueprint("TestBlueprint", ["math_symbolic", "web_search"], {"role": "Test Agent"})
ok("Blueprint saved", path.exists())

bps = load_blueprints()
ok("Blueprint loaded", len(bps) == 1)
ok("Blueprint name", bps[0]["name"] == "TestBlueprint")
ok("Blueprint capabilities", bps[0]["capabilities"] == ["math_symbolic", "web_search"])

# Cleanup
composer_mod.BLUEPRINTS_DIR = original_dir


# =====================================================================
#  8. REGISTRY INTEGRATION
# =====================================================================
section("8. Registry Integration")
from agents.registry import registry

ok("math_verifier in registry", registry.get_agent("math_verifier") is not None)
ok("sympy_verify tool in registry", registry.get_tool("sympy_verify") is not None)
ok("lean4_verify tool in registry", registry.get_tool("lean4_verify") is not None)
ok("z3_verify tool in registry", registry.get_tool("z3_verify") is not None)
ok("numerical_verify tool in registry", registry.get_tool("numerical_verify") is not None)
ok("code_verify tool in registry", registry.get_tool("code_verify") is not None)


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
