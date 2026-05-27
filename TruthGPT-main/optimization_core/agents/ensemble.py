"""
Ensemble LLM strategies: consensus, parallel, race, majority, debate, bayesian.

Designed for unit testing without live API calls — merge/run helpers are pure or injectable.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

# Run tuple: (engine_key, model_name, raw_text, elapsed_sec, token_estimate)
EngineRun = Tuple[str, str, str, float, int]

ALL_ENSEMBLE_MODES = frozenset(
    {"consensus", "parallel", "race", "majority", "debate", "bayesian"}
)
MULTI_ENGINE_MODES = frozenset(
    {"consensus", "parallel", "majority", "debate", "bayesian"}
)


def parse_agent_json(raw: str) -> Dict[str, Any]:
    """Best-effort parse of an AgentAction JSON payload."""
    if not raw or not str(raw).strip():
        return {"thought": "", "final_answer": ""}
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, str):
            data = json.loads(data)
        return data if isinstance(data, dict) else {"thought": text[:500], "final_answer": text}
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    return {"thought": text[:500], "final_answer": text}


def _extract_thought(data: Dict[str, Any]) -> str:
    return str(
        data.get("thought")
        or data.get("razonamiento")
        or data.get("reasoning")
        or ""
    ).strip()


def _extract_final(data: Dict[str, Any]) -> str:
    ans = str(
        data.get("final_answer")
        or data.get("respuesta_final")
        or data.get("answer")
        or data.get("response")
        or data.get("output")
        or data.get("text")
        or data.get("message")
        or ""
    ).strip()
    
    if not ans:
        if data.get("tool"):
            ans = f"[TOOL_CALL] {data['tool']}: {data.get('tool_input', '')}"
        else:
            rest = {k: v for k, v in data.items() if k not in ("thought", "razonamiento", "reasoning", "metadata", "confidence")}
            if rest:
                ans = json.dumps(rest, ensure_ascii=False)
                
    return ans.strip()


def _extract_confidence(data: Dict[str, Any]) -> float:
    for key in ("confidence", "score", "certainty"):
        if key in data:
            try:
                return max(0.0, min(1.0, float(data[key])))
            except (TypeError, ValueError):
                pass
    meta = data.get("metadata")
    if isinstance(meta, dict):
        for key in ("confidence", "score", "certainty"):
            if key in meta:
                try:
                    return max(0.0, min(1.0, float(meta[key])))
                except (TypeError, ValueError):
                    pass
    return 0.5


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())[:500]


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _normalize_for_compare(a), _normalize_for_compare(b)).ratio()


def _cluster_by_similarity(
    items: List[Tuple[str, str]],
    threshold: float = 0.55,
) -> List[List[Tuple[str, str]]]:
    """Group (engine_key, final_answer) by textual similarity."""
    clusters: List[List[Tuple[str, str]]] = []
    for key, final in items:
        placed = False
        for cluster in clusters:
            if _similarity(final, cluster[0][1]) >= threshold:
                cluster.append((key, final))
                placed = True
                break
        if not placed:
            clusters.append([(key, final)])
    return clusters


def _pick_largest_cluster(clustered: List[List[Tuple[str, str]]]) -> Tuple[str, str]:
    best = max(clustered, key=len)
    key, final = max(best, key=lambda x: len(x[1]))
    return key, final


def merge_ensemble_responses(mode: str, runs: List[EngineRun]) -> str:
    """Merge per-engine runs into one AgentAction JSON string."""
    mode = (mode or "consensus").lower().strip()
    if mode not in ALL_ENSEMBLE_MODES:
        mode = "consensus"

    parsed: List[Tuple[str, str, Dict[str, Any]]] = []
    for key, model, text, _elapsed, _tokens in runs:
        if not text:
            continue
        data = parse_agent_json(text)
        parsed.append((key, model, data))

    if not parsed:
        return json.dumps(
            {
                "thought": "Ensemble: no engine returned a response.",
                "final_answer": "Error: all engines failed in ensemble call.",
                "metadata": {"ensemble_mode": mode, "engines": []},
            },
            ensure_ascii=False,
        )

    engine_list = [f"{k} ({m})" for k, m, _ in parsed]
    thoughts_header = f"Ensemble [{mode}] from {', '.join(engine_list)}"

    if mode == "parallel":
        return _merge_parallel(parsed, thoughts_header, mode)

    if mode == "race":
        return _merge_race(runs, parsed, thoughts_header, mode)

    if mode == "majority":
        return _merge_majority(parsed, thoughts_header, mode)

    if mode == "debate":
        return _merge_debate(parsed, thoughts_header, mode)

    if mode == "bayesian":
        return _merge_bayesian(parsed, thoughts_header, mode)

    return _merge_consensus(parsed, thoughts_header, mode)


def _merge_parallel(
    parsed: List[Tuple[str, str, Dict[str, Any]]],
    header: str,
    mode: str,
) -> str:
    """Keep every engine answer visible in final_answer."""
    sections = []
    thought_lines = []
    for key, model, data in parsed:
        thought = _extract_thought(data)
        final = _extract_final(data)
        thought_lines.append(f"[{key}/{model}] {thought}".strip())
        sections.append(f"### {key} ({model})\n{final or '(no final_answer)'}")

    return json.dumps(
        {
            "thought": f"{header}:\n" + "\n".join(thought_lines),
            "final_answer": "\n\n---\n\n".join(sections),
            "metadata": {
                "ensemble_mode": mode,
                "engines": [k for k, _, _ in parsed],
                "parallel_outputs": {k: _extract_final(d) for k, _, d in parsed},
            },
        },
        ensure_ascii=False,
    )


def _merge_race(
    runs: List[EngineRun],
    parsed: List[Tuple[str, str, Dict[str, Any]]],
    header: str,
    mode: str,
) -> str:
    """Winner is the run with minimum elapsed time among successful responses."""
    successful = [r for r in runs if r[2]]
    if not successful:
        return json.dumps(
            {
                "thought": f"{header}: race had no finisher.",
                "final_answer": "Error: race mode — no engine finished in time.",
                "metadata": {"ensemble_mode": mode, "engines": []},
            },
            ensure_ascii=False,
        )
    winner = min(successful, key=lambda r: r[3])
    key, model, text, elapsed, tokens = winner
    data = parse_agent_json(text)
    return json.dumps(
        {
            "thought": f"{header} — winner [{key}/{model}] in {elapsed:.2f}s:\n"
            + _extract_thought(data),
            "final_answer": _extract_final(data) or text,
            "metadata": {
                "ensemble_mode": mode,
                "winner": key,
                "winner_model": model,
                "elapsed": elapsed,
                "tokens": tokens,
                "engines": [k for k, _, _ in parsed],
            },
        },
        ensure_ascii=False,
    )


def _merge_majority(
    parsed: List[Tuple[str, str, Dict[str, Any]]],
    header: str,
    mode: str,
) -> str:
    """Vote by clustering similar final_answer texts."""
    finals = [(k, _extract_final(d)) for k, _, d in parsed if _extract_final(d)]
    thought_lines = [f"[{k}/{m}] {_extract_thought(d)}" for k, m, d in parsed]

    vote_count = 1
    if len(finals) <= 1:
        merged = finals[0][1] if finals else _extract_final(parsed[0][2])
        winner_key = finals[0][0] if finals else parsed[0][0]
    else:
        clusters = _cluster_by_similarity(finals)
        winner_key, merged = _pick_largest_cluster(clusters)
        vote_count = max(len(c) for c in clusters)
    vote_info = (
        f"majority vote ({vote_count}/{len(finals)} engines aligned)"
        if len(finals) > 1
        else "single engine"
    )

    return json.dumps(
        {
            "thought": f"{header} — {vote_info}, selected [{winner_key}]:\n"
            + "\n".join(thought_lines),
            "final_answer": merged,
            "metadata": {
                "ensemble_mode": mode,
                "winner": winner_key,
                "engines": [k for k, _, _ in parsed],
            },
        },
        ensure_ascii=False,
    )


def _merge_debate(
    parsed: List[Tuple[str, str, Dict[str, Any]]],
    header: str,
    mode: str,
) -> str:
    """Synthesize a debate transcript and a reconciled verdict."""
    positions = []
    for key, model, data in parsed:
        final = _extract_final(data)
        thought = _extract_thought(data)
        positions.append(
            {
                "engine": key,
                "model": model,
                "position": final or thought,
                "thought": thought,
            }
        )

    if len(positions) == 1:
        p = positions[0]
        return json.dumps(
            {
                "thought": f"{header}:\n[{p['engine']}] {p['thought']}",
                "final_answer": p["position"],
                "metadata": {"ensemble_mode": mode, "engines": [p["engine"]]},
            },
            ensure_ascii=False,
        )

    clusters = _cluster_by_similarity(
        [(p["engine"], p["position"]) for p in positions],
        threshold=0.5,
    )
    majority_key, majority_answer = _pick_largest_cluster(clusters)

    debate_lines = ["## Debate transcript"]
    for p in positions:
        debate_lines.append(f"**{p['engine']}** ({p['model']}): {p['position'][:800]}")

    disagreements = len(clusters) > 1
    if disagreements:
        debate_lines.append("\n## Reconciliation")
        debate_lines.append(
            f"Engines disagreed ({len(clusters)} positions). "
            f"Verdict follows the largest aligned group, led by **{majority_key}**."
        )
        alt = [c[0][0] for c in clusters if c[0][0] != majority_key]
        if alt:
            debate_lines.append(f"Minority/divergent: {', '.join(alt)}.")
    else:
        debate_lines.append("\n## Reconciliation\nAll engines reached aligned conclusions.")

    return json.dumps(
        {
            "thought": f"{header}:\n" + "\n".join(
                f"[{p['engine']}] {p['thought'][:200]}" for p in positions
            ),
            "final_answer": "\n".join(debate_lines) + f"\n\n**Verdict:** {majority_answer}",
            "metadata": {
                "ensemble_mode": mode,
                "engines": [p["engine"] for p in positions],
                "aligned": not disagreements,
                "winner": majority_key,
            },
        },
        ensure_ascii=False,
    )


def _merge_bayesian(
    parsed: List[Tuple[str, str, Dict[str, Any]]],
    header: str,
    mode: str,
) -> str:
    """Weight engines by declared confidence (prior 0.5 if missing)."""
    weighted: List[Tuple[str, str, Dict[str, Any], float]] = []
    for key, model, data in parsed:
        conf = _extract_confidence(data)
        weighted.append((key, model, data, conf))

    total = sum(w for *_, w in weighted) or 1.0
    best = max(weighted, key=lambda x: x[3])
    key, model, data, conf = best

    breakdown = ", ".join(f"{k}={w:.2f}" for k, _, _, w in weighted)
    thought_lines = [f"[{k}/{m}] (p={w:.2f}) {_extract_thought(d)}" for k, m, d, w in weighted]

    return json.dumps(
        {
            "thought": f"{header} — Bayesian weights [{breakdown}], selected [{key}]:\n"
            + "\n".join(thought_lines),
            "final_answer": _extract_final(data) or _extract_thought(data),
            "metadata": {
                "ensemble_mode": mode,
                "winner": key,
                "winner_model": model,
                "confidence": conf,
                "weights": {k: w / total for k, _, _, w in weighted},
                "engines": [k for k, _, _ in parsed],
            },
        },
        ensure_ascii=False,
    )


def _merge_consensus(
    parsed: List[Tuple[str, str, Dict[str, Any]]],
    header: str,
    mode: str,
) -> str:
    """Cluster answers; prefer largest cluster, tie-break by confidence then length."""
    finals = [(k, _extract_final(d), _extract_confidence(d)) for k, _, d in parsed]
    thought_lines = [f"[{k}/{m}] {_extract_thought(d)}" for k, m, d in parsed]

    if len(finals) == 1:
        k, ans, _ = finals[0]
        merged, winner = ans, k
    else:
        clusters: List[List[Tuple[str, str, float]]] = []
        for key, final, conf in finals:
            if not final:
                continue
            placed = False
            for cluster in clusters:
                if _similarity(final, cluster[0][1]) >= 0.55:
                    cluster.append((key, final, conf))
                    placed = True
                    break
            if not placed:
                clusters.append([(key, final, conf)])

        if not clusters:
            k, _, d = parsed[0]
            merged, winner = _extract_final(d), k
        else:
            best_cluster = max(
                clusters,
                key=lambda c: (len(c), sum(x[2] for x in c) / len(c)),
            )
            winner, merged, _ = max(best_cluster, key=lambda x: (x[2], len(x[1])))

    return json.dumps(
        {
            "thought": f"{header} — consensus via [{winner}]:\n" + "\n".join(thought_lines),
            "final_answer": merged,
            "metadata": {
                "ensemble_mode": mode,
                "winner": winner,
                "engines": [k for k, _, _ in parsed],
            },
        },
        ensure_ascii=False,
    )


async def run_ensemble(
    mode: str,
    active: List[Dict[str, str]],
    prompt: str,
    run_engine: Callable,
    *,
    record_run: Optional[Callable[[str, str, float, int], None]] = None,
    **kwargs: Any,
) -> str:
    """
    Execute ensemble strategy.

    Args:
        mode: ensemble mode name
        active: [{"key", "label", "model"}, ...]
        prompt: user prompt
        run_engine: async (engine_key) -> (key, model, text, elapsed, tokens)
        record_run: optional callback(engine_key, model, elapsed, tokens)
    """
    mode = (mode or "consensus").lower().strip()
    if mode not in ALL_ENSEMBLE_MODES:
        mode = "consensus"

    async def _run_one(eng: Dict[str, str]) -> EngineRun:
        key = eng["key"]
        try:
            return await run_engine(key, eng, prompt, **kwargs)
        except Exception as exc:
            err = json.dumps(
                {
                    "thought": f"[{key}] error",
                    "final_answer": f"Error ({key}): {type(exc).__name__}: {str(exc)[:200]}",
                }
            )
            return key, eng.get("model", key), err, 0.0, 0

    if mode == "race":
        tasks = {
            eng["key"]: asyncio.create_task(_run_one(eng), name=f"ensemble-{eng['key']}")
            for eng in active
        }
        done, pending = await asyncio.wait(
            tasks.values(),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        runs: List[EngineRun] = []
        winner_run: Optional[EngineRun] = None
        for task in done:
            try:
                run = task.result()
                runs.append(run)
                if winner_run is None or run[3] < winner_run[3]:
                    winner_run = run
            except Exception:
                continue

        if winner_run and record_run:
            record_run(winner_run[0], winner_run[1], winner_run[3], winner_run[4])

        return merge_ensemble_responses("race", runs if runs else ([winner_run] if winner_run else []))

    results = await asyncio.gather(
        *[_run_one(eng) for eng in active],
        return_exceptions=True,
    )
    runs = []
    for item in results:
        if isinstance(item, Exception):
            continue
        runs.append(item)
        if record_run:
            record_run(item[0], item[1], item[3], item[4])

    return merge_ensemble_responses(mode, runs)
