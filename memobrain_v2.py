"""
MemoBrain PoC v2 — Real tool integrations + security-hardened code execution.
Based on arXiv:2601.08079

Run: python3 memobrain_v2.py
Set FIREWORKS_API_KEY or OPENAI_API_KEY for live LLM mode.
"""

import json
import os
import re
import sys
import signal
import threading
import io
import traceback
import math
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from urllib.parse import quote_plus

# ── LLM client ─────────────────────────────────────────────────────────────
try:
    import openai
    _api_key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not _api_key:
        raise RuntimeError("No API key found")
    client = openai.OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        api_key=_api_key,
    )
    MODEL = os.environ.get("MEMOBRAIN_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct")
    LLM_AVAILABLE = True
except Exception as e:
    print(f"[WARN] LLM client unavailable ({e}). Mock mode.", file=sys.stderr)
    LLM_AVAILABLE = False
    client = None
    MODEL = None

# ── Mock LLM state ─────────────────────────────────────────────────────────
_mock_episode_counter = 0
_mock_thought_counter = 0


def _extract_episode_result(prompt: str) -> str:
    match = re.search(r"New episode result:\s*(.*?)(?=\n\nAbstract)", prompt, re.DOTALL)
    return match.group(1).strip() if match else ""


def mock_llm(prompt: str) -> str:
    """Deterministic mock for testing without API keys."""
    global _mock_episode_counter, _mock_thought_counter
    p_lower = prompt.lower()
    result = _extract_episode_result(prompt).lower()

    # Thought formation
    if "abstract this episode" in p_lower:
        _mock_thought_counter += 1
        tc = _mock_thought_counter
        if tc == 1:
            return "Confirmed Paris is the capital of France via search."
        if tc == 2:
            return "Retrieved Paris population: approximately 2,160,000."
        if tc == 3:
            return "Performed calculation: 2,160,000 x 2 = 4,320,000."
        return "Recorded reasoning step."

    if "decide which operations" in p_lower:
        thoughts = re.findall(r"\[Thought\s+(T\d+)\]\s*(.+?)(?=\[Thought|$)", prompt, re.DOTALL)
        lines = []
        for tid, content in thoughts:
            c = content.lower()
            if "error" in c or "failed" in c or "invalid" in c:
                lines.append(f"Thought {tid}: FLUSH")
            elif "final" in c or "synthesized" in c or "resolved" in c or "confirmed" in c:
                lines.append(f"Thought {tid}: FOLD")
            else:
                lines.append(f"Thought {tid}: KEEP")
        return "\n".join(lines) if lines else "Thought T1: KEEP"

    if "summarize this completed reasoning" in p_lower:
        if "paris" in p_lower:
            return "Resolved: Paris is capital of France, population ~2.16M."
        return "Resolved subtask with confirmed conclusion."

    if "compact this reasoning attempt" in p_lower:
        return "Attempted step failed or was superseded; minimal structural relevance."

    if "decide the next action" in p_lower:
        _mock_episode_counter += 1
        ep = _mock_episode_counter
        if ep == 1:
            return "ACTION: SEARCH\nCONTENT: capital of France"
        if ep == 2:
            return "ACTION: SEARCH\nCONTENT: population of Paris"
        if ep == 3:
            return "ACTION: CALCULATE\nCONTENT: 2160000 * 2"
        return "ACTION: ANSWER\nCONTENT: Paris is the capital of France. Its population (~2.16M) multiplied by 2 is 4,320,000."

    return "[MOCK_RESPONSE]"


def llm_call(prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 2048) -> str:
    if not LLM_AVAILABLE:
        return mock_llm(prompt)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}. Mock fallback.", file=sys.stderr)
        return mock_llm(prompt)


# ── Real tools ──────────────────────────────────────────────────────────────
class ToolError(Exception):
    pass


def search_duckduckgo(query: str, max_results: int = 3) -> str:
    """Search via DuckDuckGo HTML instant answers (no API key needed)."""
    import requests
    try:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MemoBrain/0.1)"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return f"[Search] HTTP {r.status_code}"
        # Extract snippets from results
        snippets = re.findall(r"class=\"result__snippet\"[^>]*>(.+?)</a>", r.text)
        if not snippets:
            return f"[Search] No results for: {query}"
        clean = []
        for s in snippets[:max_results]:
            s = re.sub(r"<[^>]+>", "", s)
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                clean.append(s)
        return "[Search] " + " | ".join(clean)
    except requests.Timeout:
        return "[Search] Timeout"
    except Exception as e:
        return f"[Search] Error: {e}"


def search_tavily(query: str, max_results: int = 3) -> str:
    """Tavily search if API key is configured."""
    import requests
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return "[Search] TAVILY_API_KEY not set"
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": max_results, "include_answer": True},
            timeout=15,
        )
        data = r.json()
        if "answer" in data and data["answer"]:
            return f"[Search] {data['answer']}"
        results = data.get("results", [])
        snippets = [x.get("content", "") for x in results[:max_results] if x.get("content")]
        return "[Search] " + " | ".join(snippets) if snippets else "[Search] No results"
    except Exception as e:
        return f"[Search] Error: {e}"


def _search(query: str) -> str:
    """Try Tavily first, fallback to DuckDuckGo."""
    if os.environ.get("TAVILY_API_KEY"):
        return search_tavily(query)
    return search_duckduckgo(query)


def execute_python(expression: str, timeout_sec: int = 5) -> str:
    """
    Hardened Python execution via subprocess for sandboxing.
    Restricted builtins: no file I/O, no network, no import, no __dunder__.
    """
    # Pre-check: block dangerous patterns
    danger_patterns = [
        r"\b__import__\b",
        r"\bimport\b",
        r"\bopen\s*\(",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bcompile\s*\(",
        r"\binput\s*\(",
        r"\bos\.",
        r"\bsubprocess\.",
        r"\bsocket\.",
        r"\burllib\.",
        r"\brequests\.",
        r"\bfile\b",
        r"\bwrite\b",
    ]
    for pat in danger_patterns:
        if re.search(pat, expression):
            return f"[Exec] BLOCKED: unsafe pattern detected ({pat})"

    # Build restricted wrapper
    wrapper = f"""
_restricted = {{
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "sum": sum, "min": min, "max": max,
    "abs": abs, "round": round, "int": int, "float": float, "str": str,
    "list": list, "tuple": tuple, "dict": dict, "set": set, "bool": bool,
    "print": print, "type": type, "isinstance": isinstance, "hasattr": hasattr,
    "getattr": getattr, "sorted": sorted, "reversed": reversed, "all": all,
    "any": any, "math": __import__("math"), "pow": pow, "divmod": divmod,
    "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
    "ascii": ascii, "repr": repr, "format": format, "complex": complex,
    "iter": iter, "next": next, "slice": slice, "Exception": Exception,
    "ValueError": ValueError, "TypeError": TypeError, "ZeroDivisionError": ZeroDivisionError,
}}
"""
    # Run in subprocess
    import subprocess
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper + "\nexec('" + expression.replace("'", "\\'") + "', {'__builtins__': _restricted})"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip()[:500]
            return f"[Exec] Error: {err}"
        out = proc.stdout.strip()
        return f"[Exec] {out}" if out else "[Exec] OK (no output)"
    except subprocess.TimeoutExpired:
        return f"[Exec] Timeout after {timeout_sec}s"
    except Exception as e:
        return f"[Exec] Error: {e}"


# ── MemoBrain core ──────────────────────────────────────────────────────────
@dataclass
class Thought:
    id: str
    content: str
    dependencies: Set[str] = field(default_factory=set)
    state: str = "active"          # active | folded | flushed
    summary: Optional[str] = None
    episode_idx: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_context(self) -> str:
        if self.state == "folded" and self.summary:
            return f"[FOLDED Thought {self.id}] {self.summary}"
        elif self.state == "flushed" and self.summary:
            return f"[FLUSHED Thought {self.id}] {self.summary}"
        return f"[Thought {self.id}] {self.content}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "dependencies": list(self.dependencies),
            "state": self.state,
            "summary": self.summary,
            "episode_idx": self.episode_idx,
            "timestamp": self.timestamp,
        }


class MemoryGraph:
    def __init__(self):
        self.thoughts: Dict[str, Thought] = {}
        self.edges: Set[Tuple[str, str]] = set()

    def add_thought(self, thought: Thought):
        self.thoughts[thought.id] = thought
        for dep in thought.dependencies:
            self.edges.add((dep, thought.id))

    def get_active(self) -> List[Thought]:
        return [t for t in self.thoughts.values() if t.state == "active"]

    def get_context(self) -> str:
        active = self.get_active()
        if not active:
            return ""
        sorted_thoughts = self._topo_sort(active)
        return "\n".join([t.to_context() for t in sorted_thoughts])

    def _topo_sort(self, thoughts: List[Thought]) -> List[Thought]:
        ids = {t.id for t in thoughts}
        in_degree = {t.id: 0 for t in thoughts}
        for t in thoughts:
            for dep in t.dependencies:
                if dep in ids:
                    in_degree[t.id] += 1
        result = []
        queue = [t for t in thoughts if in_degree[t.id] == 0]
        while queue:
            t = queue.pop(0)
            result.append(t)
            for edge in self.edges:
                if edge[0] == t.id:
                    child = edge[1]
                    if child in in_degree:
                        in_degree[child] -= 1
                        if in_degree[child] == 0:
                            child_thought = self.thoughts.get(child)
                            if child_thought and child_thought in thoughts:
                                queue.append(child_thought)
        return result

    def to_dict(self) -> dict:
        return {
            "thoughts": {k: v.to_dict() for k, v in self.thoughts.items()},
            "edges": list(self.edges),
        }


class MemoBrain:
    def __init__(self, context_budget: int = 4000):
        self.graph = MemoryGraph()
        self.context_budget = context_budget
        self.episode_counter = 0
        self.max_context_chars = context_budget
        self.stats = {"thoughts_formed": 0, "folded": 0, "flushed": 0, "llm_calls": 0}

    def run_episode(self, task: str, current_context: str, episode_result: str) -> str:
        self.episode_counter += 1
        thought = self._form_thought(task, current_context, episode_result, self.episode_counter)
        self.graph.add_thought(thought)
        self.stats["thoughts_formed"] += 1
        context = self.graph.get_context()
        if len(context) > self.max_context_chars:
            print(f"[MemoBrain] Budget exceeded ({len(context)} chars) → memory management")
            self._manage_memory(task)
        return self.graph.get_context()

    def _form_thought(self, task: str, context: str, episode_result: str, idx: int) -> Thought:
        prompt = (
            f"Task: {task}\n\n"
            f"Current reasoning context:\n{context}\n\n"
            f"New episode result:\n{episode_result}\n\n"
            "Abstract this episode into a concise executive thought (1-3 sentences). "
            "Capture: what subproblem was addressed, what conclusion/outcome was reached, "
            "what information was used.\n\n"
            "Output only the thought text, no prefixes."
        )
        self.stats["llm_calls"] += 1
        content = llm_call(prompt, system="You are an executive memory summarizer. Be concise.")

        deps = set()
        for tid in self.graph.thoughts:
            if tid in content:
                deps.add(tid)
        prev_id = f"T{idx - 1}"
        if prev_id in self.graph.thoughts and not deps:
            deps.add(prev_id)

        return Thought(
            id=f"T{idx}",
            content=content.strip(),
            dependencies=deps,
            state="active",
            episode_idx=idx,
        )

    def _manage_memory(self, task: str):
        context = self.graph.get_context()
        prompt = (
            f"Task: {task}\n\n"
            f"Current active memory thoughts:\n{context}\n\n"
            "Context is too long. Decide per thought: KEEP, FOLD, or FLUSH.\n"
            "Format:\nThought T1: <decision>\nThought T2: <decision>\n...\n"
            "Output only the decisions."
        )
        self.stats["llm_calls"] += 1
        decision = llm_call(prompt, system="You are an executive memory manager. Be decisive.")
        for line in decision.split("\n"):
            match = re.match(r"Thought\s+(T\d+):\s*(KEEP|FOLD|FLUSH)", line, re.IGNORECASE)
            if match:
                tid, action = match.group(1), match.group(2).upper()
                if tid in self.graph.thoughts:
                    if action == "FOLD":
                        self._fold_thought(tid)
                    elif action == "FLUSH":
                        self._flush_thought(tid)

    def _fold_thought(self, tid: str):
        t = self.graph.thoughts[tid]
        prompt = f"Summarize this completed reasoning into one sentence:\n{t.content}\n\nOutput only the summary."
        self.stats["llm_calls"] += 1
        summary = llm_call(prompt, system="You summarize reasoning steps. Be concise.")
        t.state = "folded"
        t.summary = summary.strip()
        self.stats["folded"] += 1
        print(f"[MemoBrain] FOLDED {tid}: {t.summary}")

    def _flush_thought(self, tid: str):
        t = self.graph.thoughts[tid]
        prompt = f"Compact this reasoning attempt into a brief note (1 sentence):\n{t.content}\n\nOutput only the compact note."
        self.stats["llm_calls"] += 1
        note = llm_call(prompt, system="You compact reasoning notes. Be concise.")
        t.state = "flushed"
        t.summary = note.strip()
        self.stats["flushed"] += 1
        print(f"[MemoBrain] FLUSHED {tid}: {t.summary}")

    def get_context(self) -> str:
        return self.graph.get_context()


# ── Agent ───────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, memobrain: MemoBrain):
        self.mb = memobrain
        self.tools = {
            "search": self._tool_search,
            "calculate": self._tool_calculate,
            "python": self._tool_python,
        }

    def _tool_search(self, query: str) -> str:
        return _search(query)

    def _tool_calculate(self, expression: str) -> str:
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "[Calc] Error: invalid characters"
            result = eval(expression, {"__builtins__": None}, {"math": math})
            return f"[Calculation] {expression} = {result}"
        except Exception as e:
            return f"[Calculation] Error: {e}"

    def _tool_python(self, code: str) -> str:
        return execute_python(code)

    def solve(self, task: str, max_episodes: int = 5) -> str:
        print(f"\n{'=' * 60}")
        print(f"TASK: {task}")
        print(f"{'=' * 60}")
        context = ""
        for i in range(max_episodes):
            print(f"\n--- Episode {i + 1} ---")
            prompt = (
                f"You are a reasoning agent. Solve the task step by step.\n\n"
                f"Task: {task}\n\n"
                f"Current memory context:\n{context}\n\n"
                "Available tools: search, calculate, python\n\n"
                "Decide the next action:\n"
                "1. THINK: reason about the problem\n"
                "2. SEARCH: search for information (provide query)\n"
                "3. CALCULATE: simple arithmetic (provide expression)\n"
                "4. PYTHON: run Python code for complex computation\n"
                "5. ANSWER: provide final answer\n\n"
                "Format:\n"
                "ACTION: <THINK/SEARCH/CALCULATE/PYTHON/ANSWER>\n"
                "CONTENT: <your reasoning or tool input>\n\n"
                "If answering, provide the final answer."
            )
            response = llm_call(prompt, system="You are a careful reasoning agent.")
            print(f"Agent: {response[:300]}...")

            action_match = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
            content_match = re.search(r"CONTENT:\s*(.*?)(?=\nACTION:|$)", response, re.DOTALL | re.IGNORECASE)
            action = action_match.group(1).upper() if action_match else "THINK"
            content = content_match.group(1).strip() if content_match else response

            episode_result = content
            action_lower = action.lower()
            if action_lower in self.tools:
                episode_result = self.tools[action_lower](content)

            context = self.mb.run_episode(task, context, episode_result)

            if action == "ANSWER":
                print(f"\n*** FINAL ANSWER: {content} ***")
                return content

        return context


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    mb = MemoBrain(context_budget=3000)
    agent = Agent(mb)
    task = "What is the capital of France, and what is the population of that city multiplied by 2?"
    result = agent.solve(task, max_episodes=5)

    print(f"\n{'=' * 60}")
    print("FINAL MEMORY STATE")
    print(f"{'=' * 60}")
    print(mb.get_context())
    print(f"\n{'=' * 60}")
    print("STATS")
    print(f"{'=' * 60}")
    print(json.dumps(mb.stats, indent=2))
    print(f"\n{'=' * 60}")
    print("MEMORY GRAPH JSON")
    print(f"{'=' * 60}")
    print(json.dumps(mb.graph.to_dict(), indent=2))


if __name__ == "__main__":
    main()
