"""
MemoBrain Proof of Concept
Executive Memory as an Agentic Brain for Reasoning
Based on arXiv:2601.08079

Run: python3 memobrain.py
Set FIREWORKS_API_KEY or OPENAI_API_KEY for live LLM mode.
Without API keys, runs in deterministic mock mode.
"""

import json
import os
import re
import sys
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field

# Try to load LLM client
try:
    import openai
    client = openai.OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        api_key=os.environ.get("FIREWORKS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )
    MODEL = os.environ.get("MEMOBRAIN_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct")
    LLM_AVAILABLE = True
except Exception as e:
    print(f"[WARN] LLM client not available ({e}). Running in mock mode.", file=sys.stderr)
    LLM_AVAILABLE = False
    client = None
    MODEL = None


def llm_call(prompt: str, system: str = "", temperature: float = 0.2) -> str:
    if not LLM_AVAILABLE:
        return mock_llm(prompt)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=temperature, max_tokens=2048
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}. Falling back to mock.", file=sys.stderr)
        return mock_llm(prompt)


# Mock state tracking
_mock_episode_counter = 0


def _extract_episode_result(prompt: str) -> str:
    """Extract the episode result from thought formation prompt."""
    match = re.search(r'New episode result:\s*(.*?)(?=\n\nAbstract)', prompt, re.DOTALL)
    return match.group(1).strip() if match else ""


def mock_llm(prompt: str) -> str:
    """Deterministic mock LLM for testing without API keys."""
    global _mock_episode_counter
    p_lower = prompt.lower()
    
    # Thought formation
    if "abstract this episode" in p_lower:
        result = _extract_episode_result(prompt).lower()
        if "answer" in result or "4320000" in result or "paris" in result and "capital" in result and "population" in result:
            return "Final answer: Paris (capital), population x2 = 4,320,000."
        if "capital" in result and "france" in result:
            return "Confirmed Paris is the capital of France via search."
        if "population" in result:
            return "Retrieved Paris population: approximately 2,160,000."
        if "calculation" in result or "2160000" in result:
            return "Performed calculation: 2,160,000 x 2 = 4,320,000."
        return "Recorded reasoning step."
    
    # Memory management
    if "decide which operations" in p_lower:
        thoughts = re.findall(r'\[Thought\s+(T\d+)\]\s*(.+?)(?=\[Thought|$)', prompt, re.DOTALL)
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
    
    # Folding summary
    if "summarize this completed reasoning" in p_lower:
        if "paris" in p_lower:
            return "Resolved: Paris is capital of France, population ~2.16M."
        return "Resolved subtask with confirmed conclusion."
    
    # Flushing note
    if "compact this reasoning attempt" in p_lower:
        return "Attempted step failed or was superseded; minimal structural relevance."
    
    # Agent reasoning
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


@dataclass
class Thought:
    id: str
    content: str
    dependencies: Set[str] = field(default_factory=set)
    state: str = "active"
    summary: Optional[str] = None
    episode_idx: int = 0
    
    def to_context(self) -> str:
        if self.state == "folded" and self.summary:
            return f"[FOLDED Thought {self.id}] {self.summary}"
        elif self.state == "flushed" and self.summary:
            return f"[FLUSHED Thought {self.id}] {self.summary}"
        return f"[Thought {self.id}] {self.content}"


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
            "thoughts": {k: {
                "id": v.id, "content": v.content, "dependencies": list(v.dependencies),
                "state": v.state, "summary": v.summary, "episode_idx": v.episode_idx
            } for k, v in self.thoughts.items()},
            "edges": list(self.edges)
        }


class MemoBrain:
    def __init__(self, context_budget: int = 4000):
        self.graph = MemoryGraph()
        self.context_budget = context_budget
        self.episode_counter = 0
        self.max_context_chars = context_budget
    
    def run_episode(self, task: str, current_context: str, episode_result: str) -> str:
        self.episode_counter += 1
        thought = self._form_thought(task, current_context, episode_result, self.episode_counter)
        self.graph.add_thought(thought)
        context = self.graph.get_context()
        if len(context) > self.max_context_chars:
            print(f"[MemoBrain] Context budget exceeded ({len(context)} chars). Running memory management...")
            self._manage_memory(task)
        return self.graph.get_context()
    
    def _form_thought(self, task: str, context: str, episode_result: str, idx: int) -> Thought:
        prompt = f"""Task: {task}

Current reasoning context:
{context}

New episode result:
{episode_result}

Abstract this episode into a concise executive thought (1-3 sentences). Capture:
- What subproblem was addressed
- What conclusion/outcome was reached
- What information was used

Output only the thought text, no prefixes."""
        content = llm_call(prompt, system="You are an executive memory summarizer. Be concise.")
        
        deps = set()
        for tid in self.graph.thoughts:
            if tid in content:
                deps.add(tid)
        prev_id = f"T{idx-1}"
        if prev_id in self.graph.thoughts and not deps:
            deps.add(prev_id)
        
        thought_id = f"T{idx}"
        return Thought(
            id=thought_id, content=content.strip(), dependencies=deps,
            state="active", episode_idx=idx
        )
    
    def _manage_memory(self, task: str):
        context = self.graph.get_context()
        prompt = f"""Task: {task}

Current active memory thoughts:
{context}

Context is too long. Decide which operations to apply:
- FOLD: collapse a completed sub-trajectory into a summary
- FLUSH: compact an invalid/expired thought into a compact note

For each thought, output ONE of: KEEP, FOLD, FLUSH
Format:
Thought T1: <decision>
Thought T2: <decision>
...

Output only the decisions."""
        decision = llm_call(prompt, system="You are an executive memory manager. Be decisive.")
        for line in decision.split('\n'):
            match = re.match(r'Thought\s+(T\d+):\s*(KEEP|FOLD|FLUSH)', line, re.IGNORECASE)
            if match:
                tid, action = match.group(1), match.group(2).upper()
                if tid in self.graph.thoughts:
                    if action == "FOLD":
                        self._fold_thought(tid)
                    elif action == "FLUSH":
                        self._flush_thought(tid)
    
    def _fold_thought(self, tid: str):
        t = self.graph.thoughts[tid]
        prompt = f"""Summarize this completed reasoning into one sentence:
{t.content}

Output only the summary."""
        summary = llm_call(prompt, system="You summarize reasoning steps. Be concise.")
        t.state = "folded"
        t.summary = summary.strip()
        print(f"[MemoBrain] FOLDED {tid}: {t.summary}")
    
    def _flush_thought(self, tid: str):
        t = self.graph.thoughts[tid]
        prompt = f"""Compact this reasoning attempt into a brief note (1 sentence):
{t.content}

Output only the compact note."""
        note = llm_call(prompt, system="You compact reasoning notes. Be concise.")
        t.state = "flushed"
        t.summary = note.strip()
        print(f"[MemoBrain] FLUSHED {tid}: {t.summary}")
    
    def get_context(self) -> str:
        return self.graph.get_context()


class SimpleAgent:
    def __init__(self, memobrain: MemoBrain):
        self.mb = memobrain
        self.tools = {
            "search": self._tool_search,
            "calculate": self._tool_calculate,
        }
    
    def _tool_search(self, query: str) -> str:
        q = query.lower()
        if "capital" in q and "france" in q:
            return "[Search] Paris is the capital of France."
        if "population" in q and "paris" in q:
            return "[Search] Population of Paris: approximately 2,160,000 (2024)."
        return f"[Search] Results for: {query}"
    
    def _tool_calculate(self, expression: str) -> str:
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "[Calc] Error: invalid characters"
            result = eval(expression)
            return f"[Calculation] {expression} = {result}"
        except Exception as e:
            return f"[Calculation] Error: {e}"
    
    def solve(self, task: str, max_episodes: int = 5) -> str:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        context = ""
        for i in range(max_episodes):
            print(f"\n--- Episode {i+1} ---")
            prompt = f"""You are a reasoning agent. Solve the task step by step.

Task: {task}

Current memory context:
{context}

Available tools: search, calculate

Decide the next action:
1. THINK: reason about the problem
2. SEARCH: search for information (provide query)
3. CALCULATE: perform calculation (provide expression)
4. ANSWER: provide final answer

Format:
ACTION: <THINK/SEARCH/CALCULATE/ANSWER>
CONTENT: <your reasoning or tool input>

If answering, provide the final answer."""
            response = llm_call(prompt, system="You are a careful reasoning agent.")
            print(f"Agent: {response[:300]}...")
            
            action_match = re.search(r'ACTION:\s*(\w+)', response, re.IGNORECASE)
            content_match = re.search(r'CONTENT:\s*(.*?)(?=\nACTION:|$)', response, re.DOTALL | re.IGNORECASE)
            action = action_match.group(1).upper() if action_match else "THINK"
            content = content_match.group(1).strip() if content_match else response
            
            episode_result = content
            if action == "SEARCH" and "search" in self.tools:
                episode_result = self.tools["search"](content)
            elif action == "CALCULATE" and "calculate" in self.tools:
                episode_result = self.tools["calculate"](content)
            
            context = self.mb.run_episode(task, context, episode_result)
            
            if action == "ANSWER":
                print(f"\n*** FINAL ANSWER: {content} ***")
                return content
        
        return context


def main():
    mb = MemoBrain(context_budget=3000)
    agent = SimpleAgent(mb)
    task = "What is the capital of France, and what is the population of that city multiplied by 2?"
    result = agent.solve(task, max_episodes=5)
    
    print(f"\n{'='*60}")
    print("FINAL MEMORY STATE:")
    print(f"{'='*60}")
    print(mb.get_context())
    print(f"\n{'='*60}")
    print("MEMORY GRAPH JSON:")
    print(f"{'='*60}")
    print(json.dumps(mb.graph.to_dict(), indent=2))


if __name__ == "__main__":
    main()
