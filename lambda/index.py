"""
MemoBrain Lambda handler
Deploys the MemoBrain agent as an AWS Lambda function.
"""
import json
import os
import sys
import traceback

# Add the package root to the path
sys.path.insert(0, os.path.dirname(__file__))

from memobrain import MemoBrain, Agent

# Environment-based config
LLM_API_KEY = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("OPENAI_API_KEY")
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1")
MODEL = os.environ.get("MEMOBRAIN_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct")
CONTEXT_BUDGET = int(os.environ.get("MEMOBRAIN_CONTEXT_BUDGET", "3000"))
MAX_EPISODES = int(os.environ.get("MEMOBRAIN_MAX_EPISODES", "10"))

# Initialize once per Lambda container
_mb = None
_agent = None


def _get_agent():
    global _mb, _agent
    if _agent is None:
        _mb = MemoBrain(context_budget=CONTEXT_BUDGET)
        _agent = Agent(_mb)
    return _agent


def handler(event, context):
    """AWS Lambda entry point."""
    try:
        # Parse request
        http_method = event.get("httpMethod", "POST")
        path = event.get("path", "/")

        # Health check
        if http_method == "GET" and path == "/health":
            return _response(200, {"status": "ok", "model": MODEL})

        # Only POST /solve is supported
        if http_method != "POST" or path not in ("/solve", ""):
            return _response(405, {"error": "Method not allowed. Use POST /solve"})

        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)

        task = body.get("task", "")
        if not task:
            return _response(400, {"error": "Missing 'task' in request body"})

        max_episodes = body.get("max_episodes", MAX_EPISODES)
        context_budget = body.get("context_budget", CONTEXT_BUDGET)

        # Re-initialize if context budget changed
        agent = _get_agent()
        if context_budget != agent.mb.context_budget:
            agent.mb = MemoBrain(context_budget=context_budget)
            agent = Agent(agent.mb)

        result = agent.solve(task, max_episodes=max_episodes)
        memory_state = agent.mb.get_context()
        stats = agent.mb.stats

        return _response(200, {
            "task": task,
            "answer": result,
            "memory_state": memory_state,
            "stats": stats,
        })

    except Exception as e:
        traceback.print_exc()
        return _response(500, {"error": str(e), "traceback": traceback.format_exc()})


def _response(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, default=str),
    }
