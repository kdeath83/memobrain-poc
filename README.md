# MemoBrain PoC

> **Executive Memory as an Agentic Brain for Reasoning**
>
> A proof-of-concept implementation of the MemoBrain architecture from the research paper by **Hongjin Qian** (Beijing Academy of Artificial Intelligence) and **Zhao Cao** (Gaoling School of Artificial Intelligence, Renmin University of China).

## Origins

This project was inspired by a [LinkedIn post by Joydeep Banerjee](https://www.linkedin.com/posts/joydeepbanerjee_ai-agenticai-share-7469252879984439296-ymIy/) that highlighted the MemoBrain paper as a significant advance in agentic AI reasoning.

The original research paper is available on arXiv:
- **Title:** MemoBrain: Executive Memory as an Agentic Brain for Reasoning
- **Authors:** Hongjin Qian, Zhao Cao
- **arXiv:** [2601.08079](https://arxiv.org/pdf/2601.08079)
- **Institutions:** Beijing Academy of Artificial Intelligence; Gaoling School of Artificial Intelligence, Renmin University of China

## What is MemoBrain?

MemoBrain is an executive memory model for tool-augmented agents that constructs a **dependency-aware memory graph** over reasoning steps. Unlike standard prompting that floods the context window with raw tool outputs, MemoBrain:

1. **Forms executive thoughts** — abstracts each reasoning episode into a compact, salient summary
2. **Tracks dependencies** — builds a directed graph of logical relationships between thoughts
3. **Manages memory actively** — folds completed sub-trajectories and flushes obsolete thoughts to stay within a bounded context budget
4. **Reorganizes context** — presents the agent with a compact, topologically-ordered reasoning backbone instead of a bloated transcript

## Live Demo

An interactive HTML demo is included in the repo. Open it locally or serve it via GitHub Pages:

```bash
# Clone and open locally
git clone https://github.com/kdeath83/memobrain-poc.git
cd memobrain-poc/demo
open index.html        # macOS
# or python3 -m http.server 8080 && open http://localhost:8080
```

**Features:**
- Enter any reasoning task
- Watch episodes unfold in real-time
- View the memory graph and statistics
- Connect to your deployed AWS API endpoint for live reasoning
- Works offline in mock mode (deterministic 4-step demo)

## Architecture

```
Task Input
    ↓
Agent (LLM) → ACTION: search / calculate / python / answer
    ↓
Tool Execution → Episode Result
    ↓
MemoBrain.run_episode() → Thought Formation → Memory Graph (Gt)
    ↓
If context budget exceeded:
    Memory Manager (LLM) → FOLD / FLUSH decisions
    ↓
Context Reorganization → Compact context Ct+1
    ↓
(Repeat until ANSWER)
```

## Project Structure

```
memobrain-poc/
├── memobrain.py          # Core implementation (v1 — original PoC)
├── memobrain_v2.py       # Enhanced version with real tool integrations
├── demo/
│   └── index.html          # Interactive HTML demo page
├── lambda/
│   └── index.py            # AWS Lambda handler for serverless deployment
├── cdk/
│   ├── package.json
│   ├── tsconfig.json
│   ├── cdk.json
│   └── lib/
│       ├── bin/memobrain.ts  # CDK entry point
│       └── stack.ts         # Infrastructure stack (Lambda + API Gateway)
├── README.md
├── REVIEW.md              # Security, performance, and logic review
├── deploy.sh              # One-click deploy script
└── .gitignore
```

## Quick Start (Local)

### Prerequisites
- Python 3.9+
- `openai` Python package (`pip install openai`)
- A Fireworks AI or OpenAI API key

### Run without API keys (mock mode)
```bash
cd memobrain-poc
python3 memobrain_v2.py
```

This runs a deterministic mock LLM that simulates a 4-episode reasoning task:
1. Search: "capital of France"
2. Search: "population of Paris"
3. Calculate: 2160000 × 2
4. Answer: Paris, 4,320,000

### Run with live LLM
```bash
export FIREWORKS_API_KEY="your-key-here"
# or export OPENAI_API_KEY="your-key-here"
cd memobrain-poc
python3 memobrain_v2.py
```

### Custom task
```python
from memobrain_v2 import MemoBrain, Agent

mb = MemoBrain(context_budget=3000)
agent = Agent(mb)
result = agent.solve(
    "What is the square root of the population of Tokyo?",
    max_episodes=10
)
print(result)
```

## One-Click Deploy to AWS

### Prerequisites
- [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html) installed (`npm install -g aws-cdk`)
- AWS credentials configured (`aws configure`)
- Node.js 18+

### Deploy
```bash
cd memobrain-poc/cdk
npm install
export FIREWORKS_API_KEY="your-key-here"
cdk bootstrap   # One-time per account/region
cdk deploy      # One-click deploy
```

This creates:
- **AWS Lambda** (Python 3.11, ARM64) running the MemoBrain agent
- **API Gateway** REST API with `/solve` and `/health` endpoints
- **CloudWatch Logs** with 1-week retention
- **CORS-enabled** for web frontend integration

### API Usage
```bash
# Health check
curl https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/health

# Solve a task
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/solve \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the capital of France, and what is its population multiplied by 2?", "max_episodes": 10}'
```

### Clean up
```bash
cdk destroy
```

## Security Considerations

See [REVIEW.md](REVIEW.md) for a full security audit. Key findings:

- **Python execution** is sandboxed via subprocess with restricted `__builtins__` and regex pre-filtering. Known limitation: introspection escapes (`type(()).__bases__[0].__subclasses__()`) are theoretically possible.
- **For production:** Use a container sandbox (Docker/gVisor) or AST-based allowlist instead of regex.
- **API keys** are passed as environment variables. For production, use AWS Secrets Manager.
- **No input validation** on task length — add a limit to prevent API cost blowout.

## Performance Notes

- **LLM call amplification:** Each episode triggers 1 thought-formation LLM call. Memory management adds 1 more. FOLD/FLUSH each add 1 more. A 10-episode task with 2 folds = ~13 LLM calls.
- **Cost estimate:** At $0.10/1K tokens (Fireworks), a 10-episode task costs ~$1–3.
- **Search latency:** DuckDuckGo HTML scraping is ~1–3s and can fail on CAPTCHA. Tavily API is faster and more reliable.
- **No caching:** Repeated searches are re-executed. Add a query cache for production.


## Paper Citation

```bibtex
@article{qian2026memobrain,
  title={MemoBrain: Executive Memory as an Agentic Brain for Reasoning},
  author={Qian, Hongjin and Cao, Zhao},
  journal={arXiv preprint arXiv:2601.08079},
  year={2026},
  institution={Beijing Academy of Artificial Intelligence; Renmin University of China}
}
```

## License

MIT. This is an independent proof-of-concept implementation and is not affiliated with the original authors.

## Acknowledgments

- **Original research:** Hongjin Qian and Zhao Cao for the MemoBrain architecture
- **Discovery:** Joydeep Banerjee for highlighting this paper on LinkedIn
- **Paper:** [arXiv:2601.08079](https://arxiv.org/pdf/2601.08079)
