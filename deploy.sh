#!/bin/bash
# One-click deploy script for MemoBrain PoC to AWS
# Usage: ./deploy.sh [AWS_PROFILE]

set -e

AWS_PROFILE="${1:-default}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  MemoBrain PoC — One-Click AWS Deploy"
echo "============================================================"
echo ""

# Check prerequisites
echo "[1/5] Checking prerequisites..."

if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is required. Install from https://nodejs.org/"
    exit 1
fi

if ! command -v cdk &> /dev/null; then
    echo "ERROR: AWS CDK is required. Install with: npm install -g aws-cdk"
    exit 1
fi

if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI is required. Install from https://aws.amazon.com/cli/"
    exit 1
fi

# Check API key
if [ -z "$FIREWORKS_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: No FIREWORKS_API_KEY or OPENAI_API_KEY set."
    echo "The Lambda will run in mock mode (no live LLM)."
    echo "Set one before deploying for live reasoning:"
    echo "  export FIREWORKS_API_KEY='your-key-here'"
    echo ""
fi

# Install CDK dependencies
echo "[2/5] Installing CDK dependencies..."
cd "$SCRIPT_DIR/cdk"
npm install

# Build TypeScript
echo "[3/5] Building CDK stack..."
npm run build

# Bootstrap (one-time)
echo "[4/5] Bootstrapping CDK (one-time per account/region)..."
cdk bootstrap --profile "$AWS_PROFILE"

# Deploy
echo "[5/5] Deploying MemoBrain stack..."
cdk deploy --require-approval never --profile "$AWS_PROFILE"

echo ""
echo "============================================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "API endpoints (check CDK outputs above for exact URLs):"
echo "  POST /solve  — Submit a reasoning task"
echo "  GET  /health — Health check"
echo ""
echo "Example:"
echo "  curl -X POST \$(YOUR_API_URL)/solve \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"task\": \"What is the capital of France?\"}'"
echo ""
echo "To destroy: cdk destroy --profile $AWS_PROFILE"
echo "============================================================"
