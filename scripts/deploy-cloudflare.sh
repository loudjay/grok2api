#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

npm ci
npm run typecheck
CI=1 npx wrangler d1 migrations apply DB --remote
npx wrangler deploy
