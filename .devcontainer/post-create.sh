set -euo pipefail

cd "$(dirname "$0")/.."
uv sync --all-extras

echo "==> Claude Code CLI (native installer)"
if command -v claude >/dev/null 2>&1; then
  echo "claude は導入済み: $(claude --version)"
else
  curl -fsSL https://claude.ai/install.sh | bash
  echo "導入完了: $("$HOME/.local/bin/claude" --version)"
fi