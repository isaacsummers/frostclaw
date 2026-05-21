#!/usr/bin/env bash
set -euo pipefail

# Resolve the repo root relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_SRC="$SCRIPT_DIR/bin/frostclaw"

# XDG_BIN_HOME defaults to ~/.local/bin per XDG spec
TARGET_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
TARGET="$TARGET_DIR/frostclaw"

if [ ! -f "$BIN_SRC" ]; then
  echo "Error: source binary not found at $BIN_SRC" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
ln -sf "$BIN_SRC" "$TARGET"

echo "✅ Linked: $TARGET -> $BIN_SRC"
echo ""
echo "Ensure $TARGET_DIR is on your PATH. Add to ~/.bashrc or ~/.zshrc if needed:"
echo "  export PATH=\"\$PATH:$TARGET_DIR\""
