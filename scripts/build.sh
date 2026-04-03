#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Building claude-haha-bin..."

mkdir -p src/components/agents src/assistant src/commands/assistant src/commands/agents-platform src/services/contextCollapse src/tools/SuggestBackgroundPRTool src/tools/VerifyPlanExecutionTool src/ink

for f in src/utils/protectedNamespace.ts src/tools/REPLTool/REPLTool.ts src/tools/SuggestBackgroundPRTool/SuggestBackgroundPRTool.ts src/tools/VerifyPlanExecutionTool/VerifyPlanExecutionTool.ts src/components/agents/SnapshotUpdateDialog.tsx src/assistant/AssistantSessionChooser.tsx src/commands/assistant/assistant.ts src/commands/agents-platform/index.ts src/services/compact/cachedMicrocompact.ts src/services/compact/snipCompact.ts src/ink/devtools.ts src/services/contextCollapse/index.ts; do
  [ ! -f "$f" ] && touch "$f"
done

echo "Compiling..."
bun build ./src/entrypoints/cli-compiled.ts --compile --outfile ./bin/claude-haha-bin --define "MACRO=globalThis.MACRO"

echo "Done! Binary size:"
ls -lh ./bin/claude-haha-bin