## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.

## Lab rules for plugins (see docs/plugins.md)

- Graphify code mapping (`graphify update .`) is local and AST-only - always fine.
- `/graphify` semantic extraction over documents sends their text to a model: run it only on **public** material (FATF, regulator guidance, typology reports), never on anything under `data/real/`, `data/restricted/` or `cases/`.
- Do **not** set `GEMINI_API_KEY` / `GOOGLE_API_KEY` in this project: graphify would send documents straight to Google, bypassing the sensitivity gate. `GRAPHIFY_BACKEND=ollama` is set in `.claude/settings.json` so CLI semantic passes stay on the local stack.
- Impeccable is for UI work only. Telemetry is disabled via `IMPECCABLE_NO_TELEMETRY`; do not re-enable it.
- Diagram Design: always ask for **offline / system-font output** (no Google Fonts link), so diagrams of operating models or bank architecture make no outside requests when opened. Diagrams are for public or synthetic content only.
