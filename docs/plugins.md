# Plugins: Graphify & Impeccable

Both were chosen from the "Top 30 Claude Code Plugins" list, and both were **audited before installing**. Both are installed **project-scoped**: the config is committed in this repo, so it travels with the repo and doesn't touch your other projects.

| | Graphify | Impeccable |
|---|---|---|
| Repo | [`Graphify-Labs/graphify`](https://github.com/Graphify-Labs/graphify) | [`pbakaus/impeccable`](https://github.com/pbakaus/impeccable) |
| Licence | Apache-2.0 | Apache-2.0 |
| Pinned at | PyPI `graphifyy==0.9.100` (note the double y; that is the official package) | commit `f5c8043`, plugin v4.8.0 |
| What it does | Turns code + docs into a queryable knowledge graph (`/graphify`, `graphify query`) | Design vocabulary + anti-pattern detection for UI (`/impeccable audit`, `polish`, `critique`...) |
| Lab use | Map the codebase now; later, a typology/regulation graph for the Threat Explorer & Knowledge Assistant | Investigator Copilot / dashboard UI, when we build it |

## Audit findings (2026-11-01)

### Graphify
- **Code mapping is local:** tree-sitter AST parsing, no LLM, no network. Verified by building this repo's graph (126 nodes).
- **No telemetry.** The query log is off by default. `app.graphify.com` only appears in printed banners; no upload code was found.
- **Document "semantic" extraction does send text to a model.** Inside Claude Code it uses Claude sub-agents, so the frugal sensitivity gate applies. **But** if `GEMINI_API_KEY`/`GOOGLE_API_KEY` is set, the skill sends documents straight to Google, bypassing the gate. The CLI also auto-picks `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`. Mitigation: `GRAPHIFY_BACKEND=ollama` is set in project settings, and the lab rule is to never set Gemini keys here.
- **What it installed:**
  - `.claude/skills/graphify/`
  - a `## graphify` section in `CLAUDE.md` and `.claude/CLAUDE.md`
  - two PreToolUse hooks (`graphify hook-guard search|read`) that nudge Claude to query the graph before grepping. The hooks give advice only; they don't block.
- **Git hooks are not installed** (that needs a separate `graphify hook install`, which we haven't run).
- **Restricted material is excluded:** `.graphifyignore` excludes the same restricted paths as the sensitivity gate. Verified: a decoy file in `data/real/` did not appear in the graph. `graphify-out/` is git-ignored.

### Impeccable
- **Hooks:** they run on SessionStart, after every Edit/Write, and on Stop. They only scan UI files and return advice. They don't edit your files and make no network calls. Their only writes are a cache/state file and a git-exclude entry.
- **The engine binary** is downloaded once from GitHub Releases, **SHA-256 verified** against a checksum list shipped in the plugin, and refused on mismatch. Verified: it downloaded, verified and scanned a test file.
- **Telemetry:** it is *on by default* for the design "roll" feature. It sends design-option choices and version info to `impeccable.style` (no code or project names, per the source). **Disabled here** with `IMPECCABLE_NO_TELEMETRY=1` and `DO_NOT_TRACK=1`.
- **Optional image generation** calls OpenAI if you set a key. Don't set one in this project.

## Setup on your machine

Step 1, install the Graphify CLI (needed by its hooks and skill):

```bash
uv tool install graphifyy==0.9.100
```

(or `pipx install graphifyy==0.9.100`)

Step 2, open the repo in Claude Code. It reads `.claude/settings.json` and offers to trust the project, add the `impeccable` marketplace, and install the plugin. Accept.

Step 3, build the code graph (local, free):

```bash
graphify update .
```

Then try `/graphify .` and `/impeccable critique` inside Claude Code.

## Upgrading

- **Graphify:** bump the pinned version, read the CHANGELOG, then re-run `graphify install --project` and review the diff to `.claude/settings.json` and `CLAUDE.md` before committing.
- **Impeccable:** change the `ref` commit in `.claude/settings.json` after reading the release notes and checking that the hook commands haven't changed.

## Turning them off

- Impeccable: set `"impeccable@impeccable": false` under `enabledPlugins`.
- Graphify: `graphify uninstall --project`.
- Both, for one run: `claude --settings '{"disableAllHooks": true}'`.
