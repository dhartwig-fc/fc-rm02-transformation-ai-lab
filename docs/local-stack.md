# Local Stack & Sensitivity Gate

Phase 1 of [`tooling-strategy.md`](tooling-strategy.md): a private, local-only AI stack plus a guard that stops sensitive data being handed to Claude sub-agents.

## What you get

| Component | Version (pinned) | Address | Purpose |
|---|---|---|---|
| Ollama | `0.40.13` | `127.0.0.1:11434` | Runs open-weight models on your machine. The "sensitive data" route |
| Open WebUI | `v0.9.4` | `http://127.0.0.1:3000` | Chat + document RAG front-end. Becomes the Knowledge Assistant |
| Sensitivity gate | `.claude/frugal-sensitivity.json` | n/a | Blocks Claude Code sub-agent delegation when PII, case data or secrets appear |

Everything binds to **127.0.0.1**. Nothing is reachable from your network. Telemetry, web search and cloud model providers are off in Open WebUI.

## Quick start

**macOS** (recommended: native Ollama so it can use the Apple GPU):

1. Install Ollama from <https://ollama.com> and start it.
2. Then:

```bash
scripts/lab.sh init
```

```bash
scripts/lab.sh up
```

```bash
scripts/lab.sh models
```

```bash
scripts/lab.sh check
```

**Linux / Windows (WSL2)**: the same four commands. `init` selects docker mode, so Ollama runs in a container.
For an NVIDIA GPU, install the NVIDIA Container Toolkit and add a `deploy.resources.reservations.devices` block to the `ollama` service.

Then open <http://127.0.0.1:3000>. The **first account you create becomes admin**. After that, set `ENABLE_SIGNUP=false` in `.env` and run `scripts/lab.sh up` again.

### Choosing a model (`LAB_CHAT_MODEL` in `.env`)

| RAM | Model |
|---|---|
| 8 GB | `qwen3:4b` |
| 16 GB | `qwen3:8b` (default) |
| 32 GB+ | `qwen3:14b` or `gpt-oss:20b` |

The embedding model `nomic-embed-text` is used for Open WebUI document RAG.

### Commands

| Command | Does |
|---|---|
| `scripts/lab.sh init` | Creates `.env` from `.env.example`, generates `WEBUI_SECRET_KEY`, picks native mode on macOS |
| `scripts/lab.sh up` / `down` | Start / stop (model and chat data persist in Docker volumes) |
| `scripts/lab.sh models` | Pull the chat + embedding models |
| `scripts/lab.sh check` | Smoke test: ports bound to localhost only, Ollama up, Open WebUI healthy, model answers |
| `scripts/lab.sh logs` | Follow logs |

### Upgrading

Change the version in `.env`, read the release notes **and the security advisories**, run `up` then `check`, and record the change in `docs/licences.md`. Never switch to `latest`.

## Sensitivity gate

### How it works

The **frugal** Claude Code plugin's `guard_sensitive.py` hook runs before every sub-agent spawn. If the text being handed over matches a rule in `.claude/frugal-sensitivity.json`, the spawn is **blocked** and Claude must handle the work itself in the main conversation. That work can then be routed to Ollama if needed.

It **fails closed**: a broken config blocks all delegation rather than letting data through.

> The gate only works where the frugal plugin is installed. In any other environment the JSON file is inert. The `.gitignore` rules and the "synthetic data only" policy are the backstop.

### Rules

| Rule | Catches | Deliberately ignores |
|---|---|---|
| `iban` | Any IBAN shape, spaced or not | n/a |
| `uk-sort-code-and-account` | Sort code + 8-digit account within ~30 chars | Sort code alone (looks like a date) |
| `payment-card-pan` | Visa / MC / Amex / Discover number shapes | Transaction IDs, timestamps |
| `uk-national-insurance-number` | Valid-format NINOs | Never-issued prefixes (e.g. `QQ` specimen) |
| `passport-number-in-context` | Passport numbers near the word "passport" | Bare 9-digit numbers |
| `case-and-sar-material` | SAR/DAML/STR **reference numbers**, protective markings, `data/real/`, `data/restricted/`, `cases/`, `*.sar` | Talking *about* SARs, DAMLs or typologies |
| `secrets-and-keys` | API keys, AWS keys, private keys, `.env`, `*.pem`, `secrets/` | `.env.example` |

### Tests

```bash
python3 -m pytest tests/ -q
```

- Every rule has positive cases, and there is a set of everyday financial crime phrases that must **not** be blocked. This guards against false positives.
- If the frugal plugin is installed, the same cases also go through the **real hook**, which must return 2 for block and 0 for allow. A fail-closed test covers broken config.
- Changing a rule means adding a positive and a negative case first.

### Known limits (be honest about these)

- Regex catches **identifiers**, not **meaning**. Names, addresses and narrative descriptions of a real person are not detected. The real control is: **no real data in this lab**.
- ~~The gate covers sub-agent delegation only.~~ **Closed** by the prompt check below: what you type is now checked too. It still cannot see files Claude opens itself, so the `.gitignore` rules and "no real data" policy remain the backstop.
- Synthetic data that *looks* real (e.g. generated IBANs) will also be blocked from delegation. That is intended: it rehearses the production behaviour.

## Prompt check (your own messages)

`.claude/hooks/check_prompt.py` runs on **every message you send**, before Claude sees it. It uses the same rules file as the gate, so one set of rules covers both checks.

| Your message contains | Result |
|---|---|
| An identifier (IBAN, card, NINO, passport, SAR ref, API key...) | **Blocked.** The message is not sent. You see which rule matched, but never the value itself |
| Only a restricted *path* (`data/real/`, `cases/`, `.env`...) | **Sent, with a warning.** Claude is told not to open those files. Editing `.gitignore` still works |
| A first line starting `[synthetic-data]` | **Sent.** You attest the data is fake, and Claude is told so |
| Broken rules file | **Blocked** (fails closed) |

- The `[synthetic-data]` marker only works at the **start** of the message. A tag buried mid-message doesn't bypass the check, and that's tested.
- Use it honestly: it's an attestation, like signing a customs declaration.

Tests: `tests/test_prompt_check.py` reuses the gate's positive and negative examples, so the two checks can't drift apart.

## Verification log

Run in the Claude Code cloud sandbox, 2026-10-31:

- `pytest`: 65 passed (incl. the real-hook integration tests)
- Live check: delegating an IBAN to a sub-agent was **blocked** by the gate
- `ollama/ollama:0.40.13` container healthy; `lab.sh check` confirmed all ports bound to 127.0.0.1 and Ollama responding
- **Not verified in the sandbox:** Open WebUI image pull (`pkg-containers.githubusercontent.com` blocked) and model pulls (`registry.ollama.ai` blocked) by the sandbox network policy. Run `scripts/lab.sh check` on your machine to complete verification.
