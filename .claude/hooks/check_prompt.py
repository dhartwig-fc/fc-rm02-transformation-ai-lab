#!/usr/bin/env python3
"""UserPromptSubmit hook: check what YOU type before Claude sees it.

Closes limit #2 in docs/local-stack.md: the frugal sensitivity gate only checks
text Claude hands to sub-agents, not text pasted into the main conversation.
This hook applies the SAME rules file (.claude/frugal-sensitivity.json) to every
prompt, so one rule set governs both doors.

Behaviour
  - Content pattern match (IBAN, card, NINO, SAR ref, key...)  -> BLOCK (exit 2).
    The prompt is discarded and never reaches the model. The matched value is
    never echoed back - only the rule name.
  - Restricted path mention only (data/real/, cases/, .env...)  -> ALLOW + WARN.
    Talking about a path is not a leak (you may be editing .gitignore), but
    Claude is told not to open those files.
  - Prompt starts with "[synthetic-data]"                       -> ALLOW, with
    an attestation note added to Claude's context. Use it only for fake data.
  - No config file                                               -> allow (off).
  - Config exists but is broken                                  -> BLOCK (fail
    closed, same as the frugal gate).

Tested by tests/test_prompt_check.py.
"""
import fnmatch
import json
import os
import re
import sys

SYNTHETIC_MARKER = "[synthetic-data]"


def config_path(payload):
    env = os.environ.get("FRUGAL_SENSITIVITY_CONFIG")
    if env:
        return env
    root = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
    return os.path.join(root, ".claude", "frugal-sensitivity.json")


def load_rules(path):
    """Return the rule list. Raises ValueError on any malformed config."""
    with open(path) as f:
        config = json.load(f)
    rules = config.get("rules") if isinstance(config, dict) else config
    if not isinstance(rules, list) or not all(isinstance(r, dict) for r in rules):
        raise ValueError("rules must be a list of objects")
    for r in rules:
        for p in r.get("patterns") or []:
            re.compile(p)  # raises re.error -> treated as malformed
    return rules


def classify(text, rules):
    """Return (pattern_hits, path_hits): sorted rule names matched by content
    regex, and by path glob only. Same semantics as guard_sensitive.py."""
    tokens = text.split()
    pattern_hits, path_hits = set(), set()
    for r in rules:
        name = r.get("name", "unnamed")
        if any(re.search(p, text) for p in r.get("patterns") or []):
            pattern_hits.add(name)
        elif any(fnmatch.fnmatch(t, g) for g in r.get("paths") or [] for t in tokens):
            path_hits.add(name)
    return sorted(pattern_hits), sorted(path_hits)


def emit(system_message=None, context=None):
    out = {}
    if system_message:
        out["systemMessage"] = system_message
    if context:
        out["hookSpecificOutput"] = {"hookEventName": "UserPromptSubmit",
                                     "additionalContext": context}
    if out:
        print(json.dumps(out))
    return 0


def block(msg):
    print(f"lab prompt check: {msg}", file=sys.stderr)
    return 2


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # unreadable hook input: nothing to check
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return 0

    path = config_path(payload)
    if not os.path.exists(path):
        return 0  # gate is opt-in, like the frugal gate
    try:
        rules = load_rules(path)
    except Exception:
        return block(f"sensitivity config at {path} is broken, so every prompt is "
                     "blocked (fail closed). Fix the file, or delete it to turn checks off.")

    if prompt.lstrip().lower().startswith(SYNTHETIC_MARKER):
        return emit(context=(
            "The user has attested with [synthetic-data] that this message contains "
            "only synthetic or public test data, not real customer or case data."))

    pattern_hits, path_hits = classify(prompt, rules)

    if pattern_hits:
        return block(
            f"message NOT sent - it matched sensitivity rule(s): {', '.join(pattern_hits)}. "
            "Real customer, case or secret data must not go to a cloud model. "
            "Use the local stack (Ollama / Open WebUI) instead. If the data is fake, "
            f"start the message with {SYNTHETIC_MARKER} and resend.")

    if path_hits:
        return emit(
            system_message=(f"Warning: this message mentions restricted path(s) "
                            f"({', '.join(path_hits)}). Claude has been told not to open them."),
            context=("Lab policy: the user's message mentions restricted locations "
                     f"(rules: {', '.join(path_hits)}). Do NOT read, cat, grep or otherwise "
                     "open the contents of data/real/, data/restricted/, cases/, *.sar, "
                     "secrets/, key files or .env files. Editing ignore files or "
                     "documentation that merely names these paths is fine."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
