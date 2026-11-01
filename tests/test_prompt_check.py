"""Tests for .claude/hooks/check_prompt.py (UserPromptSubmit hook).

Reuses the POSITIVE / NEGATIVE cases from test_sensitivity_gate.py so both
checks are always held to the same examples. Runs the real script in a
subprocess, exactly as Claude Code would.

Run:  python3 -m pytest tests/ -q
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_sensitivity_gate import CONFIG, NEGATIVE, POSITIVE, RULES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude" / "hooks" / "check_prompt.py"


def run(prompt, config=CONFIG, payload_extra=None):
    payload = {"hook_event_name": "UserPromptSubmit", "prompt": prompt, "cwd": str(ROOT)}
    payload.update(payload_extra or {})
    env = dict(os.environ, FRUGAL_SENSITIVITY_CONFIG=str(config))
    return subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload),
                          text=True, capture_output=True, env=env)


def content_match(rule_name, text):
    return any(re.search(p, text) for p in RULES[rule_name].get("patterns", []))


BLOCK_CASES = [(r, t) for r, t in POSITIVE if content_match(r, t)]
WARN_CASES = [(r, t) for r, t in POSITIVE if not content_match(r, t)]


def test_cases_split_both_ways():
    # Guards against the shared fixtures drifting so one branch goes untested.
    assert BLOCK_CASES and WARN_CASES


@pytest.mark.parametrize("rule,text", BLOCK_CASES)
def test_identifiers_are_blocked(rule, text):
    r = run(text)
    assert r.returncode == 2, r.stderr
    assert rule in r.stderr and "NOT sent" in r.stderr


@pytest.mark.parametrize("rule,text", BLOCK_CASES)
def test_block_message_never_echoes_the_data(rule, text):
    r = run(text)
    for token in re.findall(r"[A-Z0-9][A-Z0-9 -]{7,}[A-Z0-9]", text):
        assert token not in r.stderr, f"leaked {token!r}"


@pytest.mark.parametrize("rule,text", WARN_CASES)
def test_restricted_paths_warn_but_allow(rule, text):
    r = run(text)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert rule in out["systemMessage"]
    assert "Do NOT read" in out["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("text", NEGATIVE)
def test_everyday_prompts_pass_silently(text):
    r = run(text)
    assert r.returncode == 0 and r.stdout == "" and r.stderr == ""


def test_synthetic_marker_allows_and_attests():
    r = run("[synthetic-data] test IBAN GB82 WEST 1234 5698 7654 32")
    assert r.returncode == 0
    ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "synthetic" in ctx


def test_marker_must_lead_the_message():
    # Burying the marker mid-message must not bypass the check.
    r = run("IBAN GB82 WEST 1234 5698 7654 32 [synthetic-data]")
    assert r.returncode == 2


def test_broken_config_fails_closed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    assert run("harmless text", config=bad).returncode == 2


def test_invalid_regex_fails_closed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rules": [{"name": "x", "patterns": ["(unclosed"]}]}))
    assert run("harmless text", config=bad).returncode == 2


def test_missing_config_means_off(tmp_path):
    assert run("IBAN GB82 WEST 1234 5698 7654 32", config=tmp_path / "none.json").returncode == 0


def test_garbage_input_is_ignored():
    r = subprocess.run([sys.executable, str(HOOK)], input="not json",
                       text=True, capture_output=True)
    assert r.returncode == 0


def test_hook_is_registered_in_project_settings():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    cmds = [h["command"] for entry in settings["hooks"].get("UserPromptSubmit", [])
            for h in entry["hooks"]]
    assert any("check_prompt.py" in c for c in cmds)
