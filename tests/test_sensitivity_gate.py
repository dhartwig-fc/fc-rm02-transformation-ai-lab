"""Tests for .claude/frugal-sensitivity.json.

Two layers:
  1. Rule tests - each rule must fire on its positive examples and stay quiet on
     everyday financial crime text (so the gate doesn't block normal lab work).
  2. Hook tests - pipe real Agent-spawn payloads through the frugal plugin's
     guard_sensitive.py and check the exit code (2 = blocked, 0 = allowed).
     Skipped if the plugin isn't installed; set FRUGAL_GUARD_SENSITIVE to its path
     to force a location.

All "sensitive" values below are well-known published test values or made up.

Run:  python3 -m pytest tests/ -q
"""
import fnmatch
import glob
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".claude" / "frugal-sensitivity.json"
RULES = {r["name"]: r for r in json.loads(CONFIG.read_text())["rules"]}


def rule_hits(rule, text):
    """Same semantics as guard_sensitive.py: regex search on the text, or an
    fnmatch glob on any whitespace-separated token."""
    if any(re.search(p, text) for p in rule.get("patterns", [])):
        return True
    tokens = text.split()
    return any(fnmatch.fnmatch(t, g) for g in rule.get("paths", []) for t in tokens)


def rules_hit(text):
    return sorted(name for name, r in RULES.items() if rule_hits(r, text))


# --- 1. Rule tests -----------------------------------------------------------

POSITIVE = [
    ("iban", "Beneficiary IBAN GB82 WEST 1234 5698 7654 32 received funds"),
    ("iban", "Sent to DE89370400440532013000 on 3 March"),
    ("uk-sort-code-and-account", "Sort code 20-00-00, account 55779911"),
    ("uk-sort-code-and-account", "paid from 12 34 56 a/c 31926819"),
    ("payment-card-pan", "card 4111 1111 1111 1111 used at ATM"),
    ("payment-card-pan", "PAN 5555-5555-5555-4444"),
    ("payment-card-pan", "Amex 3782 822463 10005"),
    ("uk-national-insurance-number", "Subject NI number JG103759A"),
    ("uk-national-insurance-number", "NINO: AB 12 34 56 C"),
    ("passport-number-in-context", "UK passport no. 123456789 presented"),
    ("passport-number-in-context", "Passport: AB1234567"),
    ("case-and-sar-material", "Linked to SAR ref 2026-0412-77"),
    ("case-and-sar-material", "DAML reference: DML/88812"),
    ("case-and-sar-material", "Marked OFFICIAL-SENSITIVE"),
    ("case-and-sar-material", "summarise data/real/alerts_oct.csv"),
    ("case-and-sar-material", "read `cases/C-1001/notes.md`"),
    ("secrets-and-keys", "token ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
    ("secrets-and-keys", "key AKIAIOSFODNN7EXAMPLE"),
    ("secrets-and-keys", "-----BEGIN RSA PRIVATE KEY-----"),
    ("secrets-and-keys", "check the values in .env"),
]

NEGATIVE = [
    "HMRC's specimen NINO QQ123456C uses a never-issued prefix",
    "Summarise FATF typologies on trade-based money laundering",
    "Explain what a SAR is and when a DAML request is needed",
    "Write a pytest for the sanctions fuzzy matcher in src/screening/match.py",
    "Refactor scripts/lab.sh and update docs/local-stack.md",
    "Alert raised on 12-03-2026 for structuring below 10000 GBP",
    "Use model qwen3:8b with nomic-embed-text embeddings on port 11434",
    "Compare OFSI, OFAC SDN and EU consolidated list formats",
    "Copy .env.example to get started",
    "Commit 3f9a2c1 changed docker-compose.yml",
    "Transaction ID TXN-000123456 flagged by rule R-17",
]


@pytest.mark.parametrize("rule,text", POSITIVE)
def test_rule_fires(rule, text):
    assert rule in rules_hit(text), f"{rule} should match: {text!r}"


@pytest.mark.parametrize("text", NEGATIVE)
def test_everyday_text_is_not_blocked(text):
    assert rules_hit(text) == [], f"false positive on: {text!r}"


def test_every_rule_has_a_positive_case():
    covered = {r for r, _ in POSITIVE}
    assert covered == set(RULES), f"untested rules: {set(RULES) - covered}"


def test_no_worker_is_allowed_sensitive_data():
    # Lab policy: no sub-agent receives matched data. Loosen deliberately, per rule.
    for name, rule in RULES.items():
        assert rule.get("allow_agents") == [], f"{name} allows workers"


# --- 2. Hook tests (real guard_sensitive.py) ---------------------------------

def find_hook():
    env = os.environ.get("FRUGAL_GUARD_SENSITIVE")
    if env:
        return env
    hits = glob.glob(os.path.expanduser("~/.claude/plugins/**/hooks/guard_sensitive.py"),
                     recursive=True)
    return hits[0] if hits else None


HOOK = find_hook()
needs_hook = pytest.mark.skipif(HOOK is None, reason="frugal plugin hook not installed")


def run_hook(prompt, agent="frugal:extractor", config=CONFIG):
    payload = {"tool_name": "Agent", "cwd": str(ROOT),
               "tool_input": {"subagent_type": agent, "description": "t", "prompt": prompt}}
    env = dict(os.environ, FRUGAL_SENSITIVITY_CONFIG=str(config))
    return subprocess.run(["python3", HOOK], input=json.dumps(payload), text=True,
                          capture_output=True, env=env)


@needs_hook
@pytest.mark.parametrize("rule,text", POSITIVE)
def test_hook_blocks_sensitive_delegation(rule, text):
    r = run_hook(text)
    assert r.returncode == 2 and rule in r.stderr, r.stderr


@needs_hook
@pytest.mark.parametrize("text", NEGATIVE)
def test_hook_allows_everyday_delegation(text):
    r = run_hook(text)
    assert r.returncode == 0, r.stderr


@needs_hook
def test_hook_fails_closed_on_broken_config(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    assert run_hook("harmless text", config=bad).returncode == 2
