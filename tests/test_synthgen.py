"""Tests for synthgen: the dataset must be reproducible, provably fake,
internally consistent, and its answer key must be correct.

Run:  python3 -m pytest tests/ -q
"""
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from synthgen.__main__ import generate  # noqa: E402
from synthgen.core import CASH_THRESHOLD, HIGH_RISK_COUNTRIES, TYPOLOGIES, Bank, iban_is_valid  # noqa: E402
from synthgen.rules import RULE_TYPOLOGY, run_rules  # noqa: E402
from test_sensitivity_gate import RULES as GATE_RULES  # noqa: E402


@pytest.fixture(scope="module")
def bank():
    b = Bank(seed=42, customers=200, days=90).build()
    b.alerts = run_rules(b)
    return b


def txn_index(b):
    return {t["txn_id"]: t for t in b.transactions}


# --- reproducibility ------------------------------------------------------------

def test_same_seed_same_data():
    a = Bank(seed=7, customers=40, days=45).build()
    b = Bank(seed=7, customers=40, days=45).build()
    assert a.transactions == b.transactions and a.cases == b.cases


def test_different_seed_different_data():
    a = Bank(seed=7, customers=40, days=45).build()
    b = Bank(seed=8, customers=40, days=45).build()
    assert a.transactions != b.transactions


# --- provably fake --------------------------------------------------------------

def test_every_iban_is_well_formed_and_uses_fake_bank(bank):
    ibans = [a["iban"] for a in bank.accounts] + \
            [t["counterparty_iban"] for t in bank.transactions if t["counterparty_iban"]]
    for iban in ibans:
        assert re.fullmatch(r"GB\d{2}TEST\d{14}", iban), iban
        assert iban_is_valid(iban), iban


def test_account_ibans_are_unique(bank):
    ibans = [a["iban"] for a in bank.accounts]
    assert len(ibans) == len(set(ibans))


def test_generated_ibans_still_trip_the_sensitivity_gate(bank):
    # Deliberate: synthetic data rehearses production handling (docs/local-stack.md limit #3).
    assert any(re.search(p, bank.accounts[0]["iban"]) for p in GATE_RULES["iban"]["patterns"])


# --- internal consistency -------------------------------------------------------

def test_ids_unique_and_references_resolve(bank):
    txn_ids = [t["txn_id"] for t in bank.transactions]
    assert len(txn_ids) == len(set(txn_ids))
    accts = {a["account_id"] for a in bank.accounts}
    custs = {c["customer_id"] for c in bank.customers}
    assert all(t["account_id"] in accts for t in bank.transactions)
    assert all(a["customer_id"] in custs for a in bank.accounts)
    idx = txn_index(bank)
    for item in bank.cases + bank.alerts:
        assert all(i in idx for i in item["txn_ids"])


def test_transactions_are_chronological_and_in_window(bank):
    ts = [t["timestamp"] for t in bank.transactions]
    assert ts == sorted(ts)
    assert (ts[-1] - ts[0]).days <= 90


# --- ground truth: each typology really looks like itself ------------------------

def cases_of(bank, typology):
    idx = txn_index(bank)
    return [[idx[i] for i in c["txn_ids"]] for c in bank.cases if c["typology"] == typology]


def test_every_typology_is_present(bank):
    assert {c["typology"] for c in bank.cases} == set(TYPOLOGIES)


def test_case_txns_belong_to_the_case_customer(bank):
    owner = {a["account_id"]: a["customer_id"] for a in bank.accounts}
    idx = txn_index(bank)
    for c in bank.cases:
        assert {owner[idx[i]["account_id"]] for i in c["txn_ids"]} == {c["customer_id"]}


def test_structuring_shape(bank):
    for txns in cases_of(bank, "structuring"):
        assert len(txns) >= 4
        assert all(t["channel"] == "CASH" and t["direction"] == "IN" for t in txns)
        assert all(0.8 * CASH_THRESHOLD <= t["amount"] < CASH_THRESHOLD for t in txns)
        assert max(t["timestamp"] for t in txns) - min(t["timestamp"] for t in txns) <= timedelta(days=10)


def test_mule_shape(bank):
    for txns in cases_of(bank, "mule_pass_through"):
        ins = [t for t in txns if t["direction"] == "IN"]
        outs = [t for t in txns if t["direction"] == "OUT"]
        assert len(ins) == len(outs) >= 5
        assert len({t["counterparty_name"] for t in outs}) == 1  # single beneficiary


def test_high_risk_shape(bank):
    for txns in cases_of(bank, "high_risk_jurisdiction"):
        assert all(t["direction"] == "OUT" and t["counterparty_country"] in HIGH_RISK_COUNTRIES
                   for t in txns)


def test_funnel_shape(bank):
    for txns in cases_of(bank, "funnel_account"):
        ins = [t for t in txns if t["direction"] == "IN"]
        assert len({t["counterparty_name"] for t in ins}) >= 10 or len(ins) >= 15
        out_total = sum(t["amount"] for t in txns if t["direction"] == "OUT")
        assert out_total == pytest.approx(0.92 * sum(t["amount"] for t in ins), rel=1e-3)


# --- alerts and the answer key ----------------------------------------------------

def test_every_hidden_case_is_alerted(bank):
    alerted = {(a["customer_id"], RULE_TYPOLOGY[a["rule_id"]]) for a in bank.alerts}
    missed = [c for c in bank.cases if (c["customer_id"], c["typology"]) not in alerted]
    assert not missed, missed


def test_labels_match_ground_truth(bank):
    truth = {(c["customer_id"], c["typology"]) for c in bank.cases}
    for a in bank.alerts:
        expected = (a["customer_id"], RULE_TYPOLOGY[a["rule_id"]]) in truth
        assert (a["label"] == "TRUE_POSITIVE") == expected


def test_queue_has_false_positives_to_triage(bank):
    labels = {a["label"] for a in bank.alerts}
    assert labels == {"TRUE_POSITIVE", "FALSE_POSITIVE"}


def test_false_positives_come_from_legitimate_lookalikes(bank):
    occ = {c["customer_id"]: c["occupation"] for c in bank.customers}
    fp = [a for a in bank.alerts if a["label"] == "FALSE_POSITIVE"]
    assert any(a["rule_id"] == "R01_CASH_STRUCTURING" and occ[a["customer_id"]] == "Market trader" for a in fp)
    assert any(a["rule_id"] == "R04_MANY_PAYERS" for a in fp)


def test_risk_rating_does_not_leak_the_answer(bank):
    bad = {c["customer_id"] for c in bank.cases}
    ratings = defaultdict(set)
    for c in bank.customers:
        ratings[c["customer_id"] in bad].add(c["risk_rating"])
    assert "LOW" in ratings[True]  # some bad actors look low-risk, as in real life


# --- files on disk -------------------------------------------------------------

def test_writes_all_files_with_synthetic_manifest(tmp_path):
    m = generate(seed=1, customers=40, days=45, out=tmp_path / "run")
    names = {p.name for p in (tmp_path / "run").iterdir()}
    assert names == {"customers.csv", "accounts.csv", "transactions.csv", "alerts.csv",
                     "ground_truth.csv", "rules.csv", "manifest.json"}
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["synthetic"] is True and manifest == json.loads(json.dumps(m))
    with open(tmp_path / "run" / "transactions.csv") as f:
        assert sum(1 for _ in csv.DictReader(f)) == m["counts"]["transactions"]


@pytest.mark.parametrize("bad", ["data/real/x", "data/restricted", "cases/c1"])
def test_refuses_restricted_output(bad, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="restricted"):
        generate(seed=1, customers=40, days=45, out=bad)
    assert not (tmp_path / bad).exists()
