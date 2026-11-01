"""Simple transaction-monitoring rules that turn transactions into alerts.

Deliberately basic, like a first-generation rules engine: they catch every
injected typology, and also fire on legitimate look-alikes (false positives),
which is what makes the alert queue useful for practising triage.
"""
from collections import defaultdict
from datetime import timedelta

from .core import CASH_THRESHOLD, HIGH_RISK_COUNTRIES

RULE_TYPOLOGY = {
    "R01_CASH_STRUCTURING": "structuring",
    "R02_RAPID_MOVEMENT": "mule_pass_through",
    "R03_HIGH_RISK_COUNTRY": "high_risk_jurisdiction",
    "R04_MANY_PAYERS": "funnel_account",
}
RULE_DESCRIPTION = {
    "R01_CASH_STRUCTURING": "3+ cash deposits between 80% and 100% of the cash threshold within 10 days",
    "R02_RAPID_MOVEMENT": "3+ incoming payments each followed within 24h by an outgoing payment of 90-110% of the amount",
    "R03_HIGH_RISK_COUNTRY": "Outgoing payment to a high-risk jurisdiction",
    "R04_MANY_PAYERS": "10+ distinct payers within 14 days",
}


def _windowed(txns, days, qualifies):
    """Union of txns in every `days`-long window (anchored at each txn) that qualifies."""
    hits = set()
    for i, first in enumerate(txns):
        window = [t for t in txns[i:] if t["timestamp"] - first["timestamp"] <= timedelta(days=days)]
        if qualifies(window):
            hits.update(id(t) for t in window)
    return [t for t in txns if id(t) in hits]


def r01(txns):
    cash = [t for t in txns if t["channel"] == "CASH" and t["direction"] == "IN"
            and 0.8 * CASH_THRESHOLD <= t["amount"] < CASH_THRESHOLD]
    return _windowed(cash, 10, lambda w: len(w) >= 3)


def r02(txns):
    ins = [t for t in txns if t["direction"] == "IN"]
    outs = [t for t in txns if t["direction"] == "OUT"]
    used, pairs = set(), []
    for i in ins:
        for o in outs:
            if (id(o) not in used and timedelta(0) < o["timestamp"] - i["timestamp"] <= timedelta(hours=24)
                    and 0.9 * i["amount"] <= o["amount"] <= 1.1 * i["amount"]):
                used.add(id(o))
                pairs += [i, o]
                break
    return pairs if len(pairs) >= 6 else []


def r03(txns):
    return [t for t in txns if t["direction"] == "OUT" and t["counterparty_country"] in HIGH_RISK_COUNTRIES]


def r04(txns):
    ins = [t for t in txns if t["direction"] == "IN"]
    return _windowed(ins, 14, lambda w: len({t["counterparty_name"] for t in w}) >= 10)


RULES = {"R01_CASH_STRUCTURING": r01, "R02_RAPID_MOVEMENT": r02,
         "R03_HIGH_RISK_COUNTRY": r03, "R04_MANY_PAYERS": r04}


def run_rules(bank):
    """Return alerts, each labelled TRUE_POSITIVE / FALSE_POSITIVE against ground truth."""
    by_acct = defaultdict(list)
    for t in bank.transactions:
        by_acct[t["account_id"]].append(t)
    owner = {a["account_id"]: a["customer_id"] for a in bank.accounts}
    truth = {(c["customer_id"], c["typology"]) for c in bank.cases}

    alerts = []
    for acct_id in sorted(by_acct):
        txns = by_acct[acct_id]  # already chronological
        for rule_id, rule in RULES.items():
            hits = rule(txns)
            if not hits:
                continue
            cust = owner[acct_id]
            last = max(t["timestamp"] for t in hits)
            alerts.append({
                "alert_id": None, "rule_id": rule_id, "customer_id": cust, "account_id": acct_id,
                "created_at": (last + timedelta(days=1)).replace(hour=6, minute=0, second=0),
                "txn_count": len(hits), "total_amount": round(sum(t["amount"] for t in hits), 2),
                "txn_ids": sorted(t["txn_id"] for t in hits),
                "label": "TRUE_POSITIVE" if (cust, RULE_TYPOLOGY[rule_id]) in truth else "FALSE_POSITIVE",
            })
    alerts.sort(key=lambda a: (a["created_at"], a["alert_id"] or "", a["account_id"]))
    for n, a in enumerate(alerts, 1):
        a["alert_id"] = f"ALT-{n:05d}"
    return alerts
