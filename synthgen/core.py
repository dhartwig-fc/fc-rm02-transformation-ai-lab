"""Synthetic bank: customers, accounts, transactions, injected typologies.

Everything is invented. The same seed always produces exactly the same data.
IBANs use the bank code TEST (no such bank exists), so they have a real IBAN
shape and valid check digits but are provably fake.

Known simplifications (v0.1): one account per customer, no balances, GBP only,
domestic counterparties use fake GB IBANs, high-risk countries are illustrative.
"""
import random
from datetime import datetime, timedelta

GENERATOR_VERSION = "0.1.0"
START = datetime(2026, 1, 1, 8, 0, 0)
FAKE_BANK_CODE = "TEST"

# Illustrative internal threshold for cash-structuring rules (not a legal limit).
CASH_THRESHOLD = 10_000
# Illustrative only - always check the current FATF lists before relying on this.
HIGH_RISK_COUNTRIES = ["IR", "KP", "MM"]

TYPOLOGIES = ["structuring", "mule_pass_through", "high_risk_jurisdiction", "funnel_account"]
NOISE_KINDS = ["cash_business", "group_collection"]  # legitimate look-alikes -> false positives

FIRST = ["Amelia", "Oliver", "Isla", "George", "Ava", "Noah", "Mia", "Arthur", "Freya", "Leo",
         "Priya", "Mohammed", "Zara", "Tomasz", "Chen", "Aisha", "Daniel", "Grace", "Kofi", "Sofia",
         "Liam", "Hannah", "Ravi", "Elena", "Jack", "Niamh", "Yusuf", "Chloe", "Marek", "Lucy"]
LAST = ["Smith", "Jones", "Taylor", "Brown", "Williams", "Wilson", "Patel", "Khan", "Evans",
        "Thomas", "Roberts", "Walker", "Wright", "Nowak", "Okafor", "Murphy", "Hughes", "Green",
        "Hall", "Wood", "Ali", "Clarke", "Shah", "Kowalski", "Edwards", "Turner", "Begum", "Lewis"]
OCCUPATIONS = ["Teacher", "Nurse", "Software developer", "Electrician", "Student", "Accountant",
               "Retail assistant", "Chef", "Driver", "Retired", "Civil servant", "Self-employed"]
EMPLOYERS = ["Harbourline Freight Ltd", "Brightwell Health Trust", "Corvane Systems Ltd",
             "Millgate Academy", "Ashdown Retail Group", "Fenwick & Rowe LLP"]
MERCHANTS = ["Grocer", "Coffee shop", "Fuel station", "Online marketplace", "Pharmacy",
             "Restaurant", "Bookshop", "Rail tickets", "Streaming service", "Hardware store"]


def iban_check_digits(country, bban):
    """ISO 13616 check digits: move country+00 to the end, letters -> numbers, mod 97."""
    rearranged = bban + country + "00"
    digits = "".join(str(int(c, 36)) for c in rearranged)
    return f"{98 - int(digits) % 97:02d}"


def iban_is_valid(iban):
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


class Bank:
    def __init__(self, seed=42, customers=200, days=90, typology_rate=0.10, noise_rate=0.05):
        if days < 30:
            raise ValueError("days must be at least 30 so typologies fit in the window")
        self.rng = random.Random(seed)
        self.seed, self.n_customers, self.days = seed, customers, days
        self.typology_rate, self.noise_rate = typology_rate, noise_rate
        self.customers, self.accounts, self.transactions, self.cases = [], [], [], []
        self._ibans = set()

    # --- helpers --------------------------------------------------------------
    def fake_iban(self):
        while True:
            bban = FAKE_BANK_CODE + f"{self.rng.randrange(10**6):06d}{self.rng.randrange(10**8):08d}"
            iban = "GB" + iban_check_digits("GB", bban) + bban
            if iban not in self._ibans:
                self._ibans.add(iban)
                return iban

    def person(self):
        return f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"

    def at(self, day, hour=None):
        hour = self.rng.randint(7, 21) if hour is None else hour
        return START + timedelta(days=day, hours=hour - START.hour,
                                 minutes=self.rng.randint(0, 59), seconds=self.rng.randint(0, 59))

    def txn(self, acct, ts, direction, amount, channel, cp_name, cp_iban="", cp_country="GB", ref=""):
        t = {"txn_id": None, "account_id": acct["account_id"], "timestamp": ts,
             "direction": direction, "amount": round(amount, 2), "currency": "GBP",
             "channel": channel, "counterparty_name": cp_name, "counterparty_iban": cp_iban,
             "counterparty_country": cp_country, "reference": ref}
        self.transactions.append(t)
        return t

    def window_start(self, length):
        return self.rng.randint(5, self.days - length - 2)

    # --- build ----------------------------------------------------------------
    def build(self):
        rng = self.rng
        for i in range(1, self.n_customers + 1):
            cust = {"customer_id": f"CUST-{i:05d}", "name": self.person(),
                    "date_of_birth": f"{rng.randint(1950, 2004)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                    "nationality": rng.choices(["GB", "IE", "PL", "IN", "NG", "PK", "FR"],
                                               [70, 5, 6, 6, 4, 5, 4])[0],
                    "occupation": rng.choice(OCCUPATIONS),
                    # Assigned before typologies, so risk rating does NOT leak the answer.
                    "risk_rating": rng.choices(["LOW", "MEDIUM", "HIGH"], [70, 25, 5])[0],
                    "onboarded": f"{rng.randint(2010, 2025)}-{rng.randint(1, 12):02d}-01"}
            acct = {"account_id": f"ACC-{i:05d}", "customer_id": cust["customer_id"],
                    "iban": self.fake_iban(), "product": "Current account", "opened": cust["onboarded"]}
            self.customers.append(cust)
            self.accounts.append(acct)
            self.baseline(acct)

        # Pick distinct customers for typologies and noise; at least one of each kind.
        idx = list(range(self.n_customers))
        rng.shuffle(idx)
        n_typ = max(len(TYPOLOGIES), round(self.typology_rate * self.n_customers))
        n_noise = max(len(NOISE_KINDS), round(self.noise_rate * self.n_customers))
        if n_typ + n_noise > self.n_customers:
            raise ValueError("too few customers for the requested typology/noise rates")
        for k, i in enumerate(idx[:n_typ]):
            typology = TYPOLOGIES[k % len(TYPOLOGIES)]
            txns = getattr(self, typology)(self.accounts[i], self.customers[i])
            self.cases.append({"customer_id": self.customers[i]["customer_id"], "typology": typology,
                               "txns": txns})
        for k, i in enumerate(idx[n_typ:n_typ + n_noise]):
            getattr(self, NOISE_KINDS[k % len(NOISE_KINDS)])(self.accounts[i], self.customers[i])

        # Chronological IDs (renumber after sorting; cases hold object references).
        self.transactions.sort(key=lambda t: (t["timestamp"], t["account_id"]))
        for n, t in enumerate(self.transactions, 1):
            t["txn_id"] = f"TXN-{n:07d}"
        for n, c in enumerate(self.cases, 1):
            c["case_id"] = f"CASE-{n:04d}"
            c["txn_ids"] = sorted(t["txn_id"] for t in c.pop("txns"))
        return self

    def baseline(self, acct):
        """Ordinary life: salary in, rent out, card spending, occasional ATM."""
        rng = self.rng
        salary = rng.randrange(1800, 6000, 10)
        employer, payday = rng.choice(EMPLOYERS), rng.randint(20, 28)
        rent = round(salary * rng.uniform(0.25, 0.40), 2)
        landlord_iban = self.fake_iban()
        for d in range(self.days):
            day = START + timedelta(days=d)
            if day.day == payday:
                self.txn(acct, self.at(d, 6), "IN", salary, "FPS", employer, self.fake_iban(), ref="SALARY")
            if day.day == 1:
                self.txn(acct, self.at(d, 7), "OUT", rent, "DD", "Rent - Homestead Lettings",
                         landlord_iban, ref="RENT")
            if rng.random() < 0.5:
                self.txn(acct, self.at(d), "OUT", rng.uniform(3, 120), "CARD", rng.choice(MERCHANTS))
            if rng.random() < 0.03:
                self.txn(acct, self.at(d), "OUT", rng.randrange(20, 220, 10), "CASH", "ATM withdrawal")

    # --- typologies (ground truth) ---------------------------------------------
    def structuring(self, acct, cust):
        """Several cash deposits kept just under a threshold, close together."""
        s = self.window_start(10)
        days = sorted(self.rng.sample(range(s, s + 10), self.rng.randint(4, 7)))
        return [self.txn(acct, self.at(d), "IN", self.rng.uniform(0.82, 0.99) * CASH_THRESHOLD,
                         "CASH", "Branch cash deposit") for d in days]

    def mule_pass_through(self, acct, cust):
        """Money arrives from several people and leaves almost immediately."""
        s, out = self.window_start(8), []
        beneficiary, ben_iban = self.person(), self.fake_iban()
        for _ in range(self.rng.randint(5, 8)):
            d = self.rng.randint(s, s + 7)
            amt = self.rng.uniform(500, 2500)
            t_in = self.txn(acct, self.at(d, self.rng.randint(8, 14)), "IN", amt, "FPS",
                            self.person(), self.fake_iban(), ref=self.rng.choice(["thanks", "loan", "for u", ""]))
            t_out = self.txn(acct, t_in["timestamp"] + timedelta(hours=self.rng.uniform(0.3, 6)), "OUT",
                             amt * self.rng.uniform(0.90, 0.98), "FPS", beneficiary, ben_iban)
            out += [t_in, t_out]
        return out

    def high_risk_jurisdiction(self, acct, cust):
        """International transfers to illustrative high-risk countries."""
        s = self.window_start(20)
        return [self.txn(acct, self.at(self.rng.randint(s, s + 20)), "OUT",
                         self.rng.uniform(3000, 20000), "SWIFT", f"{self.person()} Trading",
                         cp_country=self.rng.choice(HIGH_RISK_COUNTRIES), ref="INVOICE")
                for _ in range(self.rng.randint(2, 4))]

    def funnel_account(self, acct, cust):
        """Many unrelated small payments in, then swept out in bulk."""
        s, ins = self.window_start(16), []
        for _ in range(self.rng.randint(15, 30)):
            ins.append(self.txn(acct, self.at(self.rng.randint(s, s + 13)), "IN",
                                self.rng.uniform(100, 900), "FPS", self.person(), self.fake_iban()))
        total, ben, ben_iban = sum(t["amount"] for t in ins), self.person(), self.fake_iban()
        n = self.rng.randint(1, 2)
        outs = [self.txn(acct, self.at(s + 14 + k), "OUT", total * 0.92 / n, "FPS", ben, ben_iban)
                for k in range(n)]
        return ins + outs

    # --- noise (legitimate look-alikes) -----------------------------------------
    def cash_business(self, acct, cust):
        cust["occupation"] = "Market trader"
        s = self.window_start(10)
        for d in sorted(self.rng.sample(range(s, s + 10), self.rng.randint(3, 4))):
            self.txn(acct, self.at(d, 17), "IN", self.rng.uniform(0.82, 0.98) * CASH_THRESHOLD,
                     "CASH", "Branch cash deposit", ref="takings")

    def group_collection(self, acct, cust):
        s, total = self.window_start(8), 0
        for _ in range(self.rng.randint(10, 14)):
            amt = self.rng.uniform(50, 300)
            total += amt
            self.txn(acct, self.at(self.rng.randint(s, s + 6)), "IN", amt, "FPS", self.person(),
                     self.fake_iban(), ref="holiday share")
        self.txn(acct, self.at(s + 7), "OUT", total, "CARD", "Sunward Travel Ltd", ref="group booking")
