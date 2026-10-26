"""English legal text for the synthetic generators, so the recogniser also reads Latin text without Vietnamese marks
(contracts, bilingual documents, foreign company names, addresses, e-mails).

Two sources, both optional and both written into the same `corpus.jsonl` as documents with `"lang": "en"`:
  * `template_documents` - clauses assembled from phrase pools; needs no download, so a Colab run always has some English.
  * `stream_documents` - plain-text rows of a Hub dataset you choose (`--en-dataset`); check its licence first."""
from __future__ import annotations

import random
from typing import Iterator

from . import charset, legal_text

_PARTIES = ["the Seller", "the Buyer", "the Lessor", "the Lessee", "the Contractor", "the Employer", "the Company", "the Borrower", "the Lender", "the Licensor"]
_ACTIONS = ["deliver the Goods", "pay the Fee", "notify the other Party in writing", "keep the Confidential Information secret", "obtain all permits required by law",
            "indemnify the other Party against any loss", "provide the Services with reasonable skill and care", "return the Deposit", "maintain adequate insurance",
            "comply with applicable laws and regulations"]
_TIMES = ["within thirty (30) days of the Effective Date", "no later than 15 business days after receipt of the invoice", "within a reasonable period", "promptly and in any event within 7 days",
          "before the end of each calendar quarter", "upon termination of this Agreement"]
_CONDITIONS = ["Subject to Clause {n}", "Unless otherwise agreed in writing", "Without prejudice to any other right or remedy", "In the event of a material breach", "To the extent permitted by law",
               "Notwithstanding anything to the contrary in this Agreement"]
_REFS = ["Article {n} of the Law on Enterprises", "Section {n}.{m} of this Agreement", "Clause {n}({l}) above", "Schedule {n} attached hereto", "Decree No. {a}/{y}/ND-CP",
         "Circular No. {a}/{y}/TT-BTC", "ISO {iso}:{y}", "the Civil Code No. {a}/{y}/QH{q}"]
_TITLES = ["SERVICE AGREEMENT", "SALES CONTRACT", "LEASE AGREEMENT", "POWER OF ATTORNEY", "MEMORANDUM OF UNDERSTANDING", "CONFIDENTIALITY AGREEMENT", "EMPLOYMENT CONTRACT", "LOAN AGREEMENT"]
_FIRST = ["John", "Mary", "David", "Anna", "Robert", "Linda", "Michael", "Sarah", "Kim", "Chen", "Tanaka", "Park", "Nguyen", "Tran", "Le", "Pham"]
_LAST = ["Smith", "Johnson", "Williams", "Brown", "Miller", "Wilson", "Taylor", "Anderson", "Lee", "Wang", "Sato", "Kim", "Nguyen", "Tran", "Hoang", "Vu"]
_COMPANY_A = ["Global", "Pacific", "Sunrise", "Delta", "Apex", "Blue Ocean", "Golden Dragon", "Northern", "Vina", "Saigon", "Mekong", "Orient"]
_COMPANY_B = ["Trading", "Logistics", "Technology", "Holdings", "Consulting", "Engineering", "Capital", "Industries", "Solutions", "Foods"]
_SUFFIX = ["Co., Ltd.", "JSC", "Inc.", "Corp.", "LLC", "Ltd.", "Pte. Ltd.", "GmbH", "S.A."]
_STREETS = ["Nguyen Hue Street", "Le Loi Boulevard", "Tran Hung Dao Street", "Main Street", "Park Avenue", "Victoria Road", "Hoang Quoc Viet Street"]
_CITIES = ["Hanoi", "Ho Chi Minh City", "Da Nang", "Hai Phong", "Singapore", "Tokyo", "Seoul", "London", "New York", "Berlin"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
_CURRENCY = ["USD", "VND", "EUR", "JPY", "SGD"]
_FIELDS = ["Full name", "Date of birth", "Passport No.", "Tax code", "Registered address", "Telephone", "Position", "Place of issue", "Account No.", "Email"]

# Function words that prove a snippet is running English text rather than Latin-looking noise.
_STOP = frozenset("the of and to in is shall be by for or any that this with as on at from are its it not which such may will under other all each party agreement".split())


def looks_english(text: str, min_ratio: float = 0.2) -> bool:
    """ASCII-only words of which a fair share are English function words. Short snippets (under 3 words) pass when they are ASCII."""
    words = [w for w in (t.strip(",.;:()[]\"'") for t in text.split()) if any(c.isalpha() for c in w)]
    if not words or not all(c.isascii() for w in words for c in w):
        return False
    if len(words) < 3:
        return True
    return sum(w.lower() in _STOP for w in words) / len(words) >= min_ratio


def person(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"


def company(rng: random.Random) -> str:
    return f"{rng.choice(_COMPANY_A)} {rng.choice(_COMPANY_B)} {rng.choice(_SUFFIX)}"


def date_en(rng: random.Random) -> str:
    d, m, y = rng.randint(1, 28), rng.choice(_MONTHS), rng.randint(1995, 2026)
    return rng.choice([f"{m} {d}, {y}", f"{d} {m} {y}", f"{d}/{rng.randint(1, 12)}/{y}", f"{y}-{rng.randint(1, 12):02d}-{d:02d}"])


def address(rng: random.Random) -> str:
    return f"{rng.randint(1, 400)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"


def email(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST).lower()}.{rng.choice(_LAST).lower()}@{rng.choice(_COMPANY_A).lower().replace(' ', '')}.{rng.choice(['com', 'vn', 'com.vn', 'org'])}"


def amount_en(rng: random.Random) -> str:
    n = rng.choice([rng.randint(1, 99), rng.randint(100, 9999), rng.randint(10_000, 9_999_999)])
    text = f"{n:,}" + (f".{rng.randint(1, 99):02d}" if rng.random() < 0.4 else "")
    cur = rng.choice(_CURRENCY)
    return rng.choice([f"{cur} {text}", f"{text} {cur}", f"US${text}"])


def reference(rng: random.Random) -> str:
    return rng.choice(_REFS).format(n=rng.randint(1, 60), m=rng.randint(1, 9), l=rng.choice("abcdefgh"), a=rng.randint(1, 300), y=rng.randint(1995, 2026), iso=rng.choice([9001, 14001, 27001, 45001]), q=rng.randint(10, 15))


def doc_number(rng: random.Random) -> str:
    return f"No. {rng.randint(1, 9999)}/{rng.randint(1995, 2026)}/{rng.choice(['SC', 'HD', 'PL', 'CT', 'AG', 'LA'])}-{rng.choice(['HN', 'HCM', 'DN', 'HP', 'SG'])}"


def form_field(rng: random.Random) -> str:
    label = rng.choice(_FIELDS)
    value = rng.choice([person(rng), date_en(rng), address(rng), email(rng), str(rng.randint(10**7, 10**12 - 1)), company(rng)])
    return f"{label}: {value}" if rng.random() < 0.7 else f"{label}: {'.' * rng.randint(20, 60)}"


GENERATORS = [person, company, date_en, address, email, amount_en, reference, doc_number, form_field]


def random_fragment(rng: random.Random) -> str:
    return rng.choice(GENERATORS)(rng)


def clause(rng: random.Random) -> str:
    cond = rng.choice(_CONDITIONS).format(n=rng.randint(1, 30))
    text = f"{cond}, {rng.choice(_PARTIES)} shall {rng.choice(_ACTIONS)} {rng.choice(_TIMES)}"
    if rng.random() < 0.5:
        text += f", as provided in {reference(rng)}"
    if rng.random() < 0.3:
        text += f", in the amount of {amount_en(rng)}"
    return text + "."


def template_documents(count: int, seed: int = 0) -> Iterator[dict]:
    """`count` synthetic English contracts in the corpus block format (title, parties, numbered clauses, a small table)."""
    for i in range(count):
        rng = random.Random(f"en-template:{seed}:{i}")
        blocks = [{"kind": "title", "text": rng.choice(_TITLES), "center": True, "bold": True, "cells": None},
                  {"kind": "text", "text": f"This Agreement is made on {date_en(rng)} between {company(rng)}, having its registered office at {address(rng)} (\"{rng.choice(_PARTIES)[4:].capitalize()}\"), and {company(rng)}, "
                                           f"having its registered office at {address(rng)}.", "center": False, "bold": False, "cells": None}]
        for n in range(1, rng.randint(8, 16)):
            blocks.append({"kind": "heading", "text": f"Article {n}. {rng.choice(['Scope', 'Payment', 'Term', 'Liability', 'Notices', 'Governing law', 'Confidentiality', 'Termination'])}", "center": False, "bold": True, "cells": None})
            for k in range(1, rng.randint(2, 4)):
                blocks.append({"kind": "text", "text": f"{n}.{k} {clause(rng)}", "center": False, "bold": False, "cells": None})
            if rng.random() < 0.15:
                cells = ["Item", "Quantity", "Unit price", "Amount"]
                blocks.append({"kind": "table_row", "text": " | ".join(cells), "center": False, "bold": False, "cells": cells})
                for _ in range(rng.randint(2, 4)):
                    row = [rng.choice(["Steel pipe", "Cotton fabric", "Software licence", "Consulting service", "Rice"]), str(rng.randint(1, 5000)), amount_en(rng), amount_en(rng)]
                    blocks.append({"kind": "table_row", "text": " | ".join(row), "center": False, "bold": False, "cells": row})
        blocks.append({"kind": "text", "text": f"Signed by {person(rng)} on behalf of {company(rng)}; contact {email(rng)}.", "center": False, "bold": False, "cells": None})
        yield {"id": f"en-template-{seed}-{i}", "lang": "en", "blocks": blocks}


def stream_documents(dataset: str, limit: int, *, config: str | None = None, column: str = "text", split: str = "train", seed: int = 0) -> Iterator[dict]:
    """Plain-text rows of a Hub dataset, one paragraph per line, cut into corpus blocks. Rows that are not English are skipped."""
    from datasets import load_dataset

    stream = load_dataset(dataset, config, split=split, streaming=True).shuffle(seed=seed, buffer_size=2000)
    allowed = set(charset.build_charset())
    made = 0
    for i, row in enumerate(stream):
        if made >= limit:
            return
        blocks = []
        for para in str(row[column]).splitlines():
            text = charset.restrict(charset.normalize(para), allowed, 0.99)
            if text and 4 <= len(text) <= 400 and looks_english(text):
                blocks.append({"kind": "text", "text": text, "center": False, "bold": False, "cells": None})
        if len(blocks) >= 8:
            made += 1
            yield {"id": f"en-{dataset}-{i}", "lang": "en", "blocks": blocks[:400]}
