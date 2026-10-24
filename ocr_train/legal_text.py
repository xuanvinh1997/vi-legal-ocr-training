"""Strings that make legal documents hard for a recogniser and that a plain prose corpus under-represents: document
numbers, dates, article references, amounts and form fields."""
from __future__ import annotations

import random

_TYPES = ["QĐ", "NĐ", "TT", "CT", "NQ", "KH", "TB", "BC", "CV", "TTr", "HD", "PL"]
_ORGS = ["UBND", "CP", "BTC", "BYT", "BGDĐT", "TTg", "HĐND", "BCA", "BNV", "TANDTC", "VKSNDTC", "BTP", "BCT", "NHNN"]
_PLACES = ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Huế", "Nghệ An", "Thanh Hóa", "Lâm Đồng", "Bình Dương", "Quảng Ngãi", "Đắk Lắk"]
_UNITS = ["đồng", "triệu đồng", "tỷ đồng", "ha", "m²", "km", "kg", "%", "tháng", "năm", "ngày"]


def doc_number(rng: random.Random) -> str:
    return f"{rng.randint(1, 999)}/{rng.randint(1995, 2026)}/{rng.choice(_TYPES)}-{rng.choice(_ORGS)}"


def short_number(rng: random.Random) -> str:
    return f"{rng.randint(1, 999)}/{rng.choice(_TYPES)}-{rng.choice(_ORGS)}"


def date_line(rng: random.Random) -> str:
    return f"{rng.choice(_PLACES)}, ngày {rng.randint(1, 31)} tháng {rng.randint(1, 12)} năm {rng.randint(1995, 2026)}"


def article_ref(rng: random.Random) -> str:
    parts = []
    if rng.random() < 0.4:
        parts.append(f"điểm {rng.choice('abcdđeghik')}")
    if rng.random() < 0.7:
        parts.append(f"khoản {rng.randint(1, 12)}")
    parts.append(f"Điều {rng.randint(1, 250)}")
    ref = " ".join(parts)
    if rng.random() < 0.5:
        ref += f" {rng.choice(['Luật', 'Nghị định', 'Thông tư', 'Bộ luật'])} số {doc_number(rng)}"
    return ref


def amount(rng: random.Random) -> str:
    n = rng.choice([rng.randint(1, 99), rng.randint(100, 9999), rng.randint(10_000, 9_999_999_999)])
    text = f"{n:,}".replace(",", ".")
    if rng.random() < 0.15:
        text += f",{rng.randint(1, 99):02d}"
    return f"{text} {rng.choice(_UNITS)}"


def form_field(rng: random.Random) -> str:
    label = rng.choice(["Họ và tên", "Ngày sinh", "Số CMND/CCCD", "Địa chỉ thường trú", "Nơi cấp", "Mã số thuế", "Số điện thoại", "Chức vụ"])
    value = rng.choice([doc_number(rng), str(rng.randint(10**8, 10**12 - 1)), date_line(rng).split(", ", 1)[1], amount(rng)])
    return f"{label}: {value}" if rng.random() < 0.7 else f"{label}: {'.' * rng.randint(20, 60)}"


GENERATORS = [doc_number, short_number, date_line, article_ref, amount, form_field]


def random_fragment(rng: random.Random) -> str:
    return rng.choice(GENERATORS)(rng)
