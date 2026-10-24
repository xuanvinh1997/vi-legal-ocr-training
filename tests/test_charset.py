import unicodedata

from ocr_train import charset


def test_vietnamese_letters_complete_and_nfc():
    letters = charset.vietnamese_letters()
    assert len(letters) == len(set(letters))
    assert all(unicodedata.normalize("NFC", c) == c and len(c) == 1 for c in letters)
    for c in "ảạấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹđĐƠƯ":
        assert c in letters


def test_build_charset_no_space_no_dupes():
    chars = charset.build_charset()
    assert " " not in chars and len(chars) == len(set(chars))
    assert 250 < len(chars) < 400


def test_normalize_nfd_to_nfc():
    nfd = unicodedata.normalize("NFD", "Điều 5. Cộng hòa")
    assert charset.normalize(nfd) == "Điều 5. Cộng hòa"
    assert charset.normalize("a  b​") == "a b"


def test_extend_dict_keeps_indices():
    up = ["a", "b", "c"]
    out = charset.extend_dict(up, ["b", "ả", "ạ"])
    assert out[:3] == up and out[3:] == ["ả", "ạ"]


def test_restrict():
    cs = set(charset.build_charset())
    assert charset.restrict("Điều 1 中", cs, 0.8) == "Điều 1"
    assert charset.restrict("中中中中", cs) is None


def test_looks_vietnamese():
    assert charset.looks_vietnamese("Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015")
    assert charset.looks_vietnamese("UBND tỉnh ban hành Quyết định số 12/2023/QĐ-UBND")
    # TCVN3-decoded garbage
    assert not charset.looks_vietnamese("C¨n cø LuËt Tæ chøc chÝnh quyÒn ®Þa ph­¬ng")
    assert not charset.looks_vietnamese("Xxqz bbcd kkkl mnpq")
