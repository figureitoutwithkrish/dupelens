"""Unit tests for src/numtokens.py with realistic title fragments."""
import pytest

from src.numtokens import compare, extract, parse_number


def flags(a, b):
    return compare(extract(a), extract(b))


def test_1_litre_equals_1000_ml():
    assert flags("Botol Minum 1 L", "Botol Minum 1000 ml") == (True, False)


def test_500_ml_vs_1_litre_conflicts():
    assert flags("Shampoo 500 ml", "Shampoo 1 l") == (False, True)


def test_decimal_point_and_comma():
    assert extract("Minyak Goreng 1.5l") == {("volume", 1500.0)}
    assert extract("Beras 1,5 kg") == {("mass", 1500.0)}


def test_thousands_separator():
    assert parse_number("1.000") == 1000.0
    assert parse_number("1,000") == 1000.0
    assert extract("Gula 1.000 gr") == {("mass", 1000.0)}
    assert flags("Kopi 1 kg", "Kopi 1.000gr") == (True, False)


def test_storage_units_normalise():
    assert flags("Flashdisk 1 TB", "Flashdisk 1024 GB") == (True, False)
    assert flags("Memory card 64GB", "Memory card 32 GB") == (False, True)


def test_pack_and_set_counts():
    assert extract("Earrings pack of 2") == {("count", 2.0)}
    assert flags("Bangles set of 4", "Bangles set of 6") == (False, True)
    assert flags("Masker isi 50", "Masker 50 pcs") == (True, False)


def test_isi_followed_by_unit_is_not_a_count():
    assert extract("Sabun cair isi 500ml") == {("volume", 500.0)}


def test_2_in_1_is_not_a_length():
    assert extract("Kabel data 2 in 1") == set()
    assert extract('Phone case 6.5 inch') == {("length", 165.1)}


def test_different_kinds_do_not_conflict():
    assert flags("Lotion 200 ml", "Lotion pack of 2") == (False, False)


def test_no_tokens_means_no_match_and_no_conflict():
    assert flags("Kalung wanita emas", "Kalung wanita perak") == (False, False)


def test_unit_must_end_the_word():
    # "2 layer" must not be read as 2 litres
    assert extract("Kotak makan 2 layer") == set()
