"""Reciprocal Rank Fusion tests."""

from __future__ import annotations

import pytest

from retrieval.fusion import rank_map, rrf_fuse


def test_rank_is_one_based_and_k_is_sixty() -> None:
    fused = rrf_fuse([["a"], ["a"]])

    assert fused[0][0] == "a"
    assert fused[0][1] == 2 * (1 / 61)


def test_agreement_beats_single_list_position() -> None:
    lexical = ["x", "a", "b"]
    vector = ["y", "a", "b"]

    fused = rrf_fuse([lexical, vector])
    order = [doc_id for doc_id, _ in fused]

    assert order[0] == "a"
    assert order[1] == "b"
    assert set(order[2:]) == {"x", "y"}


def test_position_matters_within_a_list() -> None:
    fused = dict(rrf_fuse([["first", "second"]]))

    assert fused["first"] > fused["second"]


def test_disjoint_lists_are_all_kept() -> None:
    fused = rrf_fuse([["a", "b"], ["c", "d"]])

    assert {doc_id for doc_id, _ in fused} == {"a", "b", "c", "d"}
    assert fused[0][0] == "a"


def test_ties_are_broken_by_id_for_determinism() -> None:
    fused = rrf_fuse([["b"], ["a"]])

    assert fused == [("a", 1 / 61), ("b", 1 / 61)]


def test_top_n_truncates() -> None:
    assert len(rrf_fuse([["a", "b", "c"], ["c", "d"]], top_n=2)) == 2


def test_k_parameter_changes_the_weights() -> None:
    assert dict(rrf_fuse([["a"]], k=10))["a"] == 1 / 11


def test_rank_map() -> None:
    assert rank_map(["a", ("b", 0.9)]) == {"a": 1, "b": 2}


def test_weights_default_to_one_each() -> None:
    assert rrf_fuse([["a"], ["b"]]) == rrf_fuse([["a"], ["b"]], weights=[1.0, 1.0])


def test_weight_scales_a_whole_list() -> None:
    fused = dict(rrf_fuse([["lex"], ["vec"]], weights=[0.5, 1.0]))

    assert fused["vec"] == 1 / 61
    assert fused["lex"] == 0.5 / 61


def test_zero_weight_removes_a_list_from_the_ranking() -> None:
    fused = rrf_fuse([["lex"], ["vec"]], weights=[0.0, 1.0])

    assert fused == [("vec", 1 / 61)]


def test_lower_lexical_weight_lets_the_vector_list_win() -> None:
    unweighted = [doc_id for doc_id, _ in rrf_fuse([["a_lex"], ["b_vec"]])]
    weighted = [doc_id for doc_id, _ in rrf_fuse([["a_lex"], ["b_vec"]], weights=[0.3, 1.0])]

    assert unweighted[0] == "a_lex"  # tie on score, broken by id
    assert weighted[0] == "b_vec"


def test_weights_must_match_the_number_of_lists() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([["a"], ["b"]], weights=[1.0])

