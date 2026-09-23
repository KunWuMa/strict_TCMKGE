# -*- coding: utf-8 -*-
"""Integrity checks for the strict TCM InGram adaptation."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compositional_kge import make_strict_inductive, read_compositions
from ingram_tcm_baseline import make_eval_message
from train_baselines import Dataset


def main():
    data = Dataset("hybrid_holdout")
    arrays = read_compositions(data)
    make_strict_inductive(data)
    message = make_eval_message(data, arrays, data.test)
    assert len(message) % 2 == 0
    forward = message[: len(message) // 2]
    inverse = message[len(message) // 2 :]
    expected_inverse = forward[:, [2, 1, 0]].copy()
    expected_inverse[:, 1] += data.n_relations
    assert np.array_equal(inverse, expected_inverse)

    held = set(map(int, data.test[:, 0]))
    contains = data.relation_to_id["cpm_contains_chp"]
    held_rows = forward[np.isin(forward[:, 0], np.fromiter(held, dtype=np.int64))]
    assert len(held_rows) > 0
    assert np.all(held_rows[:, 1] == contains)
    assert held == set(map(int, held_rows[:, 0]))

    gold = {tuple(map(int, row)) for row in data.test}
    assert not any(tuple(map(int, row)) in gold for row in forward)
    print("InGram TCM integrity tests passed")


if __name__ == "__main__":
    main()
