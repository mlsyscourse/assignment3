from typing import Dict, List

import numpy as np
import pytest

from data.data_parallel_preprocess import split_data


def check_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mp_size: int,
    dp_size: int,
    rank: int,
    expected_x_train_ret: np.ndarray,
    expected_y_train_ret: np.ndarray,
) -> None:
    x_train_ret, y_train_ret = split_data(
        x_train=x_train,
        y_train=y_train,
        mp_size=mp_size,
        dp_size=dp_size,
        rank=rank,
    )

    assert (
        x_train_ret.shape[0] * dp_size == x_train.shape[0]
    ), f"x_train shape mismatch should be {expected_x_train_ret.shape}"
    assert (
        y_train_ret.shape[0] * dp_size == y_train.shape[0]
    ), f"y_train shape mismatch should be {expected_x_train_ret.shape}"

    np.testing.assert_allclose(actual=x_train_ret, desired=expected_x_train_ret)
    np.testing.assert_allclose(actual=y_train_ret, desired=expected_y_train_ret)


def test_mp_2_dp_1():
    x_train = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [8.0, 9.0],
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    mp_size = 2
    dp_size = 1
    rank_to_x_train = {
        0: np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [8.0, 9.0],
                [10.0, 11.0],
                [12.0, 13.0],
                [14.0, 15.0],
            ]
        ),
        1: np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [8.0, 9.0],
                [10.0, 11.0],
                [12.0, 13.0],
                [14.0, 15.0],
            ]
        ),
    }

    rank_to_y_train = {
        0: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        1: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    }

    for k in rank_to_x_train:
        check_split(
            x_train,
            y_train,
            mp_size,
            dp_size,
            k,
            rank_to_x_train[k],
            rank_to_y_train[k],
        )


def test_mp_1_dp_2():
    x_train = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [8.0, 9.0],
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    mp_size = 1
    dp_size = 2
    rank_to_x_train = {
        0: np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [6.0, 7.0]]),
        1: np.array([[8.0, 9.0], [10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]),
    }

    rank_to_y_train = {
        0: np.array([1.0, 2.0, 3.0, 4.0]),
        1: np.array([5.0, 6.0, 7.0, 8.0]),
    }

    for k in rank_to_x_train:
        check_split(
            x_train,
            y_train,
            mp_size,
            dp_size,
            k,
            rank_to_x_train[k],
            rank_to_y_train[k],
        )


def test_mp_2_dp_2():
    x_train = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [8.0, 9.0],
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    mp_size = 2
    dp_size = 2
    rank_to_x_train = {
        0: np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [6.0, 7.0]]),
        1: np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [6.0, 7.0]]),
        2: np.array([[8.0, 9.0], [10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]),
        3: np.array([[8.0, 9.0], [10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]),
    }

    rank_to_y_train = {
        0: np.array([1.0, 2.0, 3.0, 4.0]),
        1: np.array([1.0, 2.0, 3.0, 4.0]),
        2: np.array([5.0, 6.0, 7.0, 8.0]),
        3: np.array([5.0, 6.0, 7.0, 8.0]),
    }

    for k in rank_to_x_train:
        check_split(
            x_train,
            y_train,
            mp_size,
            dp_size,
            k,
            rank_to_x_train[k],
            rank_to_y_train[k],
        )


def test_mp_2_dp_4():
    x_train = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [8.0, 9.0],
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    mp_size = 2
    dp_size = 4
    rank_to_x_train = {
        0: np.array([[1.0, 2.0], [3.0, 4.0]]),
        1: np.array([[1.0, 2.0], [3.0, 4.0]]),
        2: np.array([[5.0, 6.0], [6.0, 7.0]]),
        3: np.array([[5.0, 6.0], [6.0, 7.0]]),
        4: np.array([[8.0, 9.0], [10.0, 11.0]]),
        5: np.array([[8.0, 9.0], [10.0, 11.0]]),
        6: np.array([[12.0, 13.0], [14.0, 15.0]]),
        7: np.array([[12.0, 13.0], [14.0, 15.0]]),
    }

    rank_to_y_train = {
        0: np.array([1.0, 2.0]),
        1: np.array([1.0, 2.0]),
        2: np.array([3.0, 4.0]),
        3: np.array([3.0, 4.0]),
        4: np.array([5.0, 6.0]),
        5: np.array([5.0, 6.0]),
        6: np.array([7.0, 8.0]),
        7: np.array([7.0, 8.0]),
    }

    for k in rank_to_x_train:
        check_split(
            x_train,
            y_train,
            mp_size,
            dp_size,
            k,
            rank_to_x_train[k],
            rank_to_y_train[k],
        )


if __name__ == "__main__":
    test_mp_2_dp_1()
    test_mp_1_dp_2()
    test_mp_2_dp_2()
    test_mp_2_dp_4()
