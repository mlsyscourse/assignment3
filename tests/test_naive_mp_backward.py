from typing import Dict, List
from mpi4py import MPI

import numpy as np
import pytest

from model.func_impl import naive_collect_backward_output, naive_collect_backward_x


def check_naive_mp_backward_output(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    x = input_dict["input_x"]

    output = naive_collect_backward_output(
        output_grad=x,
        mp_size=input_dict["mp_size"],
        mp_group_idx=input_dict["mp_group_idx"],
    )

    assert x.dtype == output.dtype

    np.testing.assert_allclose(
        actual=output, desired=expect_output_dict["output_array"]
    )


def check_naive_mp_backward_x(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    x = input_dict["input_x"]

    output = naive_collect_backward_x(
        grad_x=x,
        mp_size=input_dict["mp_size"],
        mp_comm=input_dict["mp_comm"],
    )

    assert x.dtype == output.dtype

    np.testing.assert_allclose(
        actual=output, desired=expect_output_dict["output_array"]
    )


@pytest.mark.mpi
def test_fc2_naive_mp_backward_output_1d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(16).reshape((1, 16)).astype(np.float64)

    input_dict = {
        "input_x": array,
        "mp_group_idx": rank,
        "mp_size": 4,
    }

    output_array_list = {
        0: np.array(
            [
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                ]
            ]
        ),
        1: np.array(
            [
                [
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                ]
            ]
        ),
        2: np.array(
            [
                [
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                ]
            ]
        ),
        3: np.array(
            [
                [
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ]
            ]
        ),
    }

    expect_output_dict = {"output_array": output_array_list[rank]}

    check_naive_mp_backward_output(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_fc2_naive_mp_backward_output_2d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(16).reshape((2, 8)).astype(np.float64)

    input_dict = {
        "input_x": array,
        "mp_group_idx": rank,
        "mp_size": 4,
    }

    output_array_list = {
        0: np.array([[0.0, 1.0], [8.0, 9.0]]),
        1: np.array([[2.0, 3.0], [10.0, 11.0]]),
        2: np.array([[4.0, 5.0], [12.0, 13.0]]),
        3: np.array([[6.0, 7.0], [14.0, 15.0]]),
    }

    expect_output_dict = {"output_array": output_array_list[rank]}

    check_naive_mp_backward_output(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_fc2_naive_mp_backward_x_1d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(32).reshape((4, 8)).astype(np.float64)

    input_dict = {
        "input_x": array[[rank]],
        "mp_comm": comm,
        "mp_size": 4,
    }

    output_array_list = {
        0: np.array([[48.0, 52.0]]),
        1: np.array([[56.0, 60.0]]),
        2: np.array([[64.0, 68.0]]),
        3: np.array([[72.0, 76.0]]),
    }

    expect_output_dict = {"output_array": output_array_list[rank]}

    check_naive_mp_backward_x(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_fc2_naive_mp_backward_x_2d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(64).reshape((8, 8)).astype(np.float64)

    input_dict = {
        "input_x": array[rank * 2 : rank * 2 + 2],
        "mp_comm": comm,
        "mp_size": 4,
    }

    output_array_list = {
        0: np.array([[96.0, 100.0], [128.0, 132.0]]),
        1: np.array([[104.0, 108.0], [136.0, 140.0]]),
        2: np.array([[112.0, 116.0], [144.0, 148.0]]),
        3: np.array([[120.0, 124.0], [152.0, 156.0]]),
    }

    expect_output_dict = {"output_array": output_array_list[rank]}

    check_naive_mp_backward_x(input_dict, expect_output_dict)
