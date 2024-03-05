from typing import Dict, List
from mpi4py import MPI

import numpy as np
import pytest

from model.func_impl import naive_collect_forward_input, naive_collect_forward_output


def check_naive_mp_forward_x(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    x = input_dict["input_x"]

    output = naive_collect_forward_input(
        x=x,
        mp_size=input_dict["mp_size"],
        mp_comm=input_dict["mp_comm"],
    )

    assert x.dtype == output.dtype

    np.testing.assert_allclose(
        actual=output, desired=expect_output_dict["output_array"]
    )


def check_naive_mp_forward_output(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    x = input_dict["input_x"]

    output = naive_collect_forward_output(
        out=x,
        mp_size=input_dict["mp_size"],
        mp_comm=input_dict["mp_comm"],
    )

    assert x.dtype == output.dtype

    np.testing.assert_allclose(
        actual=output, desired=expect_output_dict["output_array"]
    )


@pytest.mark.mpi
def test_fc2_naive_mp_forward_x_1d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(16).reshape((4, 4)).astype(np.float64)

    input_dict = {
        "input_x": array[[rank]],
        "mp_comm": comm,
        "mp_size": 4,
    }

    expect_output_dict = {
        "output_array": np.array(
            [
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ]
            ]
        )
    }

    check_naive_mp_forward_x(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_fc2_naive_mp_forward_x_2d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(16).reshape((8, 2)).astype(np.float64)

    input_dict = {
        "input_x": array[rank * 2 : rank * 2 + 2],
        "mp_comm": comm,
        "mp_size": 4,
    }

    expect_output_dict = {
        "output_array": np.array(
            [
                [0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0],
                [2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0],
            ]
        )
    }

    check_naive_mp_forward_x(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_fc2_naive_mp_forward_output_1d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(16).reshape((4, 4)).astype(np.float64)

    input_dict = {
        "input_x": array[[rank]],
        "mp_comm": comm,
        "mp_size": 4,
    }

    expect_output_dict = {
        "output_array": np.array(
            [
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ]
            ]
        )
    }

    check_naive_mp_forward_output(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_fc2_naive_mp_forward_output_2d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(16).reshape((8, 2)).astype(np.float64)

    input_dict = {
        "input_x": array[rank * 2 : rank * 2 + 2],
        "mp_comm": comm,
        "mp_size": 4,
    }

    expect_output_dict = {
        "output_array": np.array(
            [
                [0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0],
                [2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0],
            ]
        )
    }

    check_naive_mp_forward_output(input_dict, expect_output_dict)
