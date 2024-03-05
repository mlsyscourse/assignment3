from typing import Dict, List
from mpi4py import MPI

import numpy as np
import pytest

from model.func_impl import (
    megatron_collect_backward_output,
    megatron_collect_backward_x,
)


def check_megatron_mp_backward_output(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    x = input_dict["input_x"]

    output = megatron_collect_backward_output(
        output_grad=x,
        mp_size=input_dict["mp_size"],
        mp_group_idx=input_dict["mp_group_idx"],
    )

    assert x.dtype == output.dtype

    np.testing.assert_allclose(
        actual=output, desired=expect_output_dict["output_array"]
    )


def check_megatron_mp_backward_x(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    x = input_dict["input_x"]

    output = megatron_collect_backward_x(
        grad_x=x,
        mp_size=input_dict["mp_size"],
        mp_comm=input_dict["mp_comm"],
    )

    assert x.dtype == output.dtype

    np.testing.assert_allclose(
        actual=output, desired=expect_output_dict["output_array"]
    )


@pytest.mark.mpi
def test_fc2_megatron_mp_backward_output_1d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(16).reshape((1, 16)).astype(np.float64)

    input_dict = {
        "input_x": array,
        "mp_group_idx": rank,
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

    check_megatron_mp_backward_output(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_fc2_megatron_mp_backward_output_2d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(16).reshape((2, 8)).astype(np.float64)

    input_dict = {
        "input_x": array,
        "mp_group_idx": rank,
        "mp_size": 4,
    }

    expect_output_dict = {
        "output_array": np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            ]
        )
    }

    check_megatron_mp_backward_output(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_fc2_megatron_mp_backward_x_1d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(32).reshape((4, 8)).astype(np.float64)

    input_dict = {
        "input_x": array[[rank]],
        "mp_comm": comm,
        "mp_size": 4,
    }

    output_array_list = {
        0: np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            ]
        ),
        1: np.array(
            [
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            ]
        ),
        2: np.array(
            [
                [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
            ]
        ),
        3: np.array([[24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]]),
    }

    expect_output_dict = {"output_array": output_array_list[rank]}

    check_megatron_mp_backward_x(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_fc2_megatron_mp_backward_x_2d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(64).reshape((8, 8)).astype(np.float64)

    input_dict = {
        "input_x": array[rank * 2 : rank * 2 + 2],
        "mp_comm": comm,
        "mp_size": 4,
    }

    output_array_list = {
        0: np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            ]
        ),
        1: np.array(
            [
                [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                [24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
            ]
        ),
        2: np.array(
            [
                [32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
            ]
        ),
        3: np.array(
            [
                [48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
                [56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0],
            ]
        ),
    }

    expect_output_dict = {"output_array": output_array_list[rank]}

    check_megatron_mp_backward_x(input_dict, expect_output_dict)
