from typing import Dict, List
from mpi4py import MPI

import numpy as np
import pytest

from model.func_impl import collect_weight_grad


def check_dp_weight_comm(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    collected_grad_w, collected_grad_b = collect_weight_grad(
        grad_w=input_dict["input_grad_w"],
        grad_b=input_dict["input_grad_b"],
        dp_comm=input_dict["comm"],
    )

    assert input_dict["input_grad_w"].dtype == collected_grad_w.dtype
    assert input_dict["input_grad_b"].dtype == collected_grad_b.dtype

    np.testing.assert_allclose(
        actual=collected_grad_w, desired=expect_output_dict["grad_w"]
    )
    np.testing.assert_allclose(
        actual=collected_grad_b, desired=expect_output_dict["grad_b"]
    )


@pytest.mark.mpi
def test_collect_weight_grad_1():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    input_w = np.array(
        [
            [2.0, 4.0, 8.0, 9.0, 7.0, 1.0, 5.0, 8.0],
            [2.0, 0.0, 4.0, 7.0, 5.0, 7.0, 3.0, 0.0],
            [2.0, 2.0, 5.0, 7.0, 1.0, 9.0, 0.0, 9.0],
            [7.0, 9.0, 6.0, 4.0, 3.0, 5.0, 5.0, 3.0],
            [2.0, 5.0, 2.0, 6.0, 2.0, 2.0, 0.0, 7.0],
            [2.0, 8.0, 8.0, 0.0, 8.0, 0.0, 1.0, 3.0],
            [2.0, 0.0, 4.0, 7.0, 9.0, 3.0, 4.0, 7.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 6.0],
        ]
    )
    input_b = np.array(
        [
            [5.0, 5.0, 2.0, 9.0, 9.0, 8.0, 8.0, 3.0],
            [3.0, 8.0, 6.0, 8.0, 5.0, 4.0, 7.0, 3.0],
            [7.0, 0.0, 5.0, 7.0, 2.0, 8.0, 9.0, 5.0],
            [7.0, 4.0, 5.0, 6.0, 7.0, 4.0, 6.0, 5.0],
        ]
    )

    input_dict = {
        "input_grad_w": input_w[rank * 2 : rank * 2 + 2],
        "input_grad_b": input_b[[rank]],
        "comm": comm,
    }

    expect_output_dict = {
        "grad_w": np.array(
            [
                [8.0, 11.0, 19.0, 29.0, 19.0, 15.0, 9.0, 31.0],
                [12.0, 17.0, 18.0, 11.0, 17.0, 17.0, 9.0, 12.0],
            ]
        ),
        "grad_b": np.array([[22.0, 17.0, 18.0, 30.0, 23.0, 24.0, 30.0, 16.0]]),
    }

    check_dp_weight_comm(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_collect_weight_grad_2():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    input_w = np.array(
        [
            [5.0, 5.0, 7.0, 1.0, 0.0, 3.0, 3.0, 0.0],
            [1.0, 7.0, 5.0, 9.0, 6.0, 1.0, 5.0, 4.0],
            [6.0, 6.0, 6.0, 5.0, 2.0, 5.0, 5.0, 7.0],
            [4.0, 5.0, 3.0, 7.0, 5.0, 4.0, 1.0, 9.0],
            [1.0, 9.0, 3.0, 0.0, 1.0, 9.0, 4.0, 7.0],
            [3.0, 1.0, 6.0, 1.0, 3.0, 2.0, 8.0, 0.0],
            [0.0, 4.0, 4.0, 1.0, 9.0, 4.0, 7.0, 1.0],
            [3.0, 7.0, 6.0, 0.0, 6.0, 8.0, 5.0, 3.0],
            [8.0, 5.0, 2.0, 1.0, 5.0, 9.0, 7.0, 3.0],
            [1.0, 3.0, 1.0, 6.0, 2.0, 6.0, 5.0, 5.0],
            [1.0, 2.0, 5.0, 1.0, 1.0, 8.0, 9.0, 3.0],
            [3.0, 4.0, 1.0, 2.0, 7.0, 3.0, 7.0, 8.0],
            [4.0, 4.0, 3.0, 1.0, 4.0, 9.0, 8.0, 3.0],
            [4.0, 2.0, 2.0, 0.0, 8.0, 9.0, 6.0, 1.0],
            [7.0, 3.0, 9.0, 8.0, 1.0, 2.0, 9.0, 3.0],
            [1.0, 2.0, 1.0, 5.0, 4.0, 3.0, 6.0, 0.0],
        ]
    )
    input_b = np.array(
        [
            [1.0, 0.0, 9.0, 8.0, 2.0, 3.0, 6.0, 3.0],
            [7.0, 4.0, 8.0, 3.0, 7.0, 4.0, 9.0, 9.0],
            [2.0, 6.0, 1.0, 5.0, 2.0, 9.0, 9.0, 6.0],
            [3.0, 1.0, 0.0, 4.0, 7.0, 2.0, 8.0, 2.0],
        ]
    )

    input_dict = {
        "input_grad_w": input_w[rank * 4 : rank * 4 + 4],
        "input_grad_b": input_b[[rank]],
        "comm": comm,
    }

    expect_output_dict = {
        "grad_w": np.array(
            [
                [18.0, 23.0, 15.0, 3.0, 10.0, 30.0, 22.0, 13.0],
                [9.0, 13.0, 14.0, 16.0, 19.0, 18.0, 24.0, 10.0],
                [14.0, 15.0, 24.0, 15.0, 13.0, 19.0, 30.0, 14.0],
                [11.0, 18.0, 11.0, 14.0, 22.0, 18.0, 19.0, 20.0],
            ]
        ),
        "grad_b": np.array([[13.0, 11.0, 18.0, 20.0, 18.0, 18.0, 32.0, 20.0]]),
    }

    check_dp_weight_comm(input_dict, expect_output_dict)
